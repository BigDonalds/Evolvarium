"""The incident: an intruder working its way through the estate, and the security
team working its case queue.

A step is half an hour, and plays out in a fixed order. The analysts' responses
take effect. Rebuilds and investigations that are due finish. The intruder
comes back in if it was thrown out, its command channel's traffic may raise
alerts, and it carries on with what it was doing or picks something new. Alerts
that are due arrive, false ones among them, and old ones expire. Downtime and
the intruder's time on hosts are charged. Then the incident may end, and if it
goes on the queue is ranked and each free analyst is handed a host.

An incident is a function of its seed and the analysts' responses. It draws
from five random streams of its own, so the estate and the false alerts are the
same whatever the defence does, and an intruder played by a network draws
nothing from the stream the scripted one decides with.
"""

from __future__ import annotations

import math
import random
from collections import deque
from enum import Enum

from .estate import (
    APP,
    CROWN_PATH,
    DB,
    DB_SERVICE,
    DC,
    DOMAIN_ADMIN,
    IT_ADMIN,
    LOCAL_ADMIN,
    PRIVILEGED,
    RANSOM_AIM,
    STEAL_AIM,
    VALUE,
    WEB,
    WEB_SERVICE,
    WORKSTATION,
    WORKSTATIONS,
    ZONE,
    Estate,
    password,
)
from .intruder import PROFILES, Policy, scripted
from .settings import CostConfig, CyberSettings

RESPONSES = ("watch", "dismiss", "investigate", "isolate", "release", "reimage", "reset", "block")

# What an analyst sees, in order. See `Incident.view`.
INPUTS = (
    "value", "crown_path", "privileged", "managed", "workstation", "rebuild_cost", "noisy",
    "risk", "severity", "age", "fresh", "lead",
    "isolated", "isolated_for", "clean", "confirmed", "confirmed_admin", "unreviewed",
    "exposed", "zone_risk",
    "time", "free", "open_cases", "alert_rate", "contained", "confirmed_hosts",
    "c2_known", "since_block", "reset_ready", "admin_found", "slot",
)
# An analyst busy with an investigation sees nothing, and its response is ignored.
BUSY_VIEW = (0.0,) * len(INPUTS)

LOW, MEDIUM, HIGH = 1, 2, 3
# What each severity adds to a host's risk.
_RISK = (0, 1, 3, 6)
# Severity, and whether only an endpoint agent sees it, by technique.
_TECHNIQUES = {
    "entry_phish": (LOW, True), "entry_web": (MEDIUM, False), "entry_vpn": (MEDIUM, False),
    "scan": (LOW, False), "escalate": (MEDIUM, True), "dump": (HIGH, True),
    "logon": (MEDIUM, False), "exploit": (HIGH, False), "exfil": (MEDIUM, False),
    "stage": (HIGH, True), "beacon": (LOW, False),
}
# Shares of false alerts by severity.
_FALSE_LOW, _FALSE_MEDIUM = 0.65, 0.93


class EndCause(str, Enum):
    RANSOMED = "ransomed"
    BREACHED = "breached"
    EVICTED = "evicted"
    TIMEOUT = "timeout"


class Incident:
    """A single incident.

    `intruder` plays the intruder in place of the seed's scripted one. It is a
    policy over the intruder's declared actions, asked only when the intruder
    is free to act.
    """

    def __init__(self, settings: CyberSettings, seed: int, intruder: Policy | None = None):
        self.settings = settings
        costs, soc = settings.costs, settings.soc
        detect = random.Random(seed * 5)
        estate_rng = random.Random(seed * 5 + 1)
        self._rolls = random.Random(seed * 5 + 2)
        self._choices = random.Random(seed * 5 + 3)
        noise = random.Random(seed * 5 + 4)
        self._detect = detect

        self.estate = estate = Estate(settings, estate_rng)
        size = self.size = estate.size
        roles = estate.roles
        self.flaws = list(estate.flaws)
        self.rebuild_steps = tuple(
            soc.reimage_workstation if role in WORKSTATIONS
            else soc.reimage_core if role in (DC, DB) else soc.reimage_server
            for role in roles
        )
        self.down_rate = tuple(_down_rate(role, costs) for role in roles)
        self.dwell_rate = tuple(
            costs.dwell_workstation if role in WORKSTATIONS
            else costs.dwell_core if role in (DC, DB) else costs.dwell_server
            for role in roles
        )
        self.reimage_cost = tuple(
            costs.reimage + (costs.reimage_workstation if role in WORKSTATIONS else 0.0)
            for role in roles
        )

        # The intruder, drawn whatever the settings fix, so fixing one draw
        # leaves the others as they were.
        config = settings.intruder
        rolls = self._rolls
        u = rolls.random()
        drawn = ("careful" if u < config.careful_share
                 else "smash" if u < config.careful_share + config.smash_share else "operator")
        self.profile = drawn if config.profile == "mixed" else config.profile
        drawn = "ransom" if rolls.random() < config.ransom_share else "steal"
        self.objective = drawn if config.objective == "mixed" else config.objective
        self.start = rolls.randint(config.start_min, config.start_max)
        # An intruder after the domain controller comes in through the web server
        # only if that way could lead there.
        self.web_entry = (self.flaws[estate.web] and rolls.random() >= config.phish_entry
                          and (self.objective == "steal" or estate.closure(estate.web, 2)[1]))
        self.entries = rolls.randint(config.re_entries_min, config.re_entries_max)
        self.policy = intruder or scripted(PROFILES[self.profile])
        self.aims = RANSOM_AIM if self.objective == "ransom" else STEAL_AIM

        # Where the intruder is and what it holds.
        self.access = [0] * size
        self.dumped = [False] * size
        self.scanned = [False] * size
        self.known: set[int] = set()
        self.found_at = [0] * size
        self.activity = [0] * size
        self.careful_on = [False] * size
        self.retry_at = [0] * size
        self.creds = 0
        self.busy: tuple | None = None
        self.channel_up = False
        self.channel = 0
        self.switch_at = -1
        self.reentry_at = self.start
        self.reentry_kind = "first"
        # Moves that landed, as (from, to), newest last.
        self.moves: list[tuple[int, int]] = []
        self.stolen = 0.0
        self.staged = 0
        self.ever: set[int] = set()
        self.losses: list[int] = []
        self._usable_count = 0
        self.first_entry = -1

        # The security team's side.
        self.isolated = [False] * size
        self.isolated_at = [0] * size
        self.rebuilding_until = [0] * size
        self.alerts: list[list[tuple[int, int, bool]]] = [[] for _ in range(size)]
        self.clean = [False] * size
        self.confirmed = [False] * size
        self.confirmed_admin = [False] * size
        self.investigated_until = [0] * size
        self.reviewed = [0] * size
        self.analyst_busy_until = [0] * soc.analysts
        self.slots: list[int | None] = [None] * soc.analysts
        self._pending: dict[int, list[tuple[int, int, bool, bool]]] = {}
        self.c2_known = -1
        self.last_block = -1_000
        self.reset_at = -1_000
        self.admin_found = False

        self.t = 0
        self.costs = {"data": 0.0, "ransom": 0.0, "downtime": 0.0, "dwell": 0.0,
                      "effort": 0.0, "unresolved": 0.0}
        self.finished = False
        self.end: EndCause | None = None
        # Steps alerts arrived on, over the last 8.
        self._arrivals: deque[int] = deque()
        self.true_alerts = self.false_alerts = 0
        self.investigations = self.findings = self.missed = 0
        self.isolations = self.false_isolations = 0
        self.reimages = self.false_reimages = self.cleaned = 0
        self.resets = self.blocks = 0
        self.encrypted: list[int] = []
        self.detected_at = -1
        self.footholds_peak = 0
        self.dwell_steps = 0
        # What this step did, for the scene.
        self.acted: list[tuple[str, int]] = []
        self.exfil_from = -1

        self._false: dict[int, list[tuple[int, int]]] = {}
        horizon = settings.incident.max_steps
        for h in range(size):
            rate = estate.noise[h]
            step = 0
            while rate > 0:
                step += 1 + int(math.log(1.0 - noise.random()) / math.log(1.0 - rate))
                if step > horizon:
                    break
                u = noise.random()
                severity = LOW if u < _FALSE_LOW else MEDIUM if u < _FALSE_MEDIUM else HIGH
                self._false.setdefault(step, []).append((h, severity))
        self._rank()

    # ------------------------------------------------------------ the estate

    @property
    def score(self) -> float:
        """The incident's cost so far."""
        return round(sum(self.costs.values()), 1)

    def in_service(self, h: int) -> bool:
        return not self.isolated[h] and not self.rebuilding_until[h]

    def relay(self) -> int:
        """The host data leaves the database through, or -1.

        The database reaches nothing outside, so the intruder needs a host it
        moved onto the database from, still usable, to carry its traffic.
        """
        db = self.estate.db
        for source, target in reversed(self.moves):
            if target == db and source != db and self._usable_here(source):
                return source
        return -1

    def _usable_here(self, h: int) -> bool:
        return (self.access[h] > 0 and self.channel_up and not self.isolated[h]
                and not self.rebuilding_until[h])

    def usable(self, h: int) -> bool:
        """Whether the intruder can work from `h` now."""
        if not self._usable_here(h):
            return False
        return h != self.estate.db or self.relay() >= 0

    def usable_hosts(self) -> list[int]:
        return [h for h in range(self.size) if self.access[h] and self.usable(h)]

    # ------------------------------------------------------------ alerts

    def _raise(self, h: int, technique: str, careful: bool = False) -> None:
        severity, endpoint = _TECHNIQUES[technique]
        detection = self.settings.detection
        chance = getattr(detection, technique)
        if careful:
            chance *= detection.careful
        if endpoint and not self.estate.managed[h]:
            chance *= detection.no_agent
        if self._detect.random() >= chance:
            return
        due = self.t + self._detect.randint(0, detection.delay_max)
        self._pending.setdefault(due, []).append((h, severity, False, True))

    # ------------------------------------------------------------ the intruder

    def _intruder_phase(self) -> None:
        t, config = self.t, self.settings.intruder
        rolls = self._rolls
        if self.switch_at == t:
            self.channel_up = True
            self.channel += 1
            self.switch_at = -1
        usable = self.usable_hosts()
        if len(usable) < self._usable_count:
            self.losses.append(t)
        self._usable_count = len(usable)

        held = any(self.access)
        if ((self.channel_up or not held) and not usable and self.reentry_at < 0
                and self.first_entry >= 0):
            if self._vpn_targets():
                self.reentry_at = t + rolls.randint(config.vpn_delay_min, config.vpn_delay_max)
                self.reentry_kind = "vpn"
            elif self.entries > 0:
                self.entries -= 1
                self.reentry_at = t + rolls.randint(config.re_entry_delay_min,
                                                    config.re_entry_delay_max)
                self.reentry_kind = "phish"
        if 0 <= self.reentry_at <= t:
            self._enter()
            return
        # An action whose host was cut off, or whose target was, is given up.
        if self.busy is not None:
            _, source, target, _, _ = self.busy
            if not self.usable(source) or (target is not None and not self.in_service(target)):
                self.busy = None
                self.losses.append(t)
        if not usable:
            return

        for h in usable:
            if h != self.estate.db:
                self._raise(h, "beacon")
        if self.busy is None:
            self._begin(self.policy(self.intruder_view(usable), self._choices), usable)
        if self.busy is None:
            return
        tactic, source, target, left, careful = self.busy
        if left > 1:
            self.busy = (tactic, source, target, left - 1, careful)
            return
        self.busy = None
        self._finish(tactic, source, target, careful)
        self._usable_count = len(self.usable_hosts())

    def _vpn_targets(self) -> list[int]:
        """Workstations whose user's password the intruder holds, that it can log on to."""
        return [h for h in self.estate.workstations
                if self.creds & password(h) and self.in_service(h)]

    def _enter(self) -> None:
        """The first entry, or a way back in."""
        estate, rolls = self.estate, self._rolls
        kind = self.reentry_kind
        self.reentry_at = -1
        web = estate.web
        if kind == "first" and self.web_entry and self.flaws[web] and self.in_service(web):
            host, level, technique = web, 2, "entry_web"
        elif kind == "vpn":
            targets = self._vpn_targets()
            if not targets:
                # Locked out of every account it holds: the next phase decides again.
                return
            host, level, technique = rolls.choice(targets), 1, "entry_vpn"
        else:
            targets = [h for h in estate.users if self.in_service(h)]
            if not targets:
                self.reentry_at = self.t + 1
                return
            host, level, technique = rolls.choice(targets), 1, "entry_phish"
            self.creds |= password(host)
        self.access[host] = max(self.access[host], level)
        self._raise(host, technique)
        if self.first_entry < 0:
            self.first_entry = self.t
        self._learn(host)
        # A machine joined to the domain knows where its domain controller is.
        if estate.roles[host] in WORKSTATIONS:
            self._learn(estate.dc)
        self.ever.add(host)
        if not self.channel_up:
            self.channel_up = True
            self.channel += 1
            self.switch_at = -1
        self.busy = None
        self._usable_count = len(self.usable_hosts())

    def _learn(self, h: int) -> None:
        if h not in self.known:
            self.known.add(h)
            self.found_at[h] = self.t

    def objective_hosts(self) -> set[int]:
        estate = self.estate
        if self.objective == "ransom":
            return {estate.dc}
        hosts = {estate.db}
        if estate.backup >= 0 and estate.backup in self.known:
            hosts.add(estate.backup)
        return hosts

    def _act_host(self, usable: list[int]) -> int:
        estate = self.estate
        for h in usable:
            if self.objective == "ransom":
                if h == estate.dc and self.access[h] == 2:
                    return h
            elif h == estate.db or (h == estate.backup and self.access[h] == 2):
                return h
        return -1

    def _candidate_moves(self, usable: list[int], careful: bool) -> list[tuple[int, int, str]]:
        estate = self.estate
        # A database it holds but has lost the way out of can be reached again.
        cut_off = self.access[estate.db] > 0 and self.relay() < 0
        found = []
        for source in usable:
            for target in estate.reach[source]:
                if target not in self.known or not self.in_service(target):
                    continue
                if self.access[target] and not (cut_off and target == estate.db):
                    continue
                if self.creds & (estate.admin_by[target] | estate.user_by[target]):
                    found.append((source, target, "credential"))
                elif not careful and self.flaws[target] and estate.roles[target] != DB:
                    found.append((source, target, "exploit"))
        return found

    def intruder_view(self, usable: list[int] | None = None) -> list[float]:
        """What the intruder sees, in the order of `INTRUDER_INPUTS`."""
        if usable is None:
            usable = self.usable_hosts()
        t, size = self.t, self.size
        held = [h for h in range(size) if self.access[h]]
        admin = sum(1 for h in usable if self.access[h] == 2)
        moves = self._candidate_moves(usable, False)
        can_move = (1.0 if any(m[2] == "credential" for m in moves)
                    else 0.5 if moves else 0.0)
        objective = self.objective_hosts()
        toward = [m for m in moves if m[1] in objective]
        in_reach = (1.0 if any(m[2] == "credential" for m in toward)
                    else 0.5 if toward else 0.0)
        ransom = self.objective == "ransom"
        stage_steps = self.settings.intruder.stage_steps
        return [
            t / self.settings.incident.max_steps,
            min(1.0, len(usable) / 4),
            min(1.0, admin / 4),
            sum(1 for h in held if self.isolated[h]) / len(held) if held else 0.0,
            min(1.0, sum(1 for step in self.losses if step > t - 6) / 3),
            1.0 if self.busy else 0.0,
            1.0 if any(not self.scanned[h] for h in usable) else 0.0,
            1.0 if any(self.access[h] == 1 and h != self.estate.db and self.retry_at[h] <= t
                       for h in usable) else 0.0,
            1.0 if any(self.access[h] == 2 and not self.dumped[h] for h in usable) else 0.0,
            can_move,
            in_reach,
            1.0 if self._act_host(usable) >= 0 else 0.0,
            1.0 if self.creds & LOCAL_ADMIN else 0.0,
            1.0 if self.creds & IT_ADMIN else 0.0,
            1.0 if self.creds & (DB_SERVICE | WEB_SERVICE) else 0.0,
            1.0 if self.creds & DOMAIN_ADMIN else 0.0,
            1.0 if ransom else 0.0,
            self.staged / stage_steps if ransom else self.stolen,
            1.0 if self.channel_up else 0.0,
            min(1.0, self.entries / 3),
            len(self.known) / size,
        ]

    def _pick(self, hosts: list[int], aim: str) -> int:
        rolls = self._rolls
        if aim == "objective":
            return max(hosts, key=lambda h: (self.aims[self.estate.roles[h]], rolls.random()))
        if aim == "quiet":
            return min(hosts, key=lambda h: (self.activity[h], rolls.random()))
        return max(hosts, key=lambda h: (self.found_at[h], rolls.random()))

    def _begin(self, action: dict, usable: list[int]) -> None:
        tactic, aim = action["tactic"], action["aim"]
        careful = action["tempo"] == "careful"
        slow = 2 if careful else 1
        t, estate = self.t, self.estate
        if tactic == "scan":
            hosts = [h for h in usable if not self.scanned[h]]
            if hosts:
                self.busy = ("scan", self._pick(hosts, aim), None, slow, careful)
        elif tactic == "escalate":
            hosts = [h for h in usable
                     if self.access[h] == 1 and h != estate.db and self.retry_at[h] <= t]
            if hosts:
                self.busy = ("escalate", self._pick(hosts, aim), None, 2 * slow, careful)
        elif tactic == "dump":
            hosts = [h for h in usable if self.access[h] == 2 and not self.dumped[h]]
            if hosts:
                self.busy = ("dump", self._pick(hosts, aim), None, slow, careful)
        elif tactic == "move":
            moves = self._candidate_moves(usable, careful)
            if not moves:
                return
            rolls, roles = self._rolls, estate.roles
            if aim == "objective":
                objective = self.objective_hosts()
                chosen = max(moves, key=lambda m: (m[1] in objective, self.aims[roles[m[1]]],
                                                   m[2] == "credential", rolls.random()))
            elif aim == "quiet":
                chosen = max(moves, key=lambda m: (m[2] == "credential", self.aims[roles[m[1]]],
                                                   -self.activity[m[0]], rolls.random()))
            else:
                zones = {ZONE[roles[h]] for h in range(self.size) if self.access[h]}
                chosen = max(moves, key=lambda m: (ZONE[roles[m[1]]] not in zones,
                                                   self.found_at[m[1]], rolls.random()))
            source, target, method = chosen
            steps = 1 if method == "credential" else 2
            self.busy = (method, source, target, steps * slow, careful)
        elif tactic == "act":
            host = self._act_host(usable)
            if host >= 0:
                self.busy = ("act", host, None, slow, careful)

    def _finish(self, tactic: str, source: int, target: int | None, careful: bool) -> None:
        estate, rolls, config = self.estate, self._rolls, self.settings.intruder
        self.activity[source] += 1
        self.careful_on[source] = careful
        if tactic == "scan":
            self.scanned[source] = True
            for h in estate.reach[source]:
                self._learn(h)
            self._raise(source, "scan", careful)
        elif tactic == "escalate":
            chance = (config.escalate_workstation if estate.roles[source] in WORKSTATIONS
                      else config.escalate_server)
            if rolls.random() < chance:
                self.access[source] = 2
            else:
                self.retry_at[source] = self.t + config.escalate_retry
            self._raise(source, "escalate", careful)
        elif tactic == "dump":
            self.dumped[source] = True
            self.creds |= estate.cached[source]
            self._raise(source, "dump", careful)
        elif tactic in ("credential", "exploit"):
            if tactic == "exploit":
                self._raise(target, "exploit", careful)
                if rolls.random() >= config.exploit:
                    return
                level = 2
            else:
                self._raise(target, "logon", careful)
                # A reset while the move was under way leaves nothing to log on with.
                if self.creds & estate.admin_by[target]:
                    level = 2
                elif self.creds & estate.user_by[target]:
                    level = 1
                else:
                    return
            if level > self.access[target]:
                self.access[target] = level
                self.ever.add(target)
                self.moves.append((source, target))
                self.activity[target] += 1
                self.careful_on[target] = careful
            elif target == estate.db and self.relay() < 0:
                self.moves.append((source, target))
        elif tactic == "act":
            if self.objective == "steal":
                carrier = self.relay() if source == estate.db else source
                self.exfil_from = carrier
                self.stolen = min(1.0, self.stolen + 1.0 / config.exfil_steps)
                self.costs["data"] += self.settings.costs.data / config.exfil_steps
                self._raise(carrier, "exfil", careful)
            else:
                self.staged += 1
                self._raise(source, "stage", careful)
                if self.staged >= config.stage_steps:
                    self._detonate()

    def _detonate(self) -> None:
        estate = self.estate
        self.encrypted = [h for h in range(self.size) if self.in_service(h)]
        # Backups the intruder holds are gone too, so recovery takes twice as long.
        lost_backups = estate.backup >= 0 and self.access[estate.backup] == 2
        rebuild = sum(self.rebuild_steps[h] * self.down_rate[h] for h in self.encrypted)
        self.costs["ransom"] += self.settings.costs.ransom + rebuild * (2 if lost_backups else 1)
        self.finished = True
        self.end = EndCause.RANSOMED

    # ------------------------------------------------------------ the team

    def risk(self, h: int) -> int:
        return sum(_RISK[severity] for severity, _, _ in self.alerts[h])

    def _rank(self) -> None:
        """Ranks the case queue and hands each free analyst a host."""
        t, size = self.t, self.size
        cases, quiet = [], []
        for h in range(size):
            if self.rebuilding_until[h] or self.investigated_until[h] > t:
                continue
            if self.alerts[h] or self.confirmed[h]:
                priority = self.risk(h)
                if self.confirmed[h]:
                    priority += 1 if self.isolated[h] else 6
                newest = self.alerts[h][-1][1] if self.alerts[h] else -1
                cases.append((-priority, -newest, h))
            else:
                quiet.append((0 if self.isolated[h] else 1, self.reviewed[h], h))
        cases.sort()
        quiet.sort()
        order = [h for _, _, h in cases] + [h for _, _, h in quiet]
        self.open_cases = len(cases)
        handed = 0
        self.positions = [0] * len(self.slots)
        for k in range(len(self.slots)):
            if self.analyst_busy_until[k] > t or handed >= len(order):
                self.slots[k] = None
            else:
                self.slots[k] = order[handed]
                self.positions[k] = handed
                handed += 1
        self._share()

    def _share(self) -> None:
        """The inputs about the incident as a whole, which every analyst sees."""
        t, size = self.t, self.size
        soc, incident = self.settings.soc, self.settings.incident
        while self._arrivals and self._arrivals[0] <= t - 8:
            self._arrivals.popleft()
        free = sum(1 for until in self.analyst_busy_until if until <= t)
        contained = sum(1 for h in range(size) if not self.in_service(h))
        confirmed = sum(1 for h in range(size) if self.confirmed[h])
        self._shared = [
            t / incident.max_steps,
            free / soc.analysts,
            min(1.0, self.open_cases / 8),
            min(1.0, len(self._arrivals) / 8),
            contained / size,
            min(1.0, confirmed / 4),
            1.0 if self.c2_known >= 0 else 0.0,
            min(1.0, (t - self.last_block) / 24),
            1.0 if t - self.reset_at >= soc.reset_cooldown else 0.0,
            1.0 if self.admin_found else 0.0,
        ]

    def view(self, analyst: int) -> list[float] | tuple[float, ...]:
        """What analyst number `analyst` sees of the host it holds, in the order of `INPUTS`."""
        h = self.slots[analyst]
        if h is None:
            return BUSY_VIEW
        t, estate = self.t, self.estate
        role = estate.roles[h]
        alerts = self.alerts[h]
        zone = ZONE[role]
        zone_risk = sum(self.risk(o) for o in range(self.size)
                        if o != h and ZONE[estate.roles[o]] == zone)
        exposed = any(self.confirmed[c] and self.in_service(c) for c in estate.reached_by[h])
        rebuild_cost = self.reimage_cost[h] + self.rebuild_steps[h] * self.down_rate[h]
        return [
            VALUE[role],
            1.0 if CROWN_PATH[role] else 0.0,
            1.0 if estate.privileged[h] else 0.0,
            1.0 if estate.managed[h] else 0.0,
            1.0 if role in WORKSTATIONS else 0.0,
            min(1.0, rebuild_cost / 12),
            min(1.0, estate.noise[h] / 0.12),
            min(1.0, self.risk(h) / 12),
            max((a[0] for a in alerts), default=0) / 3,
            min(1.0, (t - alerts[0][1]) / 24) if alerts else 0.0,
            1.0 if alerts and alerts[-1][1] >= t - 1 else 0.0,
            1.0 if any(a[2] for a in alerts) else 0.0,
            1.0 if self.isolated[h] else 0.0,
            min(1.0, (t - self.isolated_at[h]) / 24) if self.isolated[h] else 0.0,
            1.0 if self.clean[h] else 0.0,
            1.0 if self.confirmed[h] else 0.0,
            1.0 if self.confirmed_admin[h] else 0.0,
            min(1.0, (t - self.reviewed[h]) / 24),
            1.0 if exposed else 0.0,
            min(1.0, zone_risk / 12),
            *self._shared,
            self.positions[analyst] / len(self.slots),
        ]

    def _respond(self, responses: list[str]) -> None:
        t, soc, costs = self.t, self.settings.soc, self.settings.costs
        self.acted = []
        for k, response in enumerate(responses):
            h = self.slots[k]
            if h is None or self.analyst_busy_until[k] > t:
                continue
            self.reviewed[h] = t
            if response == "dismiss":
                self.alerts[h] = []
            elif response == "investigate":
                if self.investigated_until[h] <= t:
                    self.investigated_until[h] = self.analyst_busy_until[k] = (
                        t + soc.investigate_steps)
                    self.costs["effort"] += costs.investigate
                    self.investigations += 1
                    self.acted.append(("investigate", h))
            elif response == "isolate":
                if not self.isolated[h]:
                    self.isolated[h] = True
                    self.isolated_at[h] = t
                    self.costs["effort"] += costs.isolate
                    self.isolations += 1
                    if not self.access[h]:
                        self.false_isolations += 1
                    self.acted.append(("isolate", h))
            elif response == "release":
                if self.isolated[h]:
                    self.isolated[h] = False
                    self.acted.append(("release", h))
            elif response == "reimage":
                self._reimage(h)
            elif response == "reset":
                self._reset()
            elif response == "block":
                self._block()

    def _reimage(self, h: int) -> None:
        estate = self.estate
        # Out of service from the next step for its whole rebuild.
        self.rebuilding_until[h] = self.t + self.rebuild_steps[h] + 1
        self.isolated[h] = False
        self.alerts[h] = []
        self.clean[h] = self.confirmed[h] = self.confirmed_admin[h] = False
        self.costs["effort"] += self.reimage_cost[h]
        self.reimages += 1
        if self.access[h]:
            self.cleaned += 1
        else:
            self.false_reimages += 1
        self.access[h] = 0
        self.dumped[h] = self.scanned[h] = self.careful_on[h] = False
        self.flaws[h] = False
        if estate.roles[h] in WORKSTATIONS:
            self.creds &= ~password(h)
        if h == estate.dc:
            self.staged = 0
        self.moves = [move for move in self.moves if h not in move]
        self.acted.append(("reimage", h))

    def _reset(self) -> None:
        """Every credential the intruder stole stops working. Its implants stay,
        and it can take fresh ones from hosts it holds as admin."""
        t, soc = self.t, self.settings.soc
        if t - self.reset_at < soc.reset_cooldown:
            return
        self.reset_at = t
        self.creds = 0
        for h in range(self.size):
            if self.access[h] == 2:
                self.dumped[h] = False
        self.costs["effort"] += self.settings.costs.reset
        self.admin_found = False
        self.resets += 1
        self.acted.append(("reset", self.estate.dc))

    def _block(self) -> None:
        if self.c2_known < 0:
            return
        config = self.settings.intruder
        if self.c2_known == self.channel and self.channel_up:
            self.channel_up = False
            self.switch_at = self.t + 1 + self._rolls.randint(config.channel_switch_min,
                                                              config.channel_switch_max)
        self.c2_known = -1
        self.last_block = self.t
        self.costs["effort"] += self.settings.costs.block
        self.blocks += 1
        self.acted.append(("block", -1))

    def _find(self, h: int) -> None:
        """An investigation of `h` ends."""
        soc = self.settings.soc
        chance = soc.find
        if self.careful_on[h]:
            chance *= soc.find_careful
        if not self.estate.managed[h]:
            chance *= soc.find_unmanaged
        if self.access[h] and self._detect.random() < chance:
            self.confirm(h)
        else:
            if self.access[h]:
                self.missed += 1
            self.clean[h] = True
            self.alerts[h] = []

    def confirm(self, h: int) -> None:
        """Marks the intruder found on `h`, as an investigation that finds it does.

        The command channel's address comes out of the host, and every host a
        landed move links it to raises a lead. The truth-reading bounds call this
        directly; the analysts get here only by investigating.
        """
        t = self.t
        self.confirmed[h] = True
        self.confirmed_admin[h] = self.access[h] == 2
        self.findings += 1
        if self.detected_at < 0:
            self.detected_at = t - self.first_entry
        if self.channel_up:
            self.c2_known = self.channel
        if self.access[h] == 2:
            self.admin_found = True
        for source, target in self.moves:
            other = target if source == h else source if target == h else -1
            if other >= 0:
                self._pending.setdefault(t, []).append((other, HIGH, True, True))
        # A finding made outside a step shows in every view at once.
        self._share()

    # ------------------------------------------------------------ a step

    def step(self, responses: list[str]) -> None:
        """Play one step, given a response from each analyst in order."""
        self._respond(responses)
        self.t += 1
        t = self.t
        self.exfil_from = -1
        for h in range(self.size):
            if self.rebuilding_until[h] and self.rebuilding_until[h] <= t:
                self.rebuilding_until[h] = 0
            if self.investigated_until[h] == t:
                self._find(h)
        self._intruder_phase()
        if self.finished:
            return
        for h, severity in self._false.get(t, ()):
            self._pending.setdefault(t, []).append((h, severity, False, False))
        # Leads from a finding made outside a step are due before it, and arrive now.
        arriving = []
        for due in sorted(step for step in self._pending if step <= t):
            arriving.extend(self._pending.pop(due))
        for h, severity, lead, true in arriving:
            if self.rebuilding_until[h]:
                continue
            self.alerts[h].append((severity, t, lead))
            self._arrivals.append(t)
            self.clean[h] = False
            if true:
                self.true_alerts += 1
            else:
                self.false_alerts += 1
        expire = self.settings.detection.expire_after
        footholds = 0
        for h in range(self.size):
            alerts = self.alerts[h]
            if alerts and not self.isolated[h] and t - alerts[0][1] >= expire:
                self.alerts[h] = [a for a in alerts if t - a[1] < expire]
            if not self.in_service(h):
                self.costs["downtime"] += self.down_rate[h]
            if self.access[h]:
                self.costs["dwell"] += self.dwell_rate[h]
                footholds += 1
        self.dwell_steps += footholds
        self.footholds_peak = max(self.footholds_peak, footholds)
        self._end_checks()
        if not self.finished:
            self._rank()

    def _end_checks(self) -> None:
        if self.stolen >= 1.0 - 1e-9:
            self.finished = True
            self.end = EndCause.BREACHED
        elif (self.settings.incident.end_on_eviction and self.first_entry >= 0
              and not any(self.access) and self.creds == 0 and self.entries == 0
              and self.reentry_at < 0 and all(self.in_service(h) for h in range(self.size))):
            self.finished = True
            self.end = EndCause.EVICTED
        elif self.t >= self.settings.incident.max_steps:
            self._timeout()

    def _timeout(self) -> None:
        self.finished = True
        self.end = EndCause.TIMEOUT
        if any(self.access) or self.creds & PRIVILEGED:
            self.costs["unresolved"] += self.settings.costs.unresolved

    def stop(self) -> None:
        """Ends the incident where it stands, as if its time had run out."""
        if not self.finished:
            self._timeout()

    @property
    def lost(self) -> bool:
        return self.end in (EndCause.BREACHED, EndCause.RANSOMED)

    @property
    def footholds(self) -> int:
        return sum(1 for h in range(self.size) if self.access[h])


def _down_rate(role: int, costs: CostConfig) -> float:
    """What a step out of service costs for a host of `role`.

    The IT administrator's workstation is costed as a server, since the whole
    IT team works through it.
    """
    if role == DC:
        return costs.down_dc
    if role == DB:
        return costs.down_database
    if role in (WEB, APP):
        return costs.down_frontline
    if role == WORKSTATION:
        return costs.down_workstation
    return costs.down_server
