"""Cyber defence, described to the engine through the game contract."""

from __future__ import annotations

import dataclasses

from ...contract import (
    Actions,
    Choice,
    Game,
    Label,
    Layer,
    Milestone,
    Observation,
    Problems,
    Scene,
)
from .engine import INPUTS, RESPONSES, EndCause, Incident
from .estate import DB, DC, OPTIONAL_SERVERS, WEB, WORKSTATIONS
from .settings import CyberSettings

OBSERVATION = Observation(INPUTS)
ACTIONS = Actions(Choice("response", RESPONSES))

# Where each host is drawn, by its place in the network. Workstations fill three
# columns from the IT administrator's, and servers one column in host order.
_WEB = (6, 1)
_DC = (18, 6)
_DB = (22, 6)
_SERVER_SLOTS = ((12, 2), (12, 4), (12, 6), (12, 8), (12, 10))
_C2 = (1.5, 7.0)
_ZONES = (
    (5.0, 0.25, 8.0, 2.75),
    (3.5, 5.25, 9.5, 13.5),
    (11.25, 1.25, 13.75, 11.25),
    (17.25, 5.25, 19.75, 7.75),
    (21.25, 5.25, 23.75, 7.75),
)
_FLOWS = (
    (1.5, 7.0, 6.0, 1.5), (1.5, 7.0, 3.5, 9.0), (7.0, 1.5, 12.0, 2.5),
    (9.5, 9.0, 11.25, 9.0), (13.75, 6.5, 17.25, 6.5), (13.75, 3.0, 21.25, 5.5),
    (19.75, 6.5, 21.25, 6.5),
)
# Moves drawn, newest last.
_MOVES_SHOWN = 12
# The layers from the hosts' frames up to the intruder are the incident's truth,
# which the analysts never see: the scene shows it for whoever watches.
_LAYERS = (
    Layer("zones", "line", "wall", size=0.05),
    Layer("flows", "line", "ink", size=0.04),
    Layer("contained", "cell", "agent", size=1.0),
    Layer("hosts", "cell", "wall", size=0.7),
    Layer("jewels", "cell", "goal", size=0.7),
    Layer("rebuilding", "cell", "trail", size=0.7),
    Layer("encrypted", "cell", "hazard", size=0.9),
    Layer("moves", "line", "hazard", size=0.06, fade=True),
    Layer("footholds", "cell", "hazard", size=0.35, rounded=True),
    Layer("owned", "cell", "hazard", size=0.5),
    Layer("exfil", "line", "hazard", size=0.14),
    Layer("progress", "box", "hazard"),
    Layer("c2", "dot", "hazard", size=0.45),
    Layer("c2_blocked", "dot", "wall", size=0.45),
    Layer("cases", "dot", "alert", size=0.16),
    Layer("confirmed", "dot", "alert", size=0.28),
    Layer("analysts", "dot", "agent", size=0.16),
    Layer("actions", "dot", "agent", size=0.3),
)
# Stolen is a percentage of the database. Held counts the hosts the intruder is on.
_LABELS = (Label("cost", 1), Label("stolen"), Label("staged"), Label("held"),
           Label("open_cases"), Label("out_of_service"))


def layout(roles) -> list[tuple[int, int]]:
    """The cell each host is drawn in, in host order."""
    cells, workstation, server = [], 0, 0
    for role in roles:
        if role in WORKSTATIONS:
            cells.append((4 + 2 * (workstation % 3), 6 + 2 * (workstation // 3)))
            workstation += 1
        elif role == WEB:
            cells.append(_WEB)
        elif role == DC:
            cells.append(_DC)
        elif role == DB:
            cells.append(_DB)
        else:
            cells.append(_SERVER_SLOTS[server])
            server += 1
    return cells


class CyberDefenceGame(Game):
    """A security team containing an intruder on a simulated enterprise network.

    Each analyst is an agent, and every one plays the same network. The score is
    what the incident cost, so lower is better.
    """

    name = "cyber"
    Settings = CyberSettings
    direction = "lower"
    workload_label = "footholds and open cases"
    observation = OBSERVATION
    actions = ACTIONS

    def __init__(self, settings: CyberSettings | None = None):
        super().__init__(settings)
        self.agents = tuple(f"analyst{n}" for n in range(1, self.settings.soc.analysts + 1))
        self._index = {agent: index for index, agent in enumerate(self.agents)}
        self.incident: Incident | None = None
        self._cells: list[tuple[int, int]] = []

    # -------------------------------------------------------------- episode

    def reset(self, seed: int) -> None:
        self.incident = Incident(self.settings, seed)
        self._cells = layout(self.incident.estate.roles)

    def observe(self, agent: str):
        return self.incident.view(self._index[agent])

    def step(self, actions: dict) -> None:
        self.incident.step([actions[agent] for agent in self.agents])

    @property
    def over(self) -> bool:
        return self.incident.finished

    @property
    def end(self) -> str:
        return self.incident.end.value

    @property
    def score(self) -> float:
        return self.incident.score

    @property
    def steps(self) -> int:
        return self.incident.t

    def stop(self, stalled: bool) -> None:
        self.incident.stop()

    # -------------------------------------------------------------- results

    def goal(self) -> tuple[float, dict[str, float]]:
        """Minus the incident's cost, by what it was spent on.

        The terms add up to the fitness.
        """
        terms = {term: 0.0 - value for term, value in self.incident.costs.items()}
        return sum(terms.values()), terms

    def signals(self) -> dict[str, float]:
        i = self.incident
        detected = i.detected_at >= 0
        return {
            "lost": float(i.lost),
            "breached": float(i.end == EndCause.BREACHED),
            "ransomed": float(i.end == EndCause.RANSOMED),
            "evicted": float(i.end == EndCause.EVICTED),
            "stolen": i.stolen,
            "encrypted": float(len(i.encrypted)),
            "hosts_hit": float(len(i.ever)),
            "footholds_peak": float(i.footholds_peak),
            "dwell": float(i.dwell_steps),
            "detected": float(detected),
            "time_to_detect": float(i.detected_at if detected
                                    else self.settings.incident.max_steps),
            "investigations": float(i.investigations),
            "findings": float(i.findings),
            "missed": float(i.missed),
            "isolations": float(i.isolations),
            "false_isolations": float(i.false_isolations),
            "reimages": float(i.reimages),
            "cleaned": float(i.cleaned),
            "false_reimages": float(i.false_reimages),
            "resets": float(i.resets),
            "blocks": float(i.blocks),
            "true_alerts": float(i.true_alerts),
            "false_alerts": float(i.false_alerts),
        }

    # -------------------------------------------------------------- watching

    def scene(self) -> Scene:
        return Scene(25, 14, _LAYERS, _LABELS, unit=0.25)

    def draw(self) -> dict:
        i = self.incident
        cells = self._cells
        estate = i.estate
        encrypted = set(i.encrypted)
        hosts, jewels, rebuilding, contained = [], [], [], []
        footholds, owned, cases, confirmed, analysts = [], [], [], [], []
        for h, cell in enumerate(cells):
            x, y = cell
            if h in encrypted:
                pass
            elif i.rebuilding_until[h]:
                rebuilding.append(cell)
            elif h in (estate.dc, estate.db):
                jewels.append(cell)
            else:
                hosts.append(cell)
            if i.isolated[h]:
                contained.append(cell)
            if i.access[h] == 1:
                footholds.append(cell)
            elif i.access[h] == 2:
                owned.append(cell)
            if i.alerts[h]:
                cases.append((x + 0.75, y + 0.25))
            if i.confirmed[h]:
                confirmed.append((x + 0.75, y + 0.25))
            if h in i.slots or i.investigated_until[h] > i.t:
                analysts.append((x + 0.25, y + 0.75))
        moves = [(*_centre(cells[a]), *_centre(cells[b])) for a, b in i.moves[-_MOVES_SHOWN:]]
        exfil = ([(*_centre(cells[i.exfil_from]), *_C2)] if i.exfil_from >= 0 else [])
        progress = []
        if i.stolen:
            progress.append((21.25, 8.25, 2.5 * i.stolen, 0.25))
        if i.staged:
            progress.append((17.25, 8.25, 2.5 * i.staged / self.settings.intruder.stage_steps,
                             0.25))
        held = any(i.access)
        actions = []
        for kind, h in i.acted:
            if kind == "block":
                actions.append(_C2)
            else:
                actions.append(_centre(cells[h]))
        return {
            "zones": [line for x1, y1, x2, y2 in _ZONES
                      for line in ((x1, y1, x2, y1), (x2, y1, x2, y2),
                                   (x2, y2, x1, y2), (x1, y2, x1, y1))],
            "flows": list(_FLOWS),
            "contained": contained,
            "hosts": hosts,
            "jewels": jewels,
            "rebuilding": rebuilding,
            "encrypted": [cells[h] for h in i.encrypted],
            "moves": moves,
            "footholds": footholds,
            "owned": owned,
            "exfil": exfil,
            "progress": progress,
            "c2": [_C2] if i.channel_up and held else [],
            "c2_blocked": [_C2] if not i.channel_up and i.switch_at >= 0 else [],
            "cases": cases,
            "confirmed": confirmed,
            "analysts": analysts,
            "actions": actions,
            "cost": i.score,
            "stolen": round(100 * i.stolen),
            "staged": i.staged,
            "held": i.footholds,
            "open_cases": i.open_cases,
            "out_of_service": sum(1 for h in range(i.size) if not i.in_service(h)),
        }

    def milestones(self, summaries: list[dict]) -> list[Milestone]:
        return milestones(summaries)

    # -------------------------------------------------------------- profiling

    def workload(self) -> int:
        return self.incident.footholds + self.incident.open_cases

    # -------------------------------------------------------------- settings

    @classmethod
    def setting_choices(cls) -> dict[str, tuple]:
        return {
            "intruder.profile": ("mixed", "operator", "careful", "smash"),
            "intruder.objective": ("mixed", "steal", "ransom"),
        }

    @classmethod
    def setting_problems(cls, settings: CyberSettings) -> list[str]:
        estate, intruder, detection = settings.estate, settings.intruder, settings.detection
        soc, costs, incident = settings.soc, settings.costs, settings.incident
        problems = Problems()
        # The scene has room for 12 workstations and 5 servers.
        problems.between("estate.workstations_min", estate.workstations_min, 2, 12)
        problems.between("estate.workstations_max", estate.workstations_max,
                         min(estate.workstations_min, 12), 12)
        problems.between("estate.servers_min", estate.servers_min, 2, 2 + len(OPTIONAL_SERVERS))
        most = 2 + len(OPTIONAL_SERVERS)
        problems.between("estate.servers_max", estate.servers_max,
                         min(estate.servers_min, most), most)
        for name in ("flat_user_lan", "shared_local_admin", "web_flaw", "server_flaw", "dc_flaw",
                     "admin_creds_on_server", "domain_admin_on_admin_ws", "unmanaged"):
            problems.between(f"estate.{name}", getattr(estate, name), 0, 1)

        for name in ("careful_share", "smash_share", "ransom_share", "phish_entry",
                     "escalate_server"):
            problems.between(f"intruder.{name}", getattr(intruder, name), 0, 1)
        if intruder.careful_share + intruder.smash_share > 1:
            problems.append(
                f"intruder.careful_share and intruder.smash_share add up to "
                f"{intruder.careful_share + intruder.smash_share:g}; "
                f"they can add up to 1 at most"
            )
        # Every estate's way to the database and the domain controller relies on
        # escalations and exploits working at least sometimes.
        for name in ("escalate_workstation", "exploit"):
            value = getattr(intruder, name)
            if not 0 < value <= 1:
                problems.append(f"intruder.{name} must be above 0 and at most 1, not {value}")
        problems.at_least("intruder.start_min", intruder.start_min, 1)
        problems.between("intruder.start_max", intruder.start_max, intruder.start_min,
                         incident.max_steps - 1)
        problems.at_least("intruder.re_entries_min", intruder.re_entries_min, 0)
        problems.at_least("intruder.re_entries_max", intruder.re_entries_max,
                          intruder.re_entries_min)
        for name in ("re_entry_delay", "vpn_delay", "channel_switch"):
            low = getattr(intruder, f"{name}_min")
            problems.at_least(f"intruder.{name}_min", low, 1)
            problems.at_least(f"intruder.{name}_max", getattr(intruder, f"{name}_max"), low)
        for name in ("escalate_retry", "exfil_steps", "stage_steps"):
            problems.at_least(f"intruder.{name}", getattr(intruder, name), 1)

        for field in dataclasses.fields(detection):
            if field.name not in ("false_alerts", "noise_spread", "delay_max", "expire_after"):
                problems.between(f"detection.{field.name}", getattr(detection, field.name), 0, 1)
        # Keeps the noisiest host's rate of false alerts below 1 a step.
        problems.between("detection.false_alerts", detection.false_alerts, 0, 5)
        problems.between("detection.noise_spread", detection.noise_spread, 0, 0.9)
        problems.between("detection.delay_max", detection.delay_max, 0, 10)
        problems.at_least("detection.expire_after", detection.expire_after, 1)

        problems.between("soc.analysts", soc.analysts, 1, 4)
        problems.between("soc.investigate_steps", soc.investigate_steps, 1, 24)
        for name in ("find", "find_careful", "find_unmanaged"):
            problems.between(f"soc.{name}", getattr(soc, name), 0, 1)
        for name in ("reimage_workstation", "reimage_server", "reimage_core"):
            problems.at_least(f"soc.{name}", getattr(soc, name), 1)
        problems.at_least("soc.reset_cooldown", soc.reset_cooldown, 0)

        for field in dataclasses.fields(costs):
            if field.name != "data":
                problems.at_least(f"costs.{field.name}", getattr(costs, field.name), 0)
        problems.above("costs.data", costs.data, 0)
        problems.at_least("incident.max_steps", incident.max_steps, 20)
        return problems

    @classmethod
    def unused_settings(cls, settings: CyberSettings) -> dict[str, str]:
        intruder, estate = settings.intruder, settings.estate
        unused = {}
        if intruder.profile != "mixed":
            for name in ("careful_share", "smash_share"):
                unused[f"intruder.{name}"] = f"intruder.profile is {intruder.profile}"
        if intruder.objective != "mixed":
            unused["intruder.ransom_share"] = f"intruder.objective is {intruder.objective}"
        # Which end the intruder is after can be fixed by the share as well.
        mixed = intruder.objective == "mixed"
        if intruder.objective == "steal" or (mixed and intruder.ransom_share == 0):
            why = ("intruder.objective is steal" if not mixed
                   else "intruder.ransom_share is 0")
            for name in ("intruder.stage_steps", "detection.stage", "costs.ransom"):
                unused[name] = f"{why}, so nothing is encrypted"
        if intruder.objective == "ransom" or (mixed and intruder.ransom_share == 1):
            why = ("intruder.objective is ransom" if not mixed
                   else "intruder.ransom_share is 1")
            for name in ("intruder.exfil_steps", "detection.exfil", "costs.data"):
                unused[name] = f"{why}, so nothing is stolen"
        careful = (intruder.profile == "careful"
                   or (intruder.profile == "mixed" and intruder.careful_share > 0))
        if not careful:
            for name in ("detection.careful", "soc.find_careful"):
                unused[name] = "no intruder works carefully"
        if intruder.re_entries_max == 0:
            for name in ("re_entry_delay_min", "re_entry_delay_max"):
                unused[f"intruder.{name}"] = "intruder.re_entries_max is 0"
        if estate.web_flaw == 0:
            for name in ("intruder.phish_entry", "detection.entry_web"):
                unused[name] = "estate.web_flaw is 0, so every entry is by phishing"
        elif intruder.phish_entry == 1:
            unused["detection.entry_web"] = (
                "intruder.phish_entry is 1, so every entry is by phishing")
        if estate.unmanaged == 0:
            for name in ("detection.no_agent", "soc.find_unmanaged"):
                unused[name] = "estate.unmanaged is 0, so every host runs an endpoint agent"
        if settings.detection.false_alerts == 0:
            unused["detection.noise_spread"] = (
                "detection.false_alerts is 0, so no host raises false alerts")
        return unused


def _centre(cell: tuple[float, float]) -> tuple[float, float]:
    return cell[0] + 0.5, cell[1] + 0.5


# ---------------------------------------------------------------- milestones

# Mean costs over a champion's games, from the score each summary records.
_COST_TIERS = (
    (120, "Better than nothing", "A champion's incidents cost under 120 on average"),
    (100, "Holding the line", "A champion's incidents cost under 100 on average"),
    (80, "On top of it", "A champion's incidents cost under 80 on average"),
    (60, "Sharp", "A champion's incidents cost under 60 on average"),
    (40, "Hard target", "A champion's incidents cost under 40 on average"),
)
# A champion plays only a few incidents, so the milestones that ask for every
# one of them to go well ask for it this many generations running.
_RUNNING = 3


def _first_generation_where(summaries: list[dict], predicate, running: int = 1) -> int | None:
    streak = 0
    for entry in summaries:
        try:
            met = predicate(entry)
        except (KeyError, TypeError):
            met = False
        streak = streak + 1 if met else 0
        if streak >= running:
            return int(entry["generation"])
    return None


def _champion(entry: dict) -> dict:
    return entry["best_individual"]["signals"]


def milestones(summaries: list[dict]) -> list[Milestone]:
    """Milestones reached at or before the last summary supplied."""
    found: list[Milestone] = []
    for threshold, label, detail in _COST_TIERS:
        gen = _first_generation_where(
            summaries, lambda e, t=threshold: e["best_score"] < t
        )
        found.append(Milestone(f"cost_{threshold}", label, detail, gen))
    found.append(Milestone(
        "first_find", "First find", "A champion's team found the intruder",
        _first_generation_where(summaries, lambda e: _champion(e)["findings"] > 0),
    ))
    found.append(Milestone(
        "first_cleanup", "First clean-up", "A champion's team rebuilt a host the intruder held",
        _first_generation_where(summaries, lambda e: _champion(e)["cleaned"] > 0),
    ))
    found.append(Milestone(
        "thrown_out", "Thrown out", "A champion's team threw the intruder out for good",
        _first_generation_where(summaries, lambda e: _champion(e)["evicted"] > 0),
    ))
    found.append(Milestone(
        "nothing_lost", "Nothing lost",
        f"Nothing stolen or encrypted in any of the champion's incidents, "
        f"{_RUNNING} generations running",
        _first_generation_where(summaries, lambda e: _champion(e)["lost"] == 0, _RUNNING),
    ))
    found.append(Milestone(
        "clean_sweep", "Clean sweep",
        f"The intruder thrown out of every one of the champion's incidents, "
        f"{_RUNNING} generations running",
        _first_generation_where(summaries, lambda e: _champion(e)["evicted"] == 1, _RUNNING),
    ))
    found.append(Milestone(
        "light_touch", "Light touch",
        f"Nothing lost, and on average less than one host isolated or rebuilt for nothing, "
        f"{_RUNNING} generations running",
        _first_generation_where(
            summaries,
            lambda e: _champion(e)["lost"] == 0
            and _champion(e)["false_isolations"] + _champion(e)["false_reimages"] < 1,
            _RUNNING,
        ),
    ))
    return found
