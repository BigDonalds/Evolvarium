"""The network under attack: its hosts, what can reach what, and where credentials sit.

An estate is drawn from its own random stream, so the same seed gives the same
network whatever the defence or the intruder does. Hosts always come in one
order, which the inputs, the scene and the tests all use: the IT administrator's
workstation, the other workstations, the web server, the servers, the domain
controller and the database.

Credentials are bits of one integer. A user's password is a bit of its own per
workstation, from bit 5 up.
"""

from __future__ import annotations

import random

from .settings import CyberSettings

WORKSTATION, ADMIN_WS, WEB, APP, FILE, MAIL, BACKUP, PRINT, DC, DB = range(10)
ROLE_NAMES = ("workstation", "admin_ws", "web", "app", "file", "mail", "backup", "print",
              "dc", "db")
SERVERS = (APP, FILE, MAIL, BACKUP, PRINT)
WORKSTATIONS = (WORKSTATION, ADMIN_WS)
OPTIONAL_SERVERS = (MAIL, BACKUP, PRINT)

USERS, DMZ, SERVER_ZONE, IDENTITY, RESTRICTED = range(5)
ZONE = (USERS, USERS, DMZ, SERVER_ZONE, SERVER_ZONE, SERVER_ZONE, SERVER_ZONE, SERVER_ZONE,
        IDENTITY, RESTRICTED)

# What a host is worth to the business, as the asset inventory rates it.
VALUE = (0.2, 0.5, 0.4, 0.6, 0.5, 0.5, 0.7, 0.5, 1.0, 1.0)
# Hosts on the way to the database or the domain controller.
CROWN_PATH = (False, True, False, True, False, False, True, False, True, True)
# False alerts a step, before the estate's own spread. Endpoints raise most of the
# low alerts a security team sees.
NOISE = (0.045, 0.06, 0.06, 0.03, 0.03, 0.03, 0.03, 0.03, 0.04, 0.02)
# How much each host draws an intruder, by what it is after.
STEAL_AIM = (1, 6, 2, 5, 3, 3, 9, 3, 8, 10)
RANSOM_AIM = (1, 9, 2, 5, 3, 3, 10, 3, 11, 3)

LOCAL_ADMIN, IT_ADMIN, DB_SERVICE, WEB_SERVICE, DOMAIN_ADMIN = 1, 2, 4, 8, 16
PRIVILEGED = IT_ADMIN | DB_SERVICE | WEB_SERVICE | DOMAIN_ADMIN
_USER_BIT = 5


def password(host: int) -> int:
    """The bit of the password of the user of workstation `host`."""
    return 1 << (_USER_BIT + host)


class Estate:
    """One network, drawn from `rng`."""

    def __init__(self, settings: CyberSettings, rng: random.Random):
        config = settings.estate
        workstations = rng.randint(config.workstations_min, config.workstations_max)
        servers = rng.randint(config.servers_min, config.servers_max)
        extra = rng.sample(OPTIONAL_SERVERS, servers - 2)
        roles = [ADMIN_WS] + [WORKSTATION] * (workstations - 1) + [WEB, APP, FILE]
        roles += [role for role in OPTIONAL_SERVERS if role in extra] + [DC, DB]
        self.roles = tuple(roles)
        self.size = size = len(roles)
        self.web = roles.index(WEB)
        self.app = roles.index(APP)
        self.file = roles.index(FILE)
        self.dc = roles.index(DC)
        self.db = roles.index(DB)
        self.backup = roles.index(BACKUP) if BACKUP in roles else -1
        self.workstations = tuple(h for h, role in enumerate(roles) if role in WORKSTATIONS)
        # Ordinary workstations, the ones phishing lands on.
        self.users = self.workstations[1:]
        self.servers = tuple(h for h, role in enumerate(roles) if role in SERVERS)
        self.flat = rng.random() < config.flat_user_lan
        self.shared_local_admin = rng.random() < config.shared_local_admin

        flaws = [False] * size
        for h, role in enumerate(roles):
            if role in SERVERS:
                flaws[h] = rng.random() < config.server_flaw
            elif role == WEB:
                flaws[h] = rng.random() < config.web_flaw
            elif role == DC:
                flaws[h] = rng.random() < config.dc_flaw
        self.flaws = flaws

        managed = [True] * size
        for h, role in enumerate(roles):
            if role in (WORKSTATION, FILE, MAIL, BACKUP, PRINT):
                managed[h] = rng.random() >= config.unmanaged
        self.managed = tuple(managed)

        cached = [0] * size
        for h, role in enumerate(roles):
            if role in WORKSTATIONS:
                cached[h] |= password(h) | (LOCAL_ADMIN if self.shared_local_admin else 0)
            if role == ADMIN_WS:
                cached[h] |= IT_ADMIN
            if role in SERVERS and rng.random() < config.admin_creds_on_server:
                cached[h] |= IT_ADMIN
            if role == APP:
                cached[h] |= DB_SERVICE
            if role == WEB:
                cached[h] |= WEB_SERVICE
            if role == DC:
                cached[h] |= DOMAIN_ADMIN
        if rng.random() < config.domain_admin_on_admin_ws:
            cached[0] |= DOMAIN_ADMIN
        else:
            cached[rng.choice([h for h, role in enumerate(roles)
                               if role in (FILE, MAIL, PRINT)])] |= DOMAIN_ADMIN
        self.cached = cached

        self.reach = self._reach()
        self.reached_by = tuple(tuple(c for c in range(size) if h in self.reach[c])
                                for h in range(size))
        # Credentials that give admin, or a user's level, on each host.
        admin_by, user_by = [0] * size, [0] * size
        for h, role in enumerate(roles):
            admin_by[h] = DOMAIN_ADMIN
            if role in WORKSTATIONS:
                admin_by[h] |= IT_ADMIN | (LOCAL_ADMIN if self.shared_local_admin else 0)
                user_by[h] |= password(h)
            if role in SERVERS or role in (WEB, DB):
                admin_by[h] |= IT_ADMIN
            if role == DB:
                user_by[h] |= DB_SERVICE
            if role == APP:
                user_by[h] |= WEB_SERVICE
        self.admin_by = tuple(admin_by)
        self.user_by = tuple(user_by)

        # Every estate has a way from a phished workstation to the database and to
        # the domain controller.
        self.forced_database = self.forced_domain = False
        reaches_db, holds_dc = self.closure()
        if not reaches_db:
            self.flaws[self.app] = True
            self.forced_database = True
        reaches_db, holds_dc = self.closure()
        if not holds_dc:
            self.cached[self.file] |= DOMAIN_ADMIN
            self.flaws[self.file] = True
            self.forced_domain = True
        self.cached = tuple(self.cached)
        self.privileged = tuple(bool(c & PRIVILEGED) for c in self.cached)

        spread = settings.detection.noise_spread
        self.noise = tuple(
            NOISE[role] * settings.detection.false_alerts
            * rng.uniform(1 - spread, 1 + spread) * (1.0 if managed[h] else 0.5)
            for h, role in enumerate(roles)
        )

    def _reach(self) -> tuple[frozenset, ...]:
        """Who may open a session to whom, as the firewalls allow."""
        roles, everything = self.roles, set(range(self.size))
        reach = []
        for h, role in enumerate(roles):
            if role == WORKSTATION:
                to = set(self.servers) - {self.backup} | {self.dc}
                if self.flat:
                    to |= set(self.workstations)
            elif role in (ADMIN_WS, DC):
                to = set(everything)
            elif role == WEB:
                to = {self.app}
            elif role in SERVERS:
                to = set(self.servers) | {self.dc}
                if role in (APP, BACKUP):
                    to.add(self.db)
            else:
                to = set()
            to.discard(h)
            reach.append(frozenset(to))
        return tuple(reach)

    def closure(self, start: int | None = None, level: int = 1) -> tuple[bool, bool]:
        """Whether a foothold on `start`, the first ordinary workstation unless
        given, could reach the database and hold the domain controller, if every
        attempt worked."""
        held = {self.users[0] if start is None else start: level}
        creds = 0
        grown = True
        while grown:
            grown = False
            for h in list(held):
                if held[h] < 2 and self.roles[h] != DB:
                    held[h] = 2
                    grown = True
                if held[h] == 2 and creds | self.cached[h] != creds:
                    creds |= self.cached[h]
                    grown = True
                for to in self.reach[h]:
                    if creds & self.admin_by[to] or (self.flaws[to] and self.roles[to] != DB):
                        level = 2
                    else:
                        level = 1 if creds & self.user_by[to] else 0
                    if level > held.get(to, 0):
                        held[to] = level
                        grown = True
        return held.get(self.db, 0) >= 1, held.get(self.dc, 0) == 2
