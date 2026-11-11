"""The cyber defence simulation's settings: the estate, the intruder, detection,
the security team, what an incident costs, and how long one lasts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EstateConfig:
    """The network, drawn afresh from each seed.

    Every estate has a web server, an application server, a file server, a
    domain controller and a customer database, and between them some user
    workstations and more servers.
    """

    # The IT administrator's workstation counts as one of these.
    workstations_min: int = 6
    workstations_max: int = 10
    # The application and file servers count as two of these.
    servers_min: int = 3
    servers_max: int = 5
    # Chance the user network lets workstations open sessions to each other.
    flat_user_lan: float = 0.8
    # Chance every workstation shares one local administrator password.
    shared_local_admin: float = 0.6
    # Chances that a service has a flaw an intruder can exploit.
    web_flaw: float = 0.6
    server_flaw: float = 0.4
    dc_flaw: float = 0.1
    # Chance a server holds the IT administrator's credentials in memory.
    admin_creds_on_server: float = 0.3
    # Chance a domain administrator's session sits on the IT administrator's
    # workstation, rather than on one of the file, mail and print servers.
    domain_admin_on_admin_ws: float = 0.5
    # Chance an ordinary workstation, or a file, mail, backup or print server,
    # runs no endpoint agent.
    unmanaged: float = 0.15


@dataclass(frozen=True)
class IntruderConfig:
    """Who gets in, after what, and how they go about it."""

    # One of mixed, operator, careful or smash. Mixed draws one per seed.
    profile: str = "mixed"
    careful_share: float = 0.35
    smash_share: float = 0.20
    # One of mixed, steal or ransom. Mixed draws one per seed.
    objective: str = "mixed"
    ransom_share: float = 0.4
    # Chance of a phishing email even when the web server could be exploited.
    phish_entry: float = 0.7
    # The step of the first entry.
    start_min: int = 1
    start_max: int = 12
    # Fresh phishing campaigns after being thrown out, and how long each takes.
    re_entries_min: int = 1
    re_entries_max: int = 3
    re_entry_delay_min: int = 12
    re_entry_delay_max: int = 36
    # A logon over the VPN with a stolen password, and how long before it.
    vpn_delay_min: int = 3
    vpn_delay_max: int = 8
    # A new command channel after the old one is blocked.
    channel_switch_min: int = 3
    channel_switch_max: int = 8
    # Chances that a privilege escalation works, and steps before a retry.
    escalate_workstation: float = 0.75
    escalate_server: float = 0.6
    escalate_retry: int = 8
    # Chance an exploit of a flawed service lands.
    exploit: float = 0.8
    # The database leaves in this many units, one per completed theft.
    exfil_steps: int = 12
    # Ransomware is staged on the domain controller for this many steps, then
    # set off.
    stage_steps: int = 4


@dataclass(frozen=True)
class DetectionConfig:
    """Chances that the intruder's actions raise an alert, and the noise around them."""

    entry_phish: float = 0.3
    entry_web: float = 0.5
    entry_vpn: float = 0.15
    scan: float = 0.35
    escalate: float = 0.5
    dump: float = 0.6
    logon: float = 0.2
    exploit: float = 0.6
    exfil: float = 0.15
    stage: float = 0.4
    # Per held host per step, for the command channel's traffic.
    beacon: float = 0.01
    # Multiplies the chances above for an intruder working carefully, and for
    # actions an endpoint agent would see on a host that has none.
    careful: float = 0.5
    no_agent: float = 0.25
    # Scales every host's rate of false alerts. Each host's own rate is its
    # role's times a draw between 1 - noise_spread and 1 + noise_spread.
    false_alerts: float = 1.0
    noise_spread: float = 0.5
    # Steps between an action and its alert, at most.
    delay_max: int = 2
    # An open alert nobody acts on closes after this many steps, unless its host
    # is isolated.
    expire_after: int = 48


@dataclass(frozen=True)
class SocConfig:
    """The security team. Each analyst is one agent, and all of them play the same network."""

    analysts: int = 2
    investigate_steps: int = 4
    # Chance an investigation finds an intruder that is there, and what it is
    # multiplied by when the intruder worked carefully on the host, or the host
    # runs no endpoint agent. An investigation never reports an intruder that
    # is not there.
    find: float = 0.85
    find_careful: float = 0.6
    find_unmanaged: float = 0.6
    # Steps a host is out of service while it is rebuilt.
    reimage_workstation: int = 8
    reimage_server: int = 16
    reimage_core: int = 24
    # Steps before the credentials can be reset again.
    reset_cooldown: int = 24


@dataclass(frozen=True)
class CostConfig:
    """What an incident costs, in units where the whole customer database is 100."""

    data: float = 100.0
    # Charged when the ransomware goes off, on top of rebuilding every host it
    # encrypts.
    ransom: float = 40.0
    # Per step a host is out of service, isolated or being rebuilt. The IT
    # administrator's workstation is costed as a server.
    down_workstation: float = 0.05
    down_server: float = 0.25
    down_frontline: float = 0.4
    down_database: float = 1.5
    down_dc: float = 1.5
    # Per step the intruder holds a host.
    dwell_workstation: float = 0.02
    dwell_server: float = 0.05
    dwell_core: float = 0.3
    # Per action.
    investigate: float = 0.05
    isolate: float = 0.02
    reimage: float = 0.3
    # A user's lost morning and the help desk's time, on top of the rebuild.
    reimage_workstation: float = 1.0
    reset: float = 6.0
    block: float = 0.1
    # Charged at the end of an incident the intruder is still in.
    unresolved: float = 10.0


@dataclass(frozen=True)
class IncidentConfig:
    """How long an incident lasts. A step is half an hour."""

    max_steps: int = 200
    # End the incident once the intruder is out for good and every host is back
    # in service.
    end_on_eviction: bool = True


@dataclass(frozen=True)
class CyberSettings:
    estate: EstateConfig = field(default_factory=EstateConfig)
    intruder: IntruderConfig = field(default_factory=IntruderConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    soc: SocConfig = field(default_factory=SocConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    incident: IncidentConfig = field(default_factory=IncidentConfig)
