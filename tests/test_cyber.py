"""Cyber defence: the estate, the intruder, the security team and the engine running it.

Rule tests build an incident from a seed with an intruder that never acts on its
own, then place footholds, credentials and alerts by hand, so each one checks a
single rule. Where a test needs an analyst on a particular host it hands the host
over directly.
"""

from __future__ import annotations

import dataclasses
import random

import numpy as np
import pytest
from test_settings import configure, refused

from evolvarium import games, held_out, settings
from evolvarium.config import CONFIG
from evolvarium.contract import override
from evolvarium.evolution.evaluation import EvalContext, make_runner
from evolvarium.evolution.population import Population
from evolvarium.games.cyber import CyberDefenceGame, CyberSettings, scripted
from evolvarium.games.cyber.engine import BUSY_VIEW, INPUTS, RESPONSES, EndCause, Incident
from evolvarium.games.cyber.estate import (
    ADMIN_WS,
    APP,
    BACKUP,
    DB,
    DB_SERVICE,
    DC,
    DOMAIN_ADMIN,
    FILE,
    IT_ADMIN,
    WEB,
    WORKSTATION,
    password,
)
from evolvarium.games.cyber.game import milestones
from evolvarium.games.cyber.intruder import (
    AIMS,
    INTRUDER_ACTIONS,
    INTRUDER_INPUTS,
    INTRUDER_OBSERVATION,
    PROFILES,
    TACTICS,
    TEMPOS,
)
from evolvarium.games.cyber.intruder import (
    scripted as scripted_intruder,
)
from evolvarium.nn.network import Architecture, PolicyNetwork
from evolvarium.persistence import export_model, model_file
from evolvarium.player import Player
from evolvarium.replays import unpack

CYBER = CyberSettings()
ENDS = {cause.value for cause in EndCause}
DEV = range(700_000, 700_020)


def cyber(**changes) -> CyberSettings:
    """The default settings with each change, named section__field."""
    result = CYBER
    for name, value in changes.items():
        result = override(result, name.replace("__", "."), value)
    return result


def waiting(view, rng) -> dict:
    return {"tactic": "wait", "aim": "objective", "tempo": "loud"}


def acting(*actions):
    """An intruder that takes `actions` in turn, as (tactic, aim, tempo), then waits."""
    queue = list(actions)

    def policy(view, rng):
        if not queue:
            return waiting(view, rng)
        tactic, aim, tempo = queue.pop(0)
        return {"tactic": tactic, "aim": aim, "tempo": tempo}

    return policy


def ready(settings: CyberSettings = CYBER, seed: int = 3, policy=waiting,
          evict: bool = False) -> Incident:
    """An incident whose intruder is already in, with nothing held yet and no way
    back once thrown out, so only what a test places is there. Unless `evict`,
    it does not end when nothing is held."""
    if not evict:
        settings = override(settings, "incident.end_on_eviction", False)
    incident = Incident(settings, seed, intruder=policy)
    incident.reentry_at = -1
    incident.first_entry = 0
    incident.entries = 0
    return incident


def hold(incident: Incident, *hosts: int, level: int = 2) -> Incident:
    for h in hosts:
        incident.access[h] = level
        incident.ever.add(h)
        incident.known.add(h)
    incident.channel_up = True
    incident.channel = max(incident.channel, 1)
    return incident


def hand(incident: Incident, h: int, analyst: int = 0) -> None:
    incident.slots[analyst] = h
    incident.positions[analyst] = 0


def respond(incident: Incident, *responses: str) -> None:
    padded = list(responses) + ["watch"] * (len(incident.slots) - len(responses))
    incident.step(padded)


def first(incident: Incident, role: int) -> int:
    return incident.estate.roles.index(role)


def cyber_config(**evolution):
    return CONFIG.with_overrides(
        game="cyber",
        settings=cyber(incident__max_steps=60),
        evolution=dataclasses.replace(
            CONFIG.evolution, population_size=10, evaluation_episodes=2, elite_count=2,
            immigrant_count=1, workers=1, **evolution,
        ),
    )


# ----------------------------------------------------------------- the estate


def test_an_incident_is_a_function_of_its_seed_and_responses():
    def run(seed):
        i = scripted.play(CYBER, scripted.analyst, seed)
        return i.score, i.t, i.end, dict(i.costs), i.true_alerts, i.false_alerts, i.ever

    assert run(700_005) == run(700_005)
    assert run(700_005) != run(700_006)


def test_estates_keep_their_order_and_vary_by_seed():
    sizes = set()
    for seed in range(60):
        estate = Incident(CYBER, seed).estate
        roles = estate.roles
        workstations = sum(1 for r in roles if r in (WORKSTATION, ADMIN_WS))
        assert 6 <= workstations <= 10 and 3 <= len(estate.servers) <= 5
        assert roles[0] == ADMIN_WS and roles[1:workstations] == (WORKSTATION,) * (workstations - 1)
        assert roles[workstations:workstations + 3] == (WEB, APP, FILE)
        assert roles[-2:] == (DC, DB)
        sizes.add(estate.size)
    assert len(sizes) >= 5


def test_every_estate_has_a_way_to_the_database_and_the_domain_controller():
    forced = 0
    for seed in range(200):
        estate = Incident(CYBER, seed).estate
        assert estate.closure() == (True, True)
        forced += estate.forced_database or estate.forced_domain
    assert 0 < forced < 200


@pytest.mark.parametrize("profile", sorted(PROFILES))
@pytest.mark.parametrize("objective", ["steal", "ransom"])
def test_undefended_every_intruder_gets_what_it_came_for(profile, objective):
    settings = cyber(intruder__profile=profile, intruder__objective=objective,
                     incident__max_steps=2000)
    for seed in DEV:
        incident = scripted.play(settings, scripted.idle, seed)
        assert incident.end == (EndCause.RANSOMED if objective == "ransom"
                                else EndCause.BREACHED), (seed, incident.end)


def test_the_firewall_decides_who_reaches_whom():
    for seed in range(20):
        estate = Incident(CYBER, seed).estate
        roles = estate.roles
        for h, role in enumerate(roles):
            reach = set(estate.reach[h])
            assert h not in reach
            if role == WORKSTATION:
                assert estate.db not in reach and estate.backup not in reach
                assert estate.dc in reach
            elif role in (ADMIN_WS, DC):
                assert reach == set(range(estate.size)) - {h}
            elif role == WEB:
                assert reach == {estate.app}
            elif role == DB:
                assert reach == set()
            else:
                assert (estate.db in reach) == (role in (APP, BACKUP))


def test_the_estate_and_its_false_alerts_do_not_depend_on_the_intruder():
    one = Incident(CYBER, 11)
    other = Incident(cyber(intruder__profile="smash", intruder__objective="ransom",
                           intruder__start_max=3), 11)
    for field in ("roles", "flaws", "cached", "noise", "managed"):
        assert getattr(one.estate, field) == getattr(other.estate, field)
    assert one._false == other._false


# ----------------------------------------------------------------- the intruder


def test_phishing_lands_on_an_ordinary_workstation_with_its_password():
    settings = cyber(intruder__start_min=1, intruder__start_max=1, intruder__phish_entry=1.0)
    incident = Incident(settings, 5, intruder=waiting)
    respond(incident)
    held = [h for h in range(incident.size) if incident.access[h]]
    assert len(held) == 1
    h = held[0]
    assert incident.estate.roles[h] == WORKSTATION and incident.access[h] == 1
    assert incident.creds & password(h)
    # A machine joined to the domain knows its domain controller.
    assert incident.estate.dc in incident.known
    assert incident.first_entry == 1 and incident.channel_up


def test_a_web_exploit_lands_as_admin_on_the_web_server():
    settings = cyber(intruder__start_min=1, intruder__start_max=1, intruder__phish_entry=0.0,
                     estate__web_flaw=1.0)
    incident = Incident(settings, 5, intruder=waiting)
    respond(incident)
    assert incident.access[incident.estate.web] == 2
    assert sum(1 for level in incident.access if level) == 1


def test_scanning_learns_what_the_host_reaches():
    incident = ready(policy=acting(("scan", "objective", "loud")))
    h = first(incident, WORKSTATION)
    hold(incident, h)
    respond(incident)
    assert incident.scanned[h]
    assert set(incident.estate.reach[h]) <= incident.known


def test_dumping_takes_the_credentials_the_host_holds():
    incident = ready(policy=acting(("dump", "objective", "loud")))
    hold(incident, 0)
    respond(incident)
    assert incident.dumped[0]
    assert incident.creds & incident.estate.cached[0] == incident.estate.cached[0]
    assert incident.creds & IT_ADMIN


def test_a_move_with_a_credential_lands_at_the_level_it_gives():
    incident = ready(policy=acting(("move", "objective", "loud")))
    app = incident.estate.app
    hold(incident, 0)
    incident.known = {0, app}
    incident.creds = IT_ADMIN
    respond(incident)
    assert incident.access[app] == 2
    assert incident.moves == [(0, app)]


def test_working_carefully_takes_twice_as_long_and_never_exploits():
    incident = ready(policy=acting(("move", "objective", "careful")))
    app = incident.estate.app
    hold(incident, 0)
    incident.known = {0, app}
    incident.creds = IT_ADMIN
    respond(incident)
    assert incident.access[app] == 0 and incident.busy is not None
    respond(incident)
    assert incident.access[app] == 2 and incident.careful_on[app]

    incident = ready(policy=acting(("move", "objective", "careful")))
    web = incident.estate.web
    hold(incident, 0)
    incident.known = {0, web}
    incident.flaws[web] = True
    respond(incident)
    assert incident.busy is None and incident.access[web] == 0


def test_an_action_is_abandoned_when_its_host_is_isolated():
    incident = ready(policy=acting(("escalate", "objective", "loud")))
    h = first(incident, WORKSTATION)
    hold(incident, h, level=1)
    respond(incident)
    assert incident.busy[0] == "escalate"
    incident.isolated[h] = True
    respond(incident)
    assert incident.busy is None and incident.access[h] == 1
    # Once for the host it lost, and once for the action.
    assert incident.losses == [incident.t, incident.t]

    # A move whose target is isolated is given up, though its source still works.
    incident = ready(policy=acting(("move", "objective", "careful")))
    app = incident.estate.app
    hold(incident, 0)
    incident.known = {0, app}
    incident.creds = IT_ADMIN
    respond(incident)
    incident.isolated[app] = True
    respond(incident)
    assert incident.busy is None and incident.access[app] == 0
    assert incident.losses == [incident.t]


def test_the_database_leaves_through_the_host_the_intruder_came_in_by():
    settings = cyber(intruder__objective="steal", intruder__exfil_steps=2)
    incident = ready(settings, policy=acting(("act", "objective", "loud"),
                                             ("act", "objective", "loud")))
    estate = incident.estate
    hold(incident, estate.app)
    hold(incident, estate.db, level=1)
    incident.moves = [(estate.app, estate.db)]
    assert incident.relay() == estate.app
    respond(incident)
    assert incident.exfil_from == estate.app and incident.stolen == pytest.approx(0.5)
    assert incident.costs["data"] == pytest.approx(50.0)
    respond(incident)
    assert incident.end == EndCause.BREACHED

    # With its relay isolated, the database cannot be worked from at all.
    incident = ready(settings, policy=acting(("act", "objective", "loud")))
    hold(incident, estate.app)
    hold(incident, estate.db, level=1)
    incident.moves = [(estate.app, estate.db)]
    incident.isolated[estate.app] = True
    assert not incident.usable(estate.db)
    respond(incident)
    assert incident.stolen == 0.0


def test_ransomware_encrypts_every_host_in_service_when_it_goes_off():
    settings = cyber(intruder__objective="ransom", intruder__stage_steps=2)
    incident = ready(settings, policy=acting(("act", "objective", "loud"),
                                             ("act", "objective", "loud")))
    dc = incident.estate.dc
    hold(incident, dc)
    spared = first(incident, WORKSTATION)
    incident.isolated[spared] = True
    respond(incident)
    assert incident.staged == 1
    respond(incident)
    assert incident.end == EndCause.RANSOMED
    assert spared not in incident.encrypted
    assert set(incident.encrypted) == set(range(incident.size)) - {spared}
    expected = CYBER.costs.ransom + sum(incident.rebuild_steps[h] * incident.down_rate[h]
                                        for h in incident.encrypted)
    assert incident.costs["ransom"] == pytest.approx(expected)


def test_isolating_the_domain_controller_pauses_staging_and_rebuilding_it_starts_over():
    settings = cyber(intruder__objective="ransom", intruder__stage_steps=3)
    incident = ready(settings, policy=acting(*[("act", "objective", "loud")] * 3))
    dc = incident.estate.dc
    hold(incident, dc)
    hold(incident, 0)
    respond(incident)
    assert incident.staged == 1
    incident.isolated[dc] = True
    respond(incident)
    assert incident.staged == 1 and not incident.finished
    incident.isolated[dc] = False
    hand(incident, dc)
    respond(incident, "reimage")
    assert incident.staged == 0 and incident.access[dc] == 0


def test_with_a_password_it_returns_by_vpn_and_without_one_it_phishes():
    settings = cyber(intruder__vpn_delay_min=2, intruder__vpn_delay_max=2,
                     intruder__re_entry_delay_min=3, intruder__re_entry_delay_max=3)
    incident = ready(settings)
    h = first(incident, WORKSTATION)
    incident.creds = password(h)
    incident.channel_up = True
    respond(incident)
    assert incident.reentry_kind == "vpn"
    respond(incident)
    respond(incident)
    assert incident.access[h] == 1

    # With that workstation isolated it cannot log on, and spends a campaign instead.
    incident = ready(settings)
    incident.entries = 1
    incident.creds = password(h)
    incident.isolated[h] = True
    respond(incident)
    assert incident.reentry_kind == "phish" and incident.entries == 0


def test_phishing_waits_while_every_workstation_is_isolated():
    settings = cyber(intruder__start_min=1, intruder__start_max=1, intruder__phish_entry=1.0)
    incident = Incident(settings, 5, intruder=waiting)
    for h in incident.estate.users:
        incident.isolated[h] = True
    for _ in range(3):
        respond(incident)
        assert not any(incident.access)
    for h in incident.estate.users:
        incident.isolated[h] = False
    respond(incident)
    assert any(incident.access)


def test_scripted_intruders_choose_only_declared_actions():
    for name, profile in PROFILES.items():
        chosen = []
        policy = scripted_intruder(profile)

        def recording(view, rng, policy=policy, chosen=chosen):
            assert len(view) == len(INTRUDER_INPUTS)
            action = policy(view, rng)
            chosen.append(action)
            return action

        incident = Incident(cyber(intruder__profile=name), 700_003, intruder=recording)
        while not incident.finished:
            respond(incident)
        assert chosen
        for action in chosen:
            assert action["tactic"] in TACTICS and action["aim"] in AIMS
            assert action["tempo"] in TEMPOS


def test_a_network_can_play_the_intruder():
    architecture = Architecture(INTRUDER_OBSERVATION.size, (8,), ("tanh",),
                                INTRUDER_ACTIONS.size)
    # Most random networks only pick tactics they can't use, which plays out as
    # waiting. This one scans, moves and acts, so its choices show in the incident.
    genome = np.random.default_rng(19).normal(size=architecture.genome_size).astype(np.float32)
    network = PolicyNetwork(architecture).load(genome)
    tactics = set()

    def policy(view, rng):
        assert len(view) == len(INTRUDER_INPUTS)
        action = INTRUDER_ACTIONS.decode(network.forward(INTRUDER_OBSERVATION.encode(view)))
        tactics.add(action["tactic"])
        return action

    def play(intruder):
        incident = Incident(CYBER, 700_004, intruder=intruder)
        while not incident.finished:
            respond(incident)
        return incident

    played = play(policy)
    assert len(tactics) > 1
    assert played.costs != play(waiting).costs


# ----------------------------------------------------------------- the team


def test_isolation_freezes_a_foothold_and_release_lets_it_go_on():
    incident = ready()
    h = first(incident, WORKSTATION)
    hold(incident, h)
    hand(incident, h)
    respond(incident, "isolate")
    assert incident.isolated[h] and not incident.usable(h)
    dwell = incident.costs["dwell"]
    respond(incident)
    assert incident.costs["dwell"] > dwell
    assert incident.costs["downtime"] == pytest.approx(2 * incident.down_rate[h])
    hand(incident, h)
    respond(incident, "release")
    assert not incident.isolated[h] and incident.usable(h)


def test_rebuilding_removes_the_intruder_patches_the_flaw_and_takes_the_host_down():
    incident = ready()
    web = incident.estate.web
    hold(incident, web)
    incident.flaws[web] = True
    hand(incident, web)
    respond(incident, "reimage")
    assert incident.access[web] == 0 and not incident.flaws[web]
    assert incident.costs["effort"] == pytest.approx(CYBER.costs.reimage)
    for _ in range(CYBER.soc.reimage_server - 1):
        respond(incident)
    assert incident.rebuilding_until[web]
    respond(incident)
    assert not incident.rebuilding_until[web]
    assert incident.costs["downtime"] == pytest.approx(
        CYBER.soc.reimage_server * incident.down_rate[web])


def test_rebuilding_a_workstation_costs_its_user_and_resets_their_password():
    incident = ready()
    h = first(incident, WORKSTATION)
    hold(incident, h, 0)
    incident.creds = password(h) | IT_ADMIN
    hand(incident, h)
    respond(incident, "reimage")
    assert not incident.creds & password(h) and incident.creds & IT_ADMIN
    assert incident.costs["effort"] == pytest.approx(
        CYBER.costs.reimage + CYBER.costs.reimage_workstation)


def test_a_reset_voids_stolen_credentials_but_leaves_implants():
    incident = ready()
    hold(incident, 0)
    incident.dumped[0] = True
    incident.creds = IT_ADMIN | DOMAIN_ADMIN
    hand(incident, 0)
    respond(incident, "reset")
    assert incident.creds == 0 and incident.access[0] == 2
    # Fresh credentials can be taken again from a host held as admin.
    assert not incident.dumped[0]
    assert incident.costs["effort"] == pytest.approx(CYBER.costs.reset)
    hand(incident, 0)
    respond(incident, "reset")
    assert incident.resets == 1


def test_a_block_cuts_the_live_channel_and_a_stale_address_blocks_nothing():
    incident = ready()
    hold(incident, 0)
    incident.c2_known = incident.channel
    hand(incident, 0)
    respond(incident, "block")
    assert not incident.channel_up and incident.switch_at > incident.t
    assert incident.c2_known == -1

    incident = ready()
    hold(incident, 0)
    incident.c2_known = incident.channel - 1
    hand(incident, 0)
    respond(incident, "block")
    assert incident.channel_up and incident.c2_known == -1


def test_an_investigation_keeps_its_analyst_busy_then_finds_or_clears():
    settings = cyber(soc__find=1.0, soc__find_unmanaged=1.0, detection__false_alerts=0.0)
    incident = ready(settings)
    h, quiet = first(incident, WORKSTATION), first(incident, WORKSTATION) + 1
    hold(incident, h)
    hand(incident, h, 0)
    hand(incident, quiet, 1)
    respond(incident, "investigate", "investigate")
    assert incident.slots == [None, None]
    assert incident.view(0) == BUSY_VIEW
    for _ in range(CYBER.soc.investigate_steps - 1):
        respond(incident, "reimage", "reimage")
        assert incident.reimages == 0
    respond(incident)
    assert incident.confirmed[h] and incident.findings == 1
    assert not incident.confirmed[quiet] and incident.clean[quiet]


def test_an_investigation_never_finds_an_intruder_that_is_not_there():
    settings = cyber(soc__find=1.0, detection__false_alerts=0.0)
    for seed in range(5):
        incident = ready(settings, seed)
        hand(incident, 2)
        respond(incident, "investigate")
        for _ in range(CYBER.soc.investigate_steps):
            respond(incident)
        assert not any(incident.confirmed) and incident.clean[2]


def test_a_finding_raises_a_lead_on_every_host_a_move_links_it_to():
    incident = ready()
    app, file = incident.estate.app, incident.estate.file
    hold(incident, 0, app, file)
    incident.moves = [(0, app), (app, file)]
    incident.confirm(app)
    respond(incident)
    for h in (0, file):
        assert any(lead and severity == 3 for severity, _, lead in incident.alerts[h])
    assert incident.c2_known == incident.channel and incident.admin_found


def test_alerts_expire_unless_their_host_is_isolated():
    incident = ready(cyber(detection__expire_after=3, detection__false_alerts=0.0))
    hold(incident, 0)
    incident.alerts[2] = [(2, 0, False)]
    incident.alerts[3] = [(2, 0, False)]
    incident.isolated[3] = True
    for _ in range(3):
        respond(incident)
    assert incident.alerts[2] == [] and incident.alerts[3]


def test_the_queue_puts_confirmed_and_risky_hosts_first():
    incident = ready(cyber(detection__false_alerts=0.0))
    hold(incident, 0)
    incident.alerts[1] = [(1, 0, False)]
    incident.alerts[2] = [(3, 0, False)]
    incident.alerts[4] = [(1, 0, False)]
    incident.confirmed[4] = True
    incident._rank()
    assert incident.slots == [4, 2]
    assert incident.view(1)[INPUTS.index("slot")] == pytest.approx(0.5)


def test_a_response_that_cannot_apply_costs_nothing():
    incident = ready()
    hold(incident, 0)
    hand(incident, 2)
    respond(incident, "release")
    hand(incident, 2)
    respond(incident, "block")
    assert incident.costs["effort"] == 0.0 and incident.blocks == 0


# ----------------------------------------------------------------- seeing


def test_every_input_stays_between_0_and_1_in_random_play():
    rng = random.Random(3)
    for seed in DEV[:5]:
        incident = Incident(CYBER, seed)
        while not incident.finished:
            for k in range(len(incident.slots)):
                view = incident.view(k)
                assert len(view) == len(INPUTS)
                assert all(0.0 <= value <= 1.0 for value in view), (seed, view)
            usable = incident.usable_hosts()
            if usable:
                view = incident.intruder_view(usable)
                assert len(view) == len(INTRUDER_INPUTS)
                assert all(0.0 <= value <= 1.0 for value in view)
            incident.step([rng.choice(RESPONSES) for _ in incident.slots])


def test_a_foothold_nobody_has_noticed_changes_no_analyst_s_view():
    clean, hidden = ready(seed=8), ready(seed=8)
    hold(hidden, first(hidden, WORKSTATION))
    for incident in (clean, hidden):
        incident._rank()
    assert [clean.view(k) for k in range(2)] == [hidden.view(k) for k in range(2)]


# ----------------------------------------------------------------- ends and costs


def test_a_timeout_with_the_intruder_still_in_charges_for_it():
    incident = ready(cyber(incident__max_steps=20))
    hold(incident, 0)
    while not incident.finished:
        respond(incident)
    assert incident.end == EndCause.TIMEOUT
    assert incident.costs["unresolved"] == CYBER.costs.unresolved

    stopped = ready()
    hold(stopped, 0)
    respond(stopped)
    stopped.stop()
    assert stopped.end == EndCause.TIMEOUT
    assert stopped.costs["unresolved"] == CYBER.costs.unresolved


def test_the_intruder_is_thrown_out_only_once_every_host_is_back_in_service():
    incident = ready(evict=True)
    h = first(incident, WORKSTATION)
    hold(incident, h)
    hand(incident, h)
    respond(incident, "reimage")
    assert not incident.finished
    while not incident.finished:
        respond(incident)
    assert incident.end == EndCause.EVICTED
    assert incident.t == 1 + CYBER.soc.reimage_workstation


def test_the_fitness_terms_add_up_to_minus_the_cost():
    game = CyberDefenceGame()
    game.reset(700_009)
    while not game.over:
        game.step({agent: scripted.analyst(game.observe(agent), None) for agent in game.agents})
    fitness, terms = game.goal()
    assert set(terms) == {"data", "ransom", "downtime", "dwell", "effort", "unresolved"}
    assert fitness == pytest.approx(sum(terms.values()))
    assert fitness == pytest.approx(-sum(game.incident.costs.values()))
    assert game.score == round(sum(game.incident.costs.values()), 1)
    assert all(str(value) != "-0.0" for value in terms.values())


def test_downtime_and_dwell_are_charged_by_role():
    incident = ready()
    estate = incident.estate
    hold(incident, estate.dc, 1)
    incident.isolated[estate.db] = True
    respond(incident)
    assert incident.costs["dwell"] == pytest.approx(CYBER.costs.dwell_core
                                                    + CYBER.costs.dwell_workstation)
    assert incident.costs["downtime"] == pytest.approx(CYBER.costs.down_database)
    # The IT administrator's workstation is costed as a server.
    assert incident.down_rate[0] == CYBER.costs.down_server


# ----------------------------------------------------------------- watching


def test_every_host_is_drawn_once_as_it_stands():
    game = CyberDefenceGame()
    game.reset(700_002)
    rng = random.Random(2)
    while True:
        frame = game.draw()
        drawn = frame["hosts"] + frame["jewels"] + frame["rebuilding"] + frame["encrypted"]
        assert sorted(drawn) == sorted(game._cells)
        assert frame["cost"] == game.score
        if game.over:
            break
        game.step({agent: rng.choice(RESPONSES) for agent in game.agents})


def test_drawing_leaves_the_incident_as_it_was():
    game = CyberDefenceGame()
    game.reset(700_001)
    for _ in range(30):
        game.step({agent: scripted.sweep(game.observe(agent), None) for agent in game.agents})
    before = (list(game.incident.access), [list(a) for a in game.incident.alerts],
              game.incident._detect.getstate(), game.incident._rolls.getstate(),
              dict(game.incident.costs), list(game.incident.slots))
    game.draw()
    after = (list(game.incident.access), [list(a) for a in game.incident.alerts],
             game.incident._detect.getstate(), game.incident._rolls.getstate(),
             dict(game.incident.costs), list(game.incident.slots))
    assert before == after


def test_a_cyber_replay_draws_the_incident_it_recorded():
    game = CyberDefenceGame()
    architecture = Architecture(game.observation.size, (8,), ("tanh",), game.actions.size)
    ctx = EvalContext(game="cyber", settings=CYBER, architecture=architecture.to_dict(),
                      exploration_rate=0.0, replay_steps=0)
    genome = np.random.default_rng(4).normal(size=architecture.genome_size).astype(np.float32)
    runner = make_runner(ctx)
    outcome = runner.play(genome, 700_001, True)
    unwatched = runner.play(genome, 700_001, False)
    assert (outcome.score, outcome.steps, outcome.end) == (
        unwatched.score, unwatched.steps, unwatched.end)
    frames = unpack(game.scene(), outcome.replay)
    assert len(frames) == outcome.steps + 1
    assert frames[-1]["cost"] == pytest.approx(outcome.score)
    zones = frames[0]["zones"]
    assert all(frame["zones"] == zones for frame in frames)


# ----------------------------------------------------------------- settings


def test_the_defaults_pass_every_check():
    settings.check(CONFIG.with_overrides(game="cyber"))
    assert CyberDefenceGame.setting_problems(CYBER) == []
    assert CyberDefenceGame.unused_settings(CYBER) == {}


@pytest.mark.parametrize(("assignments", "problem"), [
    ((("estate.workstations_min", "1"),),
     "estate.workstations_min must be between 2 and 12, not 1"),
    ((("estate.workstations_max", "13"),),
     "estate.workstations_max must be between 6 and 12, not 13"),
    ((("estate.servers_max", "6"),), "estate.servers_max must be between 3 and 5, not 6"),
    ((("intruder.careful_share", "0.7"), ("intruder.smash_share", "0.4")),
     "intruder.careful_share and intruder.smash_share add up to 1.1; "
     "they can add up to 1 at most"),
    ((("intruder.exploit", "0"),), "intruder.exploit must be above 0 and at most 1, not 0.0"),
    ((("intruder.start_max", "200"),), "intruder.start_max must be between 1 and 199, not 200"),
    ((("intruder.vpn_delay_max", "2"),), "intruder.vpn_delay_max must be at least 3, not 2"),
    ((("detection.false_alerts", "6"),),
     "detection.false_alerts must be between 0 and 5, not 6.0"),
    ((("soc.analysts", "5"),), "soc.analysts must be between 1 and 4, not 5"),
    ((("costs.data", "0"),), "costs.data must be above 0, not 0.0"),
    ((("incident.max_steps", "10"),), "incident.max_steps must be at least 20, not 10"),
])
def test_cyber_values_that_cannot_run_are_refused(assignments, problem):
    config = CONFIG.with_overrides(game="cyber")
    assert problem in refused(configure(*assignments, config=config))


@pytest.mark.parametrize(("assignments", "reason"), [
    ((("intruder.profile", "smash"), ("intruder.careful_share", "0.5")),
     "intruder.profile is smash"),
    ((("intruder.objective", "steal"), ("intruder.stage_steps", "6")),
     "intruder.objective is steal, so nothing is encrypted"),
    ((("intruder.objective", "ransom"), ("costs.data", "50")),
     "intruder.objective is ransom, so nothing is stolen"),
    ((("estate.unmanaged", "0"), ("soc.find_unmanaged", "0.5")),
     "estate.unmanaged is 0, so every host runs an endpoint agent"),
    ((("estate.web_flaw", "0"), ("detection.entry_web", "0.4")),
     "estate.web_flaw is 0, so every entry is by phishing"),
    ((("intruder.re_entries_min", "0"), ("intruder.re_entries_max", "0"),
      ("intruder.re_entry_delay_max", "40")), "intruder.re_entries_max is 0"),
])
def test_a_cyber_setting_with_no_effect_is_refused(assignments, reason):
    config = configure(*assignments, config=CONFIG.with_overrides(game="cyber"))
    name = assignments[-1][0]
    problems = refused(config, named=[name for name, _ in assignments])
    assert problems == [f"{name} has no effect, since {reason}"]
    settings.check(config)


# ----------------------------------------------------------------- milestones


def test_milestones_come_from_the_cost_and_the_champion_s_signals():
    def summary(generation, cost, **signals):
        base = {"findings": 0.0, "cleaned": 0.0, "evicted": 0.0, "lost": 1.0,
                "false_isolations": 0.0, "false_reimages": 0.0}
        return {"generation": generation, "best_score": cost,
                "best_individual": {"signals": {**base, **signals}}}

    summaries = [
        summary(0, 150.0),
        summary(1, 110.0, findings=1.0),
        summary(2, 85.0, findings=2.0, cleaned=1.0, evicted=0.5, lost=0.0,
                false_reimages=2.0),
        summary(3, 70.0, findings=2.0, cleaned=1.0, evicted=1.0, lost=0.0),
        summary(4, 58.0, evicted=1.0, lost=0.0),
        summary(5, 55.0, evicted=1.0, lost=0.0),
    ]
    reached = {m.key: m.generation for m in milestones(summaries)}
    assert reached == {
        "cost_120": 1, "cost_100": 2, "cost_80": 3, "cost_60": 4, "cost_40": None,
        "first_find": 1, "first_cleanup": 2, "thrown_out": 2,
        "nothing_lost": 4, "clean_sweep": 5, "light_touch": 5,
    }


# ----------------------------------------------------------------- the engine


def test_the_registry_knows_cyber_defence_by_name():
    assert "cyber" in games.names()
    assert games.get("cyber") is CyberDefenceGame
    assert games.reference(CyberDefenceGame()) == "cyber"
    assert CyberDefenceGame().agents == ("analyst1", "analyst2")


def test_a_population_plays_whole_teams():
    population = Population(cyber_config(), seed=41)
    for _ in range(2):
        report = population.run_generation()
        assert set(report.ends) <= ENDS
        assert sum(report.ends.values()) == len(report.evaluated)
    summary = report.to_dict()
    game = CyberDefenceGame()
    game.reset(1)
    assert set(summary["best_individual"]["signals"]) == set(game.signals())
    assert len(milestones([summary])) == 11


def test_cyber_scores_the_same_in_worker_processes():
    from evolvarium.evolution.pool import EvaluationPool

    config = cyber_config(exploration_rate=0.1)
    serial = Population(config, seed=42)
    parallel = Population(config, seed=42)
    with EvaluationPool(parallel.context, workers=2) as pool:
        assert pool.active
        for _ in range(2):
            a = serial.run_generation()
            b = parallel.run_generation(pool=pool)
            assert [m.fitness for m in a.evaluated] == [m.fitness for m in b.evaluated]
            assert a.ends == b.ends


def test_a_trained_team_exports_and_plays_as_trained(tmp_path):
    path = tmp_path / "cyber.json"
    population = Population(cyber_config(), seed=43)
    population.run_generation()
    genome = population.members[0].genome
    report = export_model.build_report(
        generation=0, genome=genome, summaries=[], game=population.game,
        architecture=population.architecture, run_id="run-cyber", destination=path,
    )
    export_model.write_model(report=report, genome=genome,
                             architecture=population.architecture, destination=path)
    assert model_file.read(path)["game"]["name"] == "cyber"

    player = Player.load(path)
    network = PolicyNetwork(population.architecture).load(genome)
    game = CyberDefenceGame(population.config.settings)
    game.reset(5)
    while not game.over:
        actions = {}
        for agent in game.agents:
            inputs = game.observation.encode(game.observe(agent))
            actions[agent] = game.actions.decode(network.forward(inputs))
            assert player.action(inputs.tolist()) == actions[agent]
        game.step(actions)

    champion = held_out.load_champion(path)
    assert champion.context.game == "cyber" and not champion.higher
    held_out.score([champion], [40, 41], workers=1)
    runner = make_runner(dataclasses.replace(population.context, exploration_rate=0.0))
    assert champion.scores.tolist() == [runner.score(genome, s, 1, False).score for s in (40, 41)]


# ----------------------------------------------------------------- the bar


def test_written_rules_land_in_order_on_dev_seeds():
    def mean(rule, bound=False):
        return np.mean([scripted.play(CYBER, rule, seed, bound).score for seed in DEV])

    idle = mean(scripted.idle)
    best = min(mean(scripted.POLICIES[name]) for name in ("analyst", "triage", "hunter",
                                                          "sweep", "ladder", "tuned"))
    assert mean(scripted.wander) > idle > best
    assert mean(scripted.lockdown) > idle
    assert mean(scripted.nuke) > idle
    assert mean(scripted.seer, bound=True) < best


# ----------------------------------------------------------------- more rules


def test_a_reset_during_a_careful_move_leaves_it_nothing_to_log_on_with():
    incident = ready(policy=acting(("move", "objective", "careful")))
    app = incident.estate.app
    hold(incident, 0)
    incident.known = {0, app}
    incident.creds = IT_ADMIN
    respond(incident)
    assert incident.busy is not None
    hand(incident, 0)
    respond(incident, "reset")
    assert incident.access[app] == 0 and incident.moves == []


def test_an_intruder_cut_off_from_the_database_can_reach_it_again():
    settings = cyber(intruder__objective="steal")
    incident = ready(settings, policy=acting(("move", "objective", "loud"),
                                             ("act", "objective", "loud")))
    estate = incident.estate
    hold(incident, estate.app, 0)
    hold(incident, estate.db, level=1)
    incident.moves = [(estate.app, estate.db)]
    incident.creds = DB_SERVICE
    hand(incident, estate.app)
    respond(incident, "reimage")
    assert incident.moves == [(0, estate.db)] and incident.relay() == 0
    respond(incident)
    assert incident.stolen > 0 and incident.exfil_from == 0


def test_the_command_channel_raises_beacons_at_its_rate():
    settings = cyber(detection__beacon=1.0, detection__delay_max=0,
                     detection__false_alerts=0.0)
    incident = ready(settings)
    hold(incident, 0, 1, 2)
    for _ in range(10):
        respond(incident)
    assert incident.true_alerts == 30


def test_a_ransomware_crew_comes_in_by_the_web_only_where_that_leads_to_the_dc():
    settings = cyber(intruder__objective="ransom", intruder__phish_entry=0.0)
    turned_away = 0
    for seed in range(700_000, 700_200):
        incident = Incident(settings, seed)
        estate = incident.estate
        leads_there = estate.closure(estate.web, 2)[1]
        assert incident.web_entry == (estate.flaws[estate.web] and leads_there)
        turned_away += estate.flaws[estate.web] and not leads_there
    assert turned_away > 0


def test_careful_work_and_hosts_without_an_agent_are_harder_to_find():
    careful = cyber(soc__find=1.0, soc__find_careful=0.0, soc__find_unmanaged=1.0,
                    detection__false_alerts=0.0)
    incident = ready(careful)
    h = first(incident, WORKSTATION)
    hold(incident, h)
    incident.careful_on[h] = True
    hand(incident, h)
    respond(incident, "investigate")
    for _ in range(CYBER.soc.investigate_steps):
        respond(incident)
    assert not incident.confirmed[h] and incident.missed == 1

    seed = next(s for s in DEV if not all(Incident(CYBER, s).estate.managed))
    unmanaged = cyber(soc__find=1.0, soc__find_unmanaged=0.0, detection__false_alerts=0.0)
    incident = ready(unmanaged, seed)
    h = incident.estate.managed.index(False)
    hold(incident, h)
    hand(incident, h)
    respond(incident, "investigate")
    for _ in range(CYBER.soc.investigate_steps):
        respond(incident)
    assert not incident.confirmed[h] and incident.missed == 1


def test_ransomware_that_holds_the_backups_doubles_what_recovery_costs():
    seed = next(s for s in DEV if Incident(CYBER, s).estate.backup >= 0)
    settings = cyber(intruder__objective="ransom", intruder__stage_steps=1)
    incident = ready(settings, seed, policy=acting(("act", "objective", "loud")))
    hold(incident, incident.estate.dc, incident.estate.backup)
    respond(incident)
    assert incident.end == EndCause.RANSOMED
    rebuild = sum(incident.rebuild_steps[h] * incident.down_rate[h] for h in incident.encrypted)
    assert incident.costs["ransom"] == pytest.approx(CYBER.costs.ransom + 2 * rebuild)


def test_a_blocked_channel_comes_back_after_its_switch():
    incident = ready(cyber(intruder__channel_switch_min=3, intruder__channel_switch_max=3))
    hold(incident, 0)
    old = incident.channel
    incident.c2_known = old
    hand(incident, 0)
    respond(incident, "block")
    for _ in range(2):
        respond(incident)
        assert not incident.channel_up
    respond(incident)
    assert incident.channel_up and incident.channel == old + 1


def test_the_theft_alert_lands_on_the_relay_not_the_database():
    settings = cyber(intruder__objective="steal", detection__exfil=1.0, detection__beacon=0.0,
                     detection__delay_max=0, detection__false_alerts=0.0)
    incident = ready(settings, policy=acting(("act", "objective", "loud")))
    estate = incident.estate
    hold(incident, estate.app)
    hold(incident, estate.db, level=1)
    incident.moves = [(estate.app, estate.db)]
    respond(incident)
    assert incident.alerts[estate.app] and not incident.alerts[estate.db]


def test_dismissing_closes_a_host_s_alerts_at_no_cost():
    incident = ready()
    incident.alerts[2] = [(3, 0, False), (1, 0, False)]
    hand(incident, 2)
    respond(incident, "dismiss")
    assert incident.alerts[2] == [] and incident.costs["effort"] == 0.0


def test_an_analyst_sees_the_host_it_holds_as_placed():
    incident = ready()
    estate = incident.estate
    h = first(incident, WORKSTATION)
    incident.t = 30
    incident.alerts[h] = [(1, 20, False), (3, 29, True)]
    incident.isolated[h] = True
    incident.isolated_at[h] = 18
    incident.reviewed[h] = 6
    incident.confirmed[0] = True
    incident.alerts[2] = [(2, 25, False)]
    incident._rank()
    assert incident.slots[0] == h
    rebuild = CYBER.costs.reimage + CYBER.costs.reimage_workstation + (
        CYBER.soc.reimage_workstation * CYBER.costs.down_workstation)
    expected = {
        "value": 0.2, "crown_path": 0.0, "privileged": 0.0,
        "managed": 1.0 if estate.managed[h] else 0.0, "workstation": 1.0,
        "rebuild_cost": rebuild / 12, "noisy": min(1.0, estate.noise[h] / 0.12),
        "risk": 7 / 12, "severity": 1.0, "age": 10 / 24, "fresh": 1.0, "lead": 1.0,
        "isolated": 1.0, "isolated_for": 0.5, "clean": 0.0, "confirmed": 0.0,
        "confirmed_admin": 0.0, "unreviewed": 1.0, "exposed": 1.0, "zone_risk": 3 / 12,
        "time": 30 / CYBER.incident.max_steps, "free": 1.0, "open_cases": 3 / 8,
        "alert_rate": 0.0, "contained": 1 / incident.size, "confirmed_hosts": 0.25,
        "c2_known": 0.0, "since_block": 1.0, "reset_ready": 1.0, "admin_found": 0.0,
        "slot": 0.0,
    }
    assert dict(zip(INPUTS, incident.view(0), strict=True)) == pytest.approx(expected)
