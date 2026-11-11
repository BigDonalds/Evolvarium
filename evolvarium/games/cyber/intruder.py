"""The intruder's choices, what it sees, and the scripted intruders.

An intruder is a policy over a declared set of actions, the same way an agent
is, so a network can take its place later. It picks a tactic, what to aim at,
and how carefully to go about it. The rules of the incident decide which host
the tactic is used on and what happens. Getting in and getting back in are
rules too, not choices.

A policy is asked only when the intruder holds a host it can use and is not in
the middle of something. It sees its own position and never the defenders'
alerts.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

from ...contract import Actions, Choice, Observation

TACTICS = ("wait", "scan", "escalate", "dump", "move", "act")
AIMS = ("objective", "quiet", "spread")
TEMPOS = ("loud", "careful")
INTRUDER_ACTIONS = Actions(Choice("tactic", TACTICS), Choice("aim", AIMS),
                           Choice("tempo", TEMPOS))

INTRUDER_INPUTS = (
    "time", "footholds", "admin_footholds", "frozen", "lost_recently", "busy",
    "can_scan", "can_escalate", "can_dump", "can_move", "objective_in_reach", "can_act",
    "cred_local", "cred_admin", "cred_service", "cred_domain",
    "objective_ransom", "progress", "channel_up", "entries_left", "known",
)
INTRUDER_OBSERVATION = Observation(INTRUDER_INPUTS)

Policy = Callable[[Sequence[float], random.Random], dict]

_AT = {name: index for index, name in enumerate(INTRUDER_INPUTS)}


class Profile:
    """A scripted intruder's habits.

    `pace` is its chance of acting when it could, `lie_low` its chance of
    waiting after losing ground, and `aim` and `tempo` how it goes about it.
    """

    __slots__ = ("name", "pace", "lie_low", "aim", "tempo")

    def __init__(self, name: str, pace: float, lie_low: float, aim: str, tempo: str):
        self.name = name
        self.pace = pace
        self.lie_low = lie_low
        self.aim = aim
        self.tempo = tempo


PROFILES = {
    "operator": Profile("operator", 0.5, 0.8, "objective", "loud"),
    "careful": Profile("careful", 0.35, 0.95, "quiet", "careful"),
    "smash": Profile("smash", 1.0, 0.0, "objective", "loud"),
}


def scripted(profile: Profile) -> Policy:
    """The policy of a scripted intruder: the first of its rules that applies."""
    aim, tempo, loud = profile.aim, profile.tempo, profile.tempo == "loud"

    def choose(tactic: str, aim: str = aim, tempo: str = tempo) -> dict:
        return {"tactic": tactic, "aim": aim, "tempo": tempo}

    def policy(view: Sequence[float], rng: random.Random) -> dict:
        if not view[_AT["footholds"]] or view[_AT["busy"]]:
            return choose("wait")
        # A ransom crew that holds domain admin presses on when it is noticed.
        pressing = view[_AT["objective_ransom"]] and view[_AT["cred_domain"]]
        if view[_AT["lost_recently"]] and not pressing and rng.random() < profile.lie_low:
            return choose("wait")
        if rng.random() >= profile.pace:
            return choose("wait")
        if view[_AT["can_act"]]:
            return choose("act")
        in_reach = view[_AT["objective_in_reach"]]
        if in_reach == 1.0 or (in_reach and loud):
            return choose("move", "objective")
        if view[_AT["can_dump"]]:
            return choose("dump")
        if view[_AT["can_escalate"]]:
            return choose("escalate")
        can_move = view[_AT["can_move"]]
        if can_move == 1.0 or (can_move and loud):
            return choose("move")
        if view[_AT["can_scan"]]:
            return choose("scan")
        # With no quiet way on, even a careful intruder tries an exploit.
        if can_move or in_reach:
            return choose("move", "objective", "loud")
        return choose("wait")

    return policy
