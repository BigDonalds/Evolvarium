"""Hand-written defences, the bar a trained one is measured against.

Each rule reads exactly what a trained analyst reads and gives one of the
responses, so a written rule and a trained one work the same incidents from the
same view. Every analyst of the team plays the same rule.

* idle watches and never acts, which is what an incident costs undefended.
* random responds at random.
* lockdown isolates every host that raises an alert and never lets one go.
* reimage_high rebuilds every host with a high-severity alert.
* nuke rebuilds every ordinary workstation it is handed, alert or not.
* ladder works up from dismissing, through investigating and isolating, to
  rebuilding what an investigation confirms.
* analyst investigates what looks real, contains cheap hosts early and
  dismisses old low alerts.
* triage investigates every host with an open alert.
* hunter investigates every host it is handed until it comes up clean.
* sweep rebuilds ordinary workstations on any alert and investigates the rest.
* tuned decides by the kind of host and how strong its alerts are, from a table
  found by searching rules of that form on the dev seeds.

The two bounds cheat: they read the truth of the incident. The seer knows
whether a host with an open alert is held, and the oracle whether any host is.
They show how much the evidence is worth, not what a defence could reach.

    python -m evolvarium.games.cyber.scripted [--dev] [--by-intruder]

plays every rule on the held-out seeds, or the dev seeds, and prints what each
cost.
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from statistics import fmean as mean

from ...config import HeldOutConfig
from .engine import INPUTS, RESPONSES, Incident
from .settings import CyberSettings

Policy = Callable[[Sequence[float], random.Random], str]
Bound = Callable[[Incident, int, Sequence[float], random.Random], str]

_AT = {name: index for index, name in enumerate(INPUTS)}
_MEDIUM = 2 / 3


def _get(view: Sequence[float], name: str) -> float:
    return view[_AT[name]]


def _opening(view: Sequence[float]) -> str | None:
    """What every working rule does first: block a known command channel,
    rebuild a confirmed host, and reset credentials once an admin foothold is
    found and cleared."""
    if _get(view, "c2_known"):
        return "block"
    if _get(view, "confirmed"):
        return "reimage"
    if (_get(view, "admin_found") and _get(view, "reset_ready")
            and not _get(view, "confirmed_hosts")):
        return "reset"
    return None


def _ordinary_workstation(view: Sequence[float]) -> bool:
    return bool(_get(view, "workstation")) and not _get(view, "crown_path")


def idle(view: Sequence[float], rng: random.Random) -> str:
    return "watch"


def wander(view: Sequence[float], rng: random.Random) -> str:
    return rng.choice(RESPONSES)


def lockdown(view: Sequence[float], rng: random.Random) -> str:
    return "isolate" if _get(view, "risk") else "dismiss"


def reimage_high(view: Sequence[float], rng: random.Random) -> str:
    return "reimage" if _get(view, "severity") == 1.0 else "dismiss"


def nuke(view: Sequence[float], rng: random.Random) -> str:
    """Rebuilds every ordinary workstation, alert or not."""
    opening = _opening(view)
    if opening:
        return opening
    if _ordinary_workstation(view):
        return "reimage"
    if _get(view, "risk") and not _get(view, "clean"):
        return "investigate"
    return "dismiss"


def ladder(view: Sequence[float], rng: random.Random) -> str:
    """Dismiss, investigate, isolate, rebuild: each step up taken on stronger evidence."""
    if _get(view, "c2_known"):
        return "block"
    if _get(view, "confirmed"):
        if _get(view, "confirmed_admin") and _get(view, "reset_ready"):
            return "reset"
        return "reimage"
    if _get(view, "isolated"):
        return "release" if _get(view, "clean") else "investigate"
    if _get(view, "severity") == 1.0 or _get(view, "risk") >= 0.5:
        return "isolate"
    if _get(view, "severity") >= _MEDIUM or _get(view, "lead"):
        return "investigate"
    return "dismiss"


def analyst(view: Sequence[float], rng: random.Random) -> str:
    opening = _opening(view)
    if opening:
        return opening
    risk, severity = _get(view, "risk"), _get(view, "severity")
    clean, isolated, fresh = _get(view, "clean"), _get(view, "isolated"), _get(view, "fresh")
    lead = _get(view, "lead")
    if clean and not fresh:
        return "release" if isolated else "dismiss"
    if (not isolated and not clean and _get(view, "value") <= 0.2
            and (severity >= _MEDIUM or lead)):
        return "isolate"
    if not clean and (lead or severity >= _MEDIUM or (_get(view, "crown_path") and risk)):
        return "investigate"
    if (risk and severity < _MEDIUM and _get(view, "age") >= 0.5 and not fresh
            and _get(view, "value") < 0.5 and not isolated):
        return "dismiss"
    if not risk:
        return "investigate" if isolated else "dismiss"
    return "watch"


def triage(view: Sequence[float], rng: random.Random) -> str:
    opening = _opening(view)
    if opening:
        return opening
    if _get(view, "risk") and not _get(view, "clean"):
        return "investigate"
    if _get(view, "isolated"):
        return "release"
    return "dismiss"


def hunter(view: Sequence[float], rng: random.Random) -> str:
    opening = _opening(view)
    if opening:
        return opening
    if _get(view, "clean") and not _get(view, "fresh"):
        return "release" if _get(view, "isolated") else "dismiss"
    return "investigate"


def sweep(view: Sequence[float], rng: random.Random) -> str:
    opening = _opening(view)
    if opening:
        return opening
    if _ordinary_workstation(view) and _get(view, "risk"):
        return "reimage"
    if _get(view, "risk") and not _get(view, "clean"):
        return "investigate"
    if _get(view, "isolated"):
        return "release"
    return "dismiss"


# What `tuned` does with a host, by its kind and the strength of its open alerts.
# Coordinate descent over every cell of tables like this one on dev seeds
# 700000-700199, from the sweep, triage and quiet rules, ended here from the quiet
# one, at a mean cost of 78.4, and on 800000-800199, which chose nothing, at 82.2.
# It was the best of the three on both blocks.
_TUNED = {
    "workstation": {"none": "investigate", "low": "reimage", "medium": "isolate",
                    "high": "reimage"},
    "admin_ws": {"none": "investigate", "low": "dismiss", "medium": "investigate",
                 "high": "reimage"},
    "server": {"none": "dismiss", "low": "investigate", "medium": "investigate",
               "high": "investigate"},
    "core": {"none": "dismiss", "low": "investigate", "medium": "investigate",
             "high": "investigate"},
}


def _kind(view: Sequence[float]) -> str:
    if _get(view, "workstation"):
        return "admin_ws" if _get(view, "crown_path") else "workstation"
    return "core" if _get(view, "value") >= 1.0 else "server"


def _strength(view: Sequence[float]) -> str:
    """How strong a host's open alerts are: a lead counts as high."""
    if _get(view, "lead") or _get(view, "severity") == 1.0:
        return "high"
    if not _get(view, "risk"):
        return "none"
    return "medium" if _get(view, "severity") >= _MEDIUM else "low"


def tuned(view: Sequence[float], rng: random.Random) -> str:
    opening = _opening(view)
    if opening:
        return opening
    if _get(view, "clean") and not _get(view, "fresh"):
        return "release" if _get(view, "isolated") else "dismiss"
    response = _TUNED[_kind(view)][_strength(view)]
    if _get(view, "isolated"):
        if response == "isolate":
            return "investigate"
        if response == "dismiss":
            return "release"
    return response


POLICIES: dict[str, Policy] = {
    "idle": idle,
    "random": wander,
    "lockdown": lockdown,
    "reimage_high": reimage_high,
    "nuke": nuke,
    "ladder": ladder,
    "analyst": analyst,
    "triage": triage,
    "hunter": hunter,
    "sweep": sweep,
    "tuned": tuned,
}


def _judge(incident: Incident, analyst_index: int, rng: random.Random) -> str:
    """Confirms a held host as a finding would, at once, and clears any other."""
    h = incident.slots[analyst_index]
    if incident.access[h]:
        if not incident.confirmed[h]:
            incident.confirm(h)
        return analyst(incident.view(analyst_index), rng)
    if incident.alerts[h] or incident.clean[h] is False:
        return "release" if incident.isolated[h] else "dismiss"
    return "dismiss"


def seer(incident: Incident, analyst_index: int, view: Sequence[float],
         rng: random.Random) -> str:
    """The analyst rule, judging any host with an open alert at once and rightly."""
    if incident.alerts[incident.slots[analyst_index]]:
        return _judge(incident, analyst_index, rng)
    return analyst(view, rng)


def oracle(incident: Incident, analyst_index: int, view: Sequence[float],
           rng: random.Random) -> str:
    """Judges every host it is handed at once and rightly, alert or not."""
    return _judge(incident, analyst_index, rng)


BOUNDS: dict[str, Bound] = {"seer": seer, "oracle": oracle}


def play(settings: CyberSettings, rule: Policy | Bound, seed: int,
         bound: bool = False) -> Incident:
    """One incident with every analyst playing `rule`. Analysts with no host watch."""
    incident = Incident(settings, seed)
    rng = random.Random(seed ^ 0x5EED)
    analysts = range(settings.soc.analysts)
    while not incident.finished:
        responses = []
        for k in analysts:
            if incident.slots[k] is None:
                responses.append("watch")
                continue
            view = incident.view(k)
            responses.append(rule(incident, k, view, rng) if bound else rule(view, rng))
        incident.step(responses)
    return incident


def _table(settings: CyberSettings, seeds: Sequence[int], by_intruder: bool) -> list[str]:
    count = 2 * len(str(len(seeds))) + 1
    lines = [f"{'rule':<15} {'cost':>6} {'data':>6} {'ransom':>6} {'down':>6} {'dwell':>6} "
             f"{'effort':>6} {'unres':>6} {'lost':>{count}} {'out':>{count}} {'inv':>5} "
             f"{'false':>5} {'rebuilt':>7} {'resets':>6} {'steps':>5}"]
    rules = [(name, rule, False) for name, rule in POLICIES.items()]
    rules += [(name, rule, True) for name, rule in BOUNDS.items()]
    for name, rule, bound in rules:
        incidents = [play(settings, rule, seed, bound) for seed in seeds]
        n = len(incidents)
        terms = {term: mean(i.costs[term] for i in incidents) for term in incidents[0].costs}
        ends = Counter(i.end.value for i in incidents)
        label = f"{name} (bound)" if bound else name
        lost, out = f"{sum(i.lost for i in incidents)}/{n}", f"{ends['evicted']}/{n}"
        lines.append(
            f"{label:<15} {mean(i.score for i in incidents):>6.1f} "
            + " ".join(f"{terms[term]:>6.1f}" for term in terms)
            + f" {lost:>{count}} {out:>{count}} "
            f"{mean(i.investigations for i in incidents):>5.1f} "
            f"{mean(i.false_isolations + i.false_reimages for i in incidents):>5.1f} "
            f"{mean(i.reimages for i in incidents):>7.1f} "
            f"{mean(i.resets for i in incidents):>6.1f} {mean(i.t for i in incidents):>5.0f}"
        )
        if by_intruder:
            groups: dict[tuple[str, str], list[float]] = {}
            for i in incidents:
                groups.setdefault((i.profile, i.objective), []).append(i.score)
            lines.append("    " + ", ".join(
                f"{profile} {objective} {mean(costs):.0f} ({len(costs)})"
                for (profile, objective), costs in sorted(groups.items())))
    return lines


def main(argv: list[str]) -> int:
    held_out = HeldOutConfig()
    if "--dev" in argv:
        seeds = range(700_000, 700_200)
        print("dev seeds 700000-700199")
    else:
        seeds = range(held_out.first_seed, held_out.first_seed + held_out.episodes)
        print(f"held-out seeds {held_out.first_seed}-{held_out.first_seed + held_out.episodes - 1}")
    for line in _table(CyberSettings(), seeds, "--by-intruder" in argv):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
