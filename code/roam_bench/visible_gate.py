"""Visible gate utilities for ROAM-Bench."""
from dataclasses import dataclass
NOT_VISIBLE_REASON = 'You cannot see that from where you are standing.'
NOT_REACHABLE_REASON = 'You cannot find a way to stand next to it.'

@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reason: str
    visible: bool
    reachable: bool

def decide(*, visible: bool, reachable: bool, not_visible_reason: str=NOT_VISIBLE_REASON, not_reachable_reason: str=NOT_REACHABLE_REASON) -> GateDecision:
    v, r = (bool(visible), bool(reachable))
    if not v:
        return GateDecision(False, not_visible_reason, v, r)
    if not r:
        return GateDecision(False, not_reachable_reason, v, r)
    return GateDecision(True, '', v, r)

def gate_alias(*, alias, target_room, agent_room, visible_aliases, nav_reachable) -> GateDecision:
    visible = alias in visible_aliases
    reachable = bool(nav_reachable(agent_room, target_room))
    return decide(visible=visible, reachable=reachable)
