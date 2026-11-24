"""Joint interact utilities for ROAM-Bench."""
from dataclasses import dataclass
from typing import Iterable, Tuple
from roam_bench.joint_articulation import ArticulationError, ArticulationPlan, is_articulated, plan_articulation
OPEN = 'open'
CLOSE = 'close'
_ACTIONS = (OPEN, CLOSE)
_ABS_FLOOR = 0.0001
_REL = 0.01

def readback_tol(target: float) -> float:
    return max(_ABS_FLOOR, _REL * abs(target))

def pose_within(q: float, target: float) -> bool:
    return abs(q - target) <= readback_tol(target)

@dataclass(frozen=True)
class DriveTarget:
    joint_prim_path: str
    drive: str
    target: float

@dataclass(frozen=True)
class InteractPlan:
    model: str
    instance_root: str
    action: str
    targets: Tuple[DriveTarget, ...]

def _targets_for(plan: ArticulationPlan, action: str) -> Tuple[DriveTarget, ...]:
    return tuple((DriveTarget(joint_prim_path=jp.joint_prim_path, drive=jp.params.drive, target=jp.params.open_pos if action == OPEN else jp.params.closed_pos) for jp in plan.joints))

def plan_interact(plan: ArticulationPlan, action: str) -> InteractPlan:
    if action not in _ACTIONS:
        raise ValueError('interact action must be one of %r, got %r' % (_ACTIONS, action))
    if not plan.joints:
        raise ArticulationError('interact(%r) on a plan with no joints for %r: reject, do not fail open' % (action, plan.model))
    return InteractPlan(model=plan.model, instance_root=plan.instance_root, action=action, targets=_targets_for(plan, action))

def plan_interact_from_specs(instance_root: str, specs: Iterable, action: str) -> InteractPlan:
    specs = list(specs)
    if not is_articulated(specs):
        raise ArticulationError('interact(%r) on a jointless object (no drivable joints): REJECT, do not fail open -- returning success here silently claims the world changed' % (action,))
    plan = plan_articulation(instance_root, specs)
    return plan_interact(plan, action)

def interact_dict(plan: InteractPlan) -> dict:
    return {'model': plan.model, 'instance_root': plan.instance_root, 'action': plan.action, 'n_targets': len(plan.targets), 'targets': [{'joint_prim_path': t.joint_prim_path, 'drive': t.drive, 'target': t.target} for t in plan.targets]}
