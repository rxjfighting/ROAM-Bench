"""Joint articulation utilities for ROAM-Bench."""
from dataclasses import dataclass, field
from typing import List, Tuple
from roam_bench.joint_usd_map import UsdJointParams, map_joint
from roam_bench.sdf_joints import JointSpec
from roam_bench.seg_alias import _safe

class ArticulationError(Exception):
    pass

@dataclass(frozen=True)
class LinkPlan:
    link_name: str
    prim_path: str
    is_root: bool

@dataclass(frozen=True)
class JointPlan:
    joint_name: str
    joint_prim_path: str
    params: UsdJointParams
    body0_path: str
    body1_path: str

@dataclass(frozen=True)
class ArticulationPlan:
    model: str
    instance_root: str
    root_link: str
    links: Tuple[LinkPlan, ...] = field(default_factory=tuple)
    joints: Tuple[JointPlan, ...] = field(default_factory=tuple)

def is_articulated(specs) -> bool:
    return bool(specs)

def link_prim_path(instance_root: str, link_name: str) -> str:
    return '%s/%s' % (instance_root.rstrip('/'), _safe(link_name))

def joints_scope(instance_root: str) -> str:
    return '%s/joints' % instance_root.rstrip('/')

def find_root_link(specs) -> str:
    parents = {s.parent_link for s in specs}
    children = {s.child_link for s in specs}
    roots = parents - children
    if len(roots) != 1:
        raise ArticulationError('fixed-base articulation needs exactly one root link (parent-never-child); got %r (parents=%r children=%r)' % (sorted(roots), sorted(parents), sorted(children)))
    return next(iter(roots))

def plan_articulation(instance_root: str, specs) -> ArticulationPlan:
    if not is_articulated(specs):
        raise ArticulationError('no drivable joints: this model is not an articulation (step-5 interact must REJECT it, not author an empty root)')
    root = find_root_link(specs)
    child_of = {}
    for s in specs:
        if s.child_link in child_of:
            raise ArticulationError('link %r is the child of more than one joint (%r and %r): not a tree' % (s.child_link, child_of[s.child_link], s.name))
        child_of[s.child_link] = s.name
    parents = {s.parent_link for s in specs}
    children = set(child_of)
    links = sorted(parents | children)
    link_plans = tuple((LinkPlan(link_name=ln, prim_path=link_prim_path(instance_root, ln), is_root=ln == root) for ln in links))
    jscope = joints_scope(instance_root)
    joint_plans = []
    for s in specs:
        p = map_joint(s)
        if p.body0 != s.parent_link or p.body1 != s.child_link:
            raise ArticulationError('map_joint body names disagree with spec for %r: body0=%r/parent=%r body1=%r/child=%r' % (s.name, p.body0, s.parent_link, p.body1, s.child_link))
        joint_plans.append(JointPlan(joint_name=s.name, joint_prim_path='%s/%s' % (jscope, _safe(s.name)), params=p, body0_path=link_prim_path(instance_root, s.parent_link), body1_path=link_prim_path(instance_root, s.child_link)))
    model = specs[0].model or ''
    return ArticulationPlan(model=model, instance_root=instance_root, root_link=root, links=link_plans, joints=tuple(joint_plans))

def plan_dict(plan: ArticulationPlan) -> dict:
    return {'model': plan.model, 'instance_root': plan.instance_root, 'root_link': plan.root_link, 'n_links': len(plan.links), 'n_joints': len(plan.joints), 'links': [{'name': lp.link_name, 'prim_path': lp.prim_path, 'is_root': lp.is_root} for lp in plan.links], 'joints': [{'name': jp.joint_name, 'prim_path': jp.joint_prim_path, 'drive': jp.params.drive, 'axis': jp.params.axis, 'lower': jp.params.lower, 'upper': jp.params.upper, 'open_pos': jp.params.open_pos, 'closed_pos': jp.params.closed_pos, 'interior_zero': jp.params.interior_zero, 'body0_path': jp.body0_path, 'body1_path': jp.body1_path} for jp in plan.joints]}

def plan_scene(scene_root: str) -> List[ArticulationPlan]:
    from roam_bench.sdf_joints import parse_scene_joints
    out = []
    for i, (name, _sdf, specs) in enumerate(parse_scene_joints(scene_root)):
        root = '/World/objs/_dryrun/%s_%d' % (_safe(name), i)
        out.append(plan_articulation(root, specs))
    return out
