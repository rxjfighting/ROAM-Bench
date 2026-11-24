"""Openable joint exclusions utilities for ROAM-Bench."""
import dataclasses
WINE_CABINET_MODEL_SUBSTR = 'wine_refrigerator_display_cabinet'
EXCLUDED_OPENABLE_JOINTS = {WINE_CABINET_MODEL_SUBSTR: {name: {'reason': 'Inconsistent joint geometry; excluded from the interaction skill.'} for name in ('PrismaticJoint_refrigerator_7_up1_joint', 'PrismaticJoint_refrigerator_7_down2_joint')}}

def excluded_joint_names_for(model_or_instance_name):
    out = {}
    for substr, joints in EXCLUDED_OPENABLE_JOINTS.items():
        if substr in (model_or_instance_name or ''):
            out.update(joints)
    return out

def apply_openable_joint_exclusions(plan, instance_name=None):
    model = getattr(plan, 'model', '') or ''
    excluded = excluded_joint_names_for(model) or excluded_joint_names_for(instance_name or '')
    if not excluded:
        return plan
    kept = tuple((jp for jp in plan.joints if jp.joint_name not in excluded))
    if len(kept) == len(plan.joints):
        return plan
    return dataclasses.replace(plan, joints=kept)
