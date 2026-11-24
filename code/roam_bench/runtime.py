"""Configure body geometry, feedback, scene lighting, and evidence witnesses."""
import json
import math
from pathlib import Path
_BOUND = {}
_CONFIGURED = False
ROOT = Path(__file__).resolve().parent

def bind_task(scene, spec):
    _BOUND[id(spec)] = (scene, spec.task.task_id)

def configure(robot_asset=None):
    global _CONFIGURED
    if _CONFIGURED:
        return
    from . import backend, embodiment, scene_facts, enhanced_evidence
    from .protocol import feedback, llm
    from . import body_feedback, ground_sensor, room_witnesses, source_alternatives
    body = body_feedback
    embodiment.HUMANOID_MESH_BBOX_MIN = body.MIN
    embodiment.HUMANOID_MESH_BBOX_MAX = body.MAX
    embodiment.HUMANOID_COLLISION_HEIGHT = body.MAX[2] + embodiment.HUMANOID_MESH_MARGIN
    embodiment.HUMANOID_COLLISION_RADIUS = max(embodiment.AGENT_RADIUS, max(abs(body.MIN[0]), abs(body.MAX[0])) + embodiment.HUMANOID_MESH_MARGIN)
    embodiment.HUMANOID_TURN_CLEARANCE_RADIUS = max(math.hypot(x, y) for x in (body.MIN[0], body.MAX[0]) for y in (body.MIN[1], body.MAX[1])) + embodiment.HUMANOID_MESH_MARGIN
    embodiment.HUMANOID_COLLISION_FORWARD_OFFSET = 0.20
    if robot_asset is not None:
        path = Path(robot_asset).resolve()
        if not path.is_file():
            raise FileNotFoundError('Robot asset is absent.')
        embodiment.HUMANOID_ASSET = str(path)
    original_author = embodiment.author_humanoid
    def author(stage, *args, **kwargs):
        value = original_author(stage, *args, **kwargs)
        root_path = args[0] if args else kwargs.get('root_path', embodiment.ROOT_PATH)
        robot = stage.GetPrimAtPath(root_path + '/HumanoidMount/G1')
        for side in ('left_hand', 'right_hand'):
            if not robot.GetVariantSet(side).SetVariantSelection('Inspire'):
                raise ValueError('The robot asset requires Inspire hand variants.')
        embodiment._disable_asset_physics(robot)
        value.GetPrim().SetCustomDataByKey('roam_bench:endEffectorVisual', 'Unitree G1 Inspire hands')
        return value
    embodiment.author_humanoid = author
    ground_sensor.install_accepted_harness()
    body_feedback.install_backend(backend)
    body_feedback.install_feedback(feedback)
    body_feedback.install_prompt(llm)
    source_alternatives.install()
    peers = json.loads((ROOT / 'room_witnesses.json').read_text())
    def lookup(spec, trace):
        scene, identifier = _BOUND.get(id(spec), ('', ''))
        return peers.get(scene + '/' + identifier, [])
    room_witnesses.install(enhanced_evidence, lookup)
    original_facts = scene_facts.load_facts
    def facts(scene_root):
        values = original_facts(scene_root)
        if Path(scene_root).name == 'scene_111':
            from dataclasses import replace
            alias = 'waiting_room_poster_0'
            original = values.by_alias.get(alias)
            if original is not None and original.kind == 'poster':
                corrected = replace(original, kind='decorative_framed_art')
                values.by_alias[alias] = corrected
                values.instances = [corrected if row.alias == alias else row for row in values.instances]
        return values
    scene_facts.load_facts = facts
    from .lighting import install
    install(backend.ROAMBackend)
    _CONFIGURED = True
