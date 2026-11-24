"""Inspect a scene or construct its Isaac Sim stage from supplied runtime assets."""
import argparse
import json
from pathlib import Path
from .dataset import load_scene, spawn_for, scene_for

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('dataset'))
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--task-id', help='Inspect or construct the scene variant for this task.')
    parser.add_argument('--scene-cache', type=Path, default=Path('.roam-cache/scenes'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--inspect', action='store_true', help='Read scene geometry without starting Isaac Sim.')
    parser.add_argument('--robot-asset', type=Path)
    args = parser.parse_args(argv)
    header, specs = load_scene(args.dataset, args.scene, args.assets)
    if args.task_id:
        selected = [spec for spec in specs if spec.task.task_id == args.task_id]
        if not selected:
            parser.error('The task is absent from the selected scene.')
        specs = selected
        header['scene'] = scene_for(header, specs[0], args.scene_cache)
    if args.inspect:
        from .scene_facts import load_facts
        from .scene_loader import load_scene as load_geometry
        facts = load_facts(header['scene'])
        geometry = load_geometry(header['scene'])
        print(json.dumps({'scene': args.scene, 'tasks': len(specs), 'rooms': len(facts.room_types), 'instances': len(facts.instances), 'meshes': len(geometry.instances)}))
        return 0
    if args.output is None:
        parser.error('Stage construction requires --output.')
    from .runtime import configure
    configure(args.robot_asset)
    from .backend import ROAMBackend
    from .protocol.types import Episode
    sim = ROAMBackend(headless=True)
    try:
        sim.set_embodiment('humanoid')
        sim.set_interact(True)
        sim.reset(Episode(episode_id='environment', scene=header['scene'], agent_pose=spawn_for(header, specs[0])))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sim._stage.Export(str(args.output.resolve()))
        print(json.dumps({'scene': args.scene, 'stage': str(args.output)}))
    finally:
        sim.close()
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
