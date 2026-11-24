"""Load evaluator task specifications and resolve portable reference-image paths."""
import hashlib
import json
from pathlib import Path
from .protocol.taskset import spec_from_dict
from .protocol.types import Pose

def load_scene(dataset, scene, assets=None):
    root = Path(dataset).resolve()
    path = root / 'tasksets' / (scene + '.json')
    document = json.loads(path.read_text())
    if document['scene'] != scene:
        raise ValueError('Scene identifier does not match the taskset filename.')
    for row in document['specs']:
        for reference in row.get('goal_images', []):
            image = (path.parent / reference[1]).resolve()
            if not image.is_relative_to(root) or not image.is_file():
                raise ValueError('Reference image is absent or outside the dataset.')
            reference[1] = str(image)
    header = {key: value for key, value in document.items() if key != 'specs'}
    if assets is not None:
        scene_path = Path(assets).resolve() / scene
        if not scene_path.is_dir():
            raise FileNotFoundError('Supply the scene asset directory for ' + scene)
        header['scene'] = str(scene_path)
    return (header, [spec_from_dict(row) for row in document['specs']])

def spawn_for(header, spec):
    values = spec.truth.get('spawn') or header['spawn']
    return Pose(**{key: values[key] for key in ('x', 'y', 'z', 'yaw_deg', 'horizon_deg')})

def dataset_identity(dataset):
    root = Path(dataset)
    files = sorted([root / 'manifest.json', *(root / 'tasksets').glob('*.json'), *(root / 'references').glob('*.png')])
    records = [(path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()) for path in files]
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()


def scene_for(header, spec, cache):
    from .scene_variants import prepare_scene_variant
    return prepare_scene_variant(header['scene'], spec.scene_variant, cache)
