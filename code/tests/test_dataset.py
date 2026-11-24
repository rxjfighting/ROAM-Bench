"""Check released identities, references, scoring rules, and resume binding."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
from PIL import Image
from roam_bench.dataset import dataset_identity, load_scene
from roam_bench.protocol.score import match_answer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'dataset'

def test_complete_manifest_and_reference_integrity():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    assert manifest['task_count'] == len(manifest['tasks']) == 250
    assert Counter(row['group'] for row in manifest['tasks']) == {'G1': 49, 'G2': 38, 'G3': 77, 'G4': 56, 'G5': 30}
    assert len({row['task_type'] for row in manifest['tasks']}) == 11
    scenes = {row['scene'] for row in manifest['tasks']}
    assert len(scenes) == manifest['scene_count'] == 27
    actual = set()
    references = set()
    for scene in sorted(scenes):
        header, specs = load_scene(DATA, scene)
        assert header['scene'] == scene
        for spec in specs:
            actual.add((scene, spec.task.task_id, spec.family))
            assert spec.task.max_steps > 0
            references.update(Path(path) for _, path in spec.task.goal_images)
    assert actual == {(row['scene'], row['task_id'], row['family']) for row in manifest['tasks']}
    assert len(references) == 388
    for path in references:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == path.stem
        with Image.open(path) as image:
            assert max(image.size) <= 256
            assert not image.info
            image.verify()

def test_path_integration_remains_pose_dependent():
    specs = [spec for path in (DATA / 'tasksets').glob('*.json') for spec in load_scene(DATA, path.stem)[1] if spec.family == 'path_integration']
    assert specs
    assert all(spec.task.answer_from_pose and spec.task.answer_match == 'direction' for spec in specs)

def test_numeric_and_discrete_acceptance():
    assert match_answer('7', (), 'count', std=7)
    assert not match_answer('6', (), 'count', std=7)
    assert match_answer('2.04 m', (), 'count', std=2, tol=.1, unit='m')
    assert not match_answer('3 m', (), 'count', std=2, tol=.1, unit='m')
    assert match_answer('B', (('B',),), choices=('A', 'B', 'C'))
    assert not match_answer('A or B', (('B',),), choices=('A', 'B', 'C'))

def test_resume_identity_binds_reference_and_manifest(tmp_path):
    dataset = tmp_path / 'dataset'
    shutil.copytree(DATA, dataset)
    before = dataset_identity(dataset)
    image = next((dataset / 'references').glob('*.png'))
    image.write_bytes(image.read_bytes() + b'changed')
    assert dataset_identity(dataset) != before
    shutil.copy2(DATA / 'references' / image.name, image)
    assert dataset_identity(dataset) == before
    manifest = dataset / 'manifest.json'
    manifest.write_text(manifest.read_text() + '\n')
    assert dataset_identity(dataset) != before
