"""Validate task references, scene bindings, asset links, and optional checksums."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roam_bench.dataset import load_scene, scene_for
from roam_bench.scene_facts import load_facts
from roam_bench.scene_loader import load_scene as load_geometry
from build_scene_bundle import verify_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--checksums', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    dataset, scenes = root / 'dataset', root / 'scenes'
    manifest = json.loads((dataset / 'manifest.json').read_text())
    errors = verify_bundle(scenes)
    expected = {(row['scene'], row['task_id']) for row in manifest['tasks']}
    actual = set()
    count = 0
    with tempfile.TemporaryDirectory(prefix='roam-scene-check-') as cache:
        for name in sorted({scene for scene, _ in expected}):
            header, specs = load_scene(dataset, name, scenes)
            checked = {}
            for spec in specs:
                path = scene_for(header, spec, cache)
                if path not in checked:
                    facts = load_facts(path)
                    geometry = load_geometry(path)
                    checked[path] = (set(facts.by_alias), {item.name for item in geometry.instances})
                aliases, rendered = checked[path]
                required = set(spec.referents) | set(spec.evidence)
                if spec.evidence_bound:
                    required |= set(spec.evidence_bound.must_see) | set(spec.evidence_bound.must_open)
                missing = required - aliases
                if missing:
                    errors.append(name + '/' + spec.task.task_id + ': missing object aliases ' + ', '.join(sorted(missing)))
                absent = required - rendered
                if absent:
                    errors.append(name + '/' + spec.task.task_id + ': missing visual geometry ' + ', '.join(sorted(absent)))
                actual.add((name, spec.task.task_id))
                count += 1
    if actual != expected or count != manifest['task_count']:
        errors.append('Task manifest does not match the task files')
    references = list((dataset / 'references').glob('*.png'))
    for path in references:
        if hashlib.sha256(path.read_bytes()).hexdigest() != path.stem:
            errors.append('Reference-image hash differs: ' + path.name)
    if args.checksums:
        for line in (root / 'SHA256SUMS').read_text().splitlines():
            digest, relative = line.split('  ', 1)
            path = root / relative
            if not path.resolve().is_relative_to(root):
                errors.append('Checksum path leaves release directory: ' + relative)
                continue
            actual_digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    actual_digest.update(block)
            if actual_digest.hexdigest() != digest:
                errors.append('Checksum differs: ' + relative)
    for error in errors:
        print(error)
    print(json.dumps({'tasks': count, 'scenes': manifest['scene_count'], 'references': len(references), 'errors': len(errors)}))
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
