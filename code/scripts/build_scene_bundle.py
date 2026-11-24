"""Export portable scene sources with a complete local asset dependency graph."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from urllib.parse import unquote

COMBINED_DIRS = ('combined_house_after_furniture', 'combined_house')
DMD_NAMES = ('house_furniture_welded.dmd.yaml', 'house.dmd.yaml')
URI_RE = re.compile(r'<uri>\s*([^<]+?)\s*</uri>')
DMD_FILE_RE = re.compile(r'^(\s*file:\s*)([^\s#]+)', re.MULTILINE)
COMMENT_RE = re.compile(r'<!--.*?-->', re.DOTALL)
PATH_KEYS = {'sdf_path', 'geometry_path', 'path'}
OMIT_KEYS = {'timestamp', 'generation_timestamp', 'image_path'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def portable(raw, scene):
    normal = raw.replace('\\', '/')
    marker = '/' + scene.name + '/'
    if marker in normal:
        return normal.split(marker, 1)[1]
    if normal.startswith('package://scene/'):
        return normal[len('package://scene/'):]
    return normal


def resolve(raw, scene, base=None, room=None, shared=None, allow_directory=False):
    normal = unquote(portable(raw, scene))
    if '://' in normal or normal.startswith('data:'):
        raise ValueError('Scene assets must be local: ' + raw)
    path = Path(normal)
    candidates = [path] if path.is_absolute() else []
    if base is not None:
        candidates.append(base / path)
    if room:
        candidates.append(scene / ('room_' + room) / path)
    candidates.append(scene / path)
    if shared and 'materials/' in normal:
        candidates.append(shared / normal.split('materials/', 1)[1])
    for candidate in candidates:
        if candidate.is_file() or (allow_directory and candidate.is_dir()):
            result = candidate.resolve()
            if not result.is_relative_to(scene) and not (shared and result.is_relative_to(shared)):
                raise ValueError('Asset is outside the scene and shared materials: ' + raw)
            return result
    raise FileNotFoundError('Missing scene dependency: ' + raw)


def clean_state(value, scene, add, room=None):
    if isinstance(value, dict):
        room = value.get('room_id') or room
        result = {}
        for key, item in value.items():
            if key in OMIT_KEYS:
                continue
            if key == 'rooms' and isinstance(item, dict):
                result[key] = {rid: clean_state(record, scene, add, rid) for rid, record in item.items()}
            elif key in PATH_KEYS and isinstance(item, str) and item:
                try:
                    source = resolve(item, scene, room=room, shared=add.shared, allow_directory=key == 'path')
                except FileNotFoundError:
                    if key == 'path':
                        continue
                    if key != 'geometry_path' or not value.get('sdf_path'):
                        raise
                    sdf = resolve(value['sdf_path'], scene, room=room, shared=add.shared)
                    visuals = re.findall(r'<visual\b[^>]*>(.*?)</visual>', sdf.read_text(), re.DOTALL)
                    raw = next((uri for visual in visuals for uri in URI_RE.findall(visual)), None)
                    if raw is None:
                        result[key] = None
                        continue
                    source = resolve(raw, scene, base=sdf.parent, shared=add.shared)
                    item = str(source)
                add(source)
                normal = portable(item, scene)
                result[key] = add.relative(source).as_posix() if Path(item).is_absolute() or not (scene / ('room_' + str(room)) / normal).is_file() else normal
            else:
                result[key] = clean_state(item, scene, add, room)
        return result
    if isinstance(value, list):
        return [clean_state(item, scene, add, room) for item in value]
    if isinstance(value, str):
        return portable(value, scene)
    return value


def export_scene(scene, output, shared=None):
    scene = scene.resolve()
    shared = shared.resolve() if shared else None
    destination = output / scene.name
    destination.mkdir(parents=True, exist_ok=False)
    combined = next((scene / name for name in COMBINED_DIRS if (scene / name).is_dir()), None)
    if combined is None:
        raise FileNotFoundError('Missing combined scene directives: ' + scene.name)
    dmd = next((combined / name for name in DMD_NAMES if (combined / name).is_file()), None)
    state_path = next((combined / name for name in ('house_state.json', 'house_state.json.bak') if (combined / name).is_file()), None)
    if dmd is None or state_path is None:
        raise FileNotFoundError('Missing scene state or directives: ' + scene.name)
    queue = deque()
    visited = set()

    def relative(path):
        if path.is_relative_to(scene):
            return path.relative_to(scene)
        return Path('materials') / path.relative_to(shared)

    def add(path):
        if path.is_dir():
            for texture in sorted(path.rglob('*')):
                if texture.suffix.lower() in {'.png', '.jpg', '.jpeg', '.exr', '.hdr', '.tif', '.tiff'} and texture.is_file():
                    add(texture)
            return
        if path not in visited:
            visited.add(path)
            queue.append(path)

    add.shared, add.relative = shared, relative
    raw_state = json.loads(state_path.read_text())
    state = clean_state(raw_state, scene, add)
    target_combined = destination / 'combined_house'
    target_combined.mkdir()
    (target_combined / 'house_state.json').write_text(json.dumps(state, indent=2) + '\n')
    if (scene / 'house_layout.json').is_file():
        layout = clean_state(json.loads((scene / 'house_layout.json').read_text()), scene, add)
        (destination / 'house_layout.json').write_text(json.dumps(layout, indent=2) + '\n')
    package = scene / 'package.xml'
    if not package.is_file():
        raise FileNotFoundError('Missing scene package declaration: ' + scene.name)
    (destination / 'package.xml').write_text(COMMENT_RE.sub('', package.read_text()))
    document = dmd.read_text()

    def directive(match):
        source = resolve(match.group(2).strip('"\''), scene, shared=shared)
        add(source)
        return match.group(1) + 'package://scene/' + relative(source).as_posix()

    document = DMD_FILE_RE.sub(directive, document)
    document = '\n'.join(line for line in document.splitlines() if not line.lstrip().startswith('#')) + '\n'
    (target_combined / 'house.dmd.yaml').write_text(document)
    while queue:
        source = queue.popleft()
        target = destination / relative(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        suffix = source.suffix.lower()
        if suffix in {'.sdf', '.xml'}:
            text = COMMENT_RE.sub('', source.read_text())

            def uri(match):
                dependency = resolve(match.group(1), scene, base=source.parent, shared=shared)
                add(dependency)
                ref = os.path.relpath(destination / relative(dependency), target.parent).replace(os.sep, '/')
                return '<uri>' + ref + '</uri>'

            target.write_text(URI_RE.sub(uri, text))
        elif suffix == '.gltf':
            data = json.loads(source.read_text())
            for field in ('buffers', 'images'):
                for entry in data.get(field, []):
                    raw = entry.get('uri')
                    if not raw or raw.startswith('data:'):
                        continue
                    dependency = resolve(raw, scene, base=source.parent, shared=shared)
                    add(dependency)
                    entry['uri'] = os.path.relpath(destination / relative(dependency), target.parent).replace(os.sep, '/')
            target.write_text(json.dumps(data, separators=(',', ':')) + '\n')
        elif suffix in {'.obj', '.mtl'}:
            lines = []
            for line in source.read_text().splitlines():
                if line.lstrip().startswith('#'):
                    continue
                words = line.split()
                if words and (words[0] == 'mtllib' or (suffix == '.mtl' and words[0].lower().startswith('map_'))):
                    dependency = resolve(words[-1], scene, base=source.parent, shared=shared)
                    add(dependency)
                    words[-1] = os.path.relpath(destination / relative(dependency), target.parent).replace(os.sep, '/')
                    line = ' '.join(words)
                lines.append(line)
            target.write_text('\n'.join(lines) + '\n')
        else:
            shutil.copyfile(source, target)
    origins = Counter()

    def count_origins(value):
        if isinstance(value, dict):
            if value.get('asset_source'):
                origins[str(value['asset_source'])] += 1
            for item in value.values():
                count_origins(item)
        elif isinstance(value, list):
            for item in value:
                count_origins(item)

    count_origins(raw_state)
    files = sorted(p for p in destination.rglob('*') if p.is_file())
    return {'scene': scene.name, 'file_count': len(files), 'bytes': sum(p.stat().st_size for p in files), 'asset_source_records': dict(origins)}


def verify_bundle(output):
    output = Path(output).resolve()
    files = []
    for folder, _, names in os.walk(output):
        files.extend(Path(folder) / name for name in names)
    existing = {str(path) for path in files}

    def check(raw, path, scene):
        normal = unquote(raw.replace('\\', '/'))
        if normal.startswith('data:'):
            return None
        if normal.startswith('package://scene/'):
            candidates = [scene / normal[len('package://scene/'):]]
        elif '://' in normal or Path(normal).is_absolute():
            return 'Non-local asset reference: ' + str(path.relative_to(output))
        else:
            candidates = [path.parent / normal, scene / normal]
        if not any(os.path.abspath(candidate) in existing for candidate in candidates):
            return 'Missing asset ' + repr(raw) + ': ' + str(path.relative_to(output))
        return None

    def inspect(path):
        relative = path.relative_to(output)
        if not relative.parts[0].startswith('scene_'):
            return []
        scene = output / relative.parts[0]
        suffix = path.suffix.lower()
        errors = []
        if suffix in {'.bak', '.blend', '.blend1', '.log'}:
            errors.append('Unexpected non-runtime file: ' + str(relative))
        if suffix not in {'.json', '.yaml', '.sdf', '.xml', '.gltf', '.mtl'}:
            return errors
        text = path.read_text()
        if any(marker in text for marker in ('/export/', '/mnt/', '/home/', '/Users/')):
            errors.append('Absolute host path: ' + str(relative))
        references = []
        if suffix == '.sdf':
            references = URI_RE.findall(text)
        elif path.name.endswith('.dmd.yaml'):
            references = [match.group(2) for match in DMD_FILE_RE.finditer(text)]
        elif suffix == '.gltf':
            document = json.loads(text)
            references = [entry['uri'] for field in ('buffers', 'images') for entry in document.get(field, []) if entry.get('uri')]
        elif suffix == '.mtl':
            references = [words[-1] for line in text.splitlines() if (words := line.split()) and words[0].lower().startswith('map_')]
        errors.extend(error for raw in references if (error := check(raw, path, scene)))
        return errors

    with ThreadPoolExecutor(max_workers=8) as executor:
        errors = [error for result in executor.map(inspect, files) for error in result]
    for scene in sorted(output.glob('scene_*')):
        for name in ('house_state.json', 'house.dmd.yaml'):
            if str(scene / 'combined_house' / name) not in existing:
                errors.append(scene.name + ': missing ' + name)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=Path(__file__).resolve().parents[2] / 'dataset')
    parser.add_argument('--shared-materials-root', type=Path)
    parser.add_argument('--scenes', nargs='+')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        errors = verify_bundle(args.output.resolve())
        for error in errors:
            print(error, flush=True)
        print('Dependency errors:', len(errors), flush=True)
        return bool(errors)
    if args.source_root is None:
        parser.error('--source-root is required for export')
    if args.workers < 1:
        parser.error('--workers must be positive')
    scenes = args.scenes or sorted({row['scene'] for row in json.loads((args.dataset / 'manifest.json').read_text())['tasks']})
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error('Output directory must be empty')
    summaries = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(export_scene, args.source_root / name, args.output, args.shared_materials_root) for name in scenes]
        for future in as_completed(futures):
            summary = future.result()
            summaries.append(summary)
            print(summary['scene'], summary['file_count'], 'files,', round(summary['bytes'] / 1024**2, 1), 'MiB', flush=True)
    summaries.sort(key=lambda row: row['scene'])
    errors = verify_bundle(args.output.resolve())
    if errors:
        raise SystemExit('\n'.join(errors))
    manifest = {'format': 'ROAM-Bench-scenes-v1', 'scene_count': len(summaries), 'scenes': summaries}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Scene export complete:', len(summaries), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
