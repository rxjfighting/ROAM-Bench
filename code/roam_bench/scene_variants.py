"""Build task-scoped scene views without modifying shared scene assets."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import yaml

from .scene_loader import _AngleAxis, _DrakeLoader, _find_combined, _resolve_package

SCHEMA = 'task-scene-variant-v1'


class _DirectiveDumper(yaml.SafeDumper):
    pass


_DirectiveDumper.add_representer(
    _AngleAxis,
    lambda dumper, value: dumper.represent_mapping(
        '!AngleAxis', {'angle_deg': value.angle_deg, 'axis': value.axis}),
)


def _tuple(dumper, value):
    if len(value) == 2 and value[0] == 'Rpy' and isinstance(value[1], dict):
        return dumper.represent_mapping('!Rpy', value[1])
    return dumper.represent_list(value)


_DirectiveDumper.add_representer(tuple, _tuple)


def validate_variant(value, protected=()):
    if value is None or value == {}:
        return {}
    if not isinstance(value, dict) or set(value) - {'schema', 'exclude', 'reposition'}:
        raise ValueError('Unknown scene-variant fields.')
    if value.get('schema') != SCHEMA:
        raise ValueError('Unsupported scene-variant schema.')
    excluded = value.get('exclude', [])
    if not isinstance(excluded, list) or not excluded or any(
            not isinstance(a, str) or not a for a in excluded) or len(set(excluded)) != len(excluded):
        raise ValueError('A scene variant needs distinct excluded aliases.')
    if set(excluded) & set(protected):
        raise ValueError('A scene variant excludes a task-required object.')
    moves = value.get('reposition', [])
    if not isinstance(moves, list):
        raise ValueError('Reposition entries must be a list.')
    seen = set()
    for move in moves:
        if not isinstance(move, dict) or set(move) != {'alias', 'translation', 'support'}:
            raise ValueError('A reposition entry needs alias, translation, and support.')
        alias = move['alias']
        xyz = move['translation']
        if not isinstance(alias, str) or not alias or alias in seen or alias in excluded:
            raise ValueError('Invalid repositioned alias.')
        if move['support'] != 'floor':
            raise ValueError('Only explicit floor support is supported.')
        if not isinstance(xyz, list) or len(xyz) != 3 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or
                not math.isfinite(v) for v in xyz):
            raise ValueError('Reposition translation must contain three finite numbers.')
        seen.add(alias)
    if seen & set(protected):
        raise ValueError('A scene variant repositions a task-required object.')
    return copy.deepcopy(value)


def _index(state):
    result = {}
    for room, record in state['rooms'].items():
        for key, obj in record.get('objects', {}).items():
            alias = room + '_' + str(obj.get('object_id') or key)
            if alias in result:
                raise ValueError('Duplicate scene alias: ' + alias)
            result[alias] = (room, key, obj)
    return result


def _variant_documents(source, config):
    combined, directive_path = _find_combined(str(source))
    combined = Path(combined)
    state = json.loads((combined / 'house_state.json').read_text())
    index = _index(state)
    excluded = set(config['exclude'])
    moved = {entry['alias']: entry for entry in config.get('reposition', [])}
    missing = (excluded | set(moved)) - set(index)
    if missing:
        raise ValueError('Scene-variant aliases are absent: ' + ', '.join(sorted(missing)))
    model_names = set(excluded)
    for alias in excluded:
        room, key, obj = index[alias]
        model_names.update((obj.get('metadata') or {}).get('member_model_names') or [])
        surfaces = {s['surface_id'] for s in obj.get('support_surfaces', []) if 'surface_id' in s}
        for child_alias, (child_room, _, child) in index.items():
            parent = (child.get('placement_info') or {}).get('parent_surface_id')
            if child_room == room and parent in surfaces and child_alias not in excluded | set(moved):
                raise ValueError('Excluded support still holds an object: ' + child_alias)
        del state['rooms'][room]['objects'][key]
    for alias, move in moved.items():
        room, _, obj = index[alias]
        low = float((obj.get('bbox_min') or [0, 0, 0])[2])
        if abs(float(move['translation'][2]) + low) > 0.005:
            raise ValueError('Repositioned object does not rest on the floor: ' + alias)
        obj['transform']['translation'] = list(move['translation'])
        obj['transform']['rotation_wxyz'] = [1.0, 0.0, 0.0, 0.0]
        obj['placement_info'] = {
            'parent_surface_id': room + '_floor', 'position_2d': list(move['translation'][:2]),
            'rotation_2d': 0.0, 'placement_method': 'floor_placement',
        }
    document = yaml.load(Path(directive_path).read_text(), Loader=_DrakeLoader)
    directives = []
    for row in document['directives']:
        model = row.get('add_model')
        weld = row.get('add_weld')
        if model and model['name'] in model_names | set(moved):
            continue
        if weld and weld['child'].split('::', 1)[0] in model_names | set(moved):
            continue
        if model:
            model['file'] = str(Path(_resolve_package(model['file'], str(source))).resolve())
        directives.append(row)
    document['directives'] = directives
    return combined, Path(directive_path), state, document


def prepare_scene_variant(scene, variant, cache):
    """Return a content-addressed scene directory shared by every runtime reader."""
    config = validate_variant(variant)
    source = Path(scene).resolve()
    if not config:
        return str(source)
    combined, directive_path = _find_combined(str(source))
    inputs = {'source': str(source), 'config': config, 'builder_version': 1,
              'state': hashlib.sha256((Path(combined) / 'house_state.json').read_bytes()).hexdigest(),
              'directives': hashlib.sha256(Path(directive_path).read_bytes()).hexdigest()}
    identity = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    cache = Path(cache).resolve()
    if cache.is_relative_to(source):
        raise ValueError('The scene cache must be outside the source scene.')
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / identity / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_dir():
        receipt = json.loads((destination / 'scene_variant.json').read_text())
        if receipt['identity'] != identity:
            raise ValueError('Cached scene-variant identity differs.')
        for name, digest in receipt['generated_sha256'].items():
            if hashlib.sha256((destination / name).read_bytes()).hexdigest() != digest:
                raise ValueError('Cached scene variant was modified.')
        return str(destination)
    combined, directive_path, state, document = _variant_documents(source, config)
    temporary = Path(tempfile.mkdtemp(prefix='.building-', dir=cache))
    try:
        for entry in source.iterdir():
            if entry.name != combined.name:
                (temporary / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
        target = temporary / combined.name
        target.mkdir()
        for entry in combined.iterdir():
            if entry.name not in {directive_path.name, 'house_state.json', 'house_state.json.bak'}:
                (target / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
        state_path = target / 'house_state.json'
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        dmd_path = target / directive_path.name
        dmd_path.write_text(yaml.dump(document, Dumper=_DirectiveDumper, sort_keys=False))
        generated = {p.relative_to(temporary).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (state_path, dmd_path)}
        (temporary / 'scene_variant.json').write_text(json.dumps({
            'schema': SCHEMA, 'identity': identity, 'inputs': inputs, 'generated_sha256': generated,
        }, indent=2) + '\n')
        try:
            temporary.rename(destination)
        except OSError:
            if not destination.is_dir():
                raise
            shutil.rmtree(temporary)
            return prepare_scene_variant(source, config, cache)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return str(destination)


def relation_source(scene_root, house_path, current_sha256):
    """Retain source identity for unchanged, independently checked support relations."""
    root = Path(scene_root)
    receipt_path = root / 'scene_variant.json'
    if not receipt_path.is_file():
        return root.name, current_sha256
    receipt = json.loads(receipt_path.read_text())
    relative = Path(house_path).relative_to(root).as_posix()
    if receipt.get('schema') != SCHEMA or receipt['generated_sha256'].get(relative) != current_sha256:
        raise ValueError('Scene-variant relation metadata was modified.')
    return Path(receipt['inputs']['source']).name, receipt['inputs']['state']
