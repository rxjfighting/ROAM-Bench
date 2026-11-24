"""Create a G1 Inspire entry from supplied official assets with valid wrist references."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Official Unitree/G1 directory containing g1.usd.')
    parser.add_argument('--output', type=Path, required=True, help='New directory for derived USD layers.')
    args = parser.parse_args(argv)
    source_root = args.source.resolve()
    output = args.output.resolve()
    if not (source_root / 'g1.usd').is_file():
        parser.error('Source must contain the official g1.usd entry.')
    if output.exists() and any(output.iterdir()):
        parser.error('Output must be absent or empty.')
    if output == source_root or output.is_relative_to(source_root):
        parser.error('Output must be outside the original asset directory.')
    from pxr import Sdf, Usd, UsdUtils
    sources = set(source_root.rglob('*.usd')) | set(source_root.rglob('*.usda'))
    inventory = {path: sha256(path) for path in sources}
    dependencies = {}
    for path in sorted(sources):
        original = Sdf.Layer.FindOrOpen(str(path))
        if original is None:
            raise ValueError('Unreadable USD layer: ' + path.name)
        layer = Sdf.Layer.CreateAnonymous()
        layer.TransferContent(original)
        edges = set()
        def collect(asset):
            if asset and '://' not in asset:
                resolved = (path.parent / asset).resolve()
                if resolved in sources:
                    edges.add(resolved)
            return asset
        UsdUtils.ModifyAssetPaths(layer, collect)
        dependencies[path] = edges
    selected = {path for path in sources if path.name in ('inspire_left_base.usd', 'inspire_right_base.usd')}
    if len(selected) != 2:
        raise ValueError('Expected both Inspire hand base layers.')
    while True:
        expanded = selected | {path for path, edges in dependencies.items() if edges & selected}
        if expanded == selected:
            break
        selected = expanded
    entry_source = source_root / 'g1.usd'
    if entry_source not in selected:
        raise ValueError('Hand layers are not connected to g1.usd.')
    output.mkdir(parents=True, exist_ok=True)
    mapping = {path: output / 'layers' / path.relative_to(source_root) for path in selected}
    edits = []
    records = []
    for path in sorted(selected):
        destination = mapping[path]
        destination.parent.mkdir(parents=True, exist_ok=True)
        original = Sdf.Layer.FindOrOpen(str(path))
        layer = Sdf.Layer.CreateNew(str(destination))
        layer.TransferContent(original)
        def remap(asset):
            if not asset or '://' in asset:
                return asset
            resolved = (path.parent / asset).resolve()
            if not resolved.is_file():
                return asset
            target = mapping.get(resolved, resolved)
            return os.path.relpath(target, destination.parent).replace(os.sep, '/')
        UsdUtils.ModifyAssetPaths(layer, remap)
        if path.name in ('inspire_left_base.usd', 'inspire_right_base.usd'):
            side = 'left' if 'left' in path.name else 'right'
            letter = 'L' if side == 'left' else 'R'
            for branch in ('visuals', 'colliders'):
                name = f'/{branch}/{letter}_hand_base_link/{side}_wrist_yaw_link'
                prim = layer.GetPrimAtPath(name)
                if prim is None:
                    raise ValueError('Expected wrist prim is absent: ' + name)
                references = list(prim.referenceList.prependedItems)
                target = f'/meshes/{side}_wrist_yaw_link'
                if len(references) != 1 or references[0].assetPath or str(references[0].primPath) != target or original.GetPrimAtPath(target) is not None:
                    raise ValueError('Unexpected wrist reference structure: ' + name)
                prim.referenceList.ClearEdits()
                prim.referenceList.explicitItems = []
                edits.append({'layer': destination.relative_to(output).as_posix(), 'prim': name, 'removed_target': target})
        layer.Save()
        records.append({'source_layer': path.relative_to(source_root).as_posix(), 'source_sha256': inventory[path], 'derived_layer': destination.relative_to(output).as_posix(), 'derived_sha256': sha256(destination)})
    if len(edits) != 4:
        raise ValueError('Expected exactly four absent-target wrist references.')
    entry = output / 'g1_inspire.usda'
    stage = Usd.Stage.CreateNew(str(entry))
    root = stage.DefinePrim('/G1', 'Xform')
    root.GetReferences().AddReference(os.path.relpath(mapping[entry_source], output).replace(os.sep, '/'))
    for side in ('left_hand', 'right_hand'):
        if not root.GetVariantSet(side).SetVariantSelection('Inspire'):
            raise ValueError('Inspire hand variant is absent.')
    stage.SetDefaultPrim(root)
    stage.GetRootLayer().Save()
    if any(sha256(path) != digest for path, digest in inventory.items()):
        raise ValueError('Original asset content changed during construction.')
    receipt = {'schema': 'roam-g1-asset-v1', 'entry': entry.name, 'layers': records, 'removed_empty_references': edits, 'source_assets_modified': False}
    (output / 'manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'entry': str(entry), 'derived_layers': len(records), 'removed_empty_references': len(edits)}))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
