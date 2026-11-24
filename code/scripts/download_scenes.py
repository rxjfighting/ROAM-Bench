"""Download and safely extract selected ROAM-Bench scene archives."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile


def verify_archive(path, record):
    if path.stat().st_size != record['bytes']:
        raise ValueError('Archive size differs: ' + record['archive'])
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != record['sha256']:
        raise ValueError('Archive checksum differs: ' + record['archive'])


def extract_scene(path, record, output):
    """Reject unsafe members and install a complete scene without overwriting one."""
    scene = record['scene']
    destination = output / scene
    if destination.exists():
        raise FileExistsError('Scene already exists; use another output directory: ' + str(destination))
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.extract-', dir=output) as temporary:
        temporary = Path(temporary)
        with tarfile.open(path, 'r:gz') as archive:
            members = archive.getmembers()
            files = []
            for member in members:
                name = PurePosixPath(member.name)
                if (name.is_absolute() or '..' in name.parts or '\\' in member.name
                        or name.parts[:2] != ('scenes', scene)
                        or not (member.isdir() or member.isfile())):
                    raise ValueError('Unsafe archive member: ' + member.name)
                if member.isfile():
                    files.append(member)
            if len({member.name for member in files}) != len(files):
                raise ValueError('Archive contains duplicate file paths')
            if len(files) != record['file_count'] or sum(m.size for m in files) != record['unpacked_bytes']:
                raise ValueError('Archive contents differ from the scene manifest: ' + scene)
            # Python 3.11.8 added extraction filters. The validated member list
            # also keeps extraction safe on earlier Python 3.11 runtimes.
            options = {'filter': 'data'} if hasattr(tarfile, 'data_filter') else {}
            archive.extractall(temporary, members=members, **options)
        extracted = temporary / 'scenes' / scene
        for required in ('house_layout.json', 'combined_house/house_state.json', 'combined_house/house.dmd.yaml'):
            if not (extracted / required).is_file():
                raise ValueError('Scene entry is absent: ' + required)
        # The temporary directory is on the same filesystem as the destination.
        extracted.rename(destination)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--repo-id', help='Hugging Face dataset ID, e.g. YOUR_ACCOUNT/ROAM-Bench-scenes.')
    source.add_argument('--source-dir', type=Path, help='Local directory containing the scene archives.')
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--scene', action='append', help='Scene ID; may be repeated.')
    selection.add_argument('--all', action='store_true', help='Download all 27 scenes.')
    parser.add_argument('--output', type=Path, default=Path('scenes'))
    parser.add_argument('--cache-dir', type=Path, default=Path('.roam-cache/downloads'))
    args = parser.parse_args(argv)
    catalog = json.loads((Path(__file__).resolve().parents[1] / 'scene_archives.json').read_text())
    records = {row['scene']: row for row in catalog['archives']}
    selected = sorted(records) if args.all else list(dict.fromkeys(args.scene))
    unknown = set(selected) - records.keys()
    if unknown:
        parser.error('Unknown scenes: ' + ', '.join(sorted(unknown)))
    output = args.output.resolve()
    existing = [scene for scene in selected if (output / scene).exists()]
    if existing:
        parser.error('Scenes already exist; select only missing scenes or another output directory: ' + ', '.join(existing))
    if args.repo_id:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            parser.error("Install Hugging Face support with: python -m pip install -e './code[hub]'")
    for scene in selected:
        record = records[scene]
        print('Obtaining ' + record['archive'], flush=True)
        if args.source_dir:
            path = args.source_dir / record['archive']
        else:
            path = Path(hf_hub_download(repo_id=args.repo_id, repo_type='dataset',
                                       filename=record['archive'], cache_dir=str(args.cache_dir)))
        verify_archive(path, record)
        destination = extract_scene(path, record, output)
        print('Installed ' + str(destination), flush=True)
    # Global scene metadata accompanies the archives in the environment repository.
    for name in ('manifest.json', 'asset_provenance.json', 'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md'):
        if args.source_dir:
            path = args.source_dir / name
            if not path.is_file():
                continue
        else:
            path = Path(hf_hub_download(repo_id=args.repo_id, repo_type='dataset',
                                       filename=name, cache_dir=str(args.cache_dir)))
        if not (output / name).exists():
            shutil.copy2(path, output / name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
