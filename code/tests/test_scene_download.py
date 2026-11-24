"""Archive download safety and complete extraction checks."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile

import pytest


spec = importlib.util.spec_from_file_location('download_scenes', Path(__file__).parents[1] / 'scripts/download_scenes.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def make_archive(path, extra=None):
    entries = {
        'scenes/scene_148/house_layout.json': b'{}',
        'scenes/scene_148/combined_house/house_state.json': b'{}',
        'scenes/scene_148/combined_house/house.dmd.yaml': b'models: []',
    }
    with tarfile.open(path, 'w:gz') as archive:
        for name, data in entries.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        if extra:
            archive.addfile(extra, io.BytesIO(b''))
    return {'scene': 'scene_148', 'archive': path.name, 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'file_count': len(entries), 'unpacked_bytes': sum(map(len, entries.values()))}


def test_complete_extraction_and_overwrite_refusal(tmp_path):
    path = tmp_path / 'scene_148.tar.gz'
    record = make_archive(path)
    download.verify_archive(path, record)
    result = download.extract_scene(path, record, tmp_path / 'scenes')
    assert (result / 'combined_house/house_state.json').read_text() == '{}'
    with pytest.raises(FileExistsError):
        download.extract_scene(path, record, tmp_path / 'scenes')


@pytest.mark.parametrize('kind', ['traversal', 'symlink'])
def test_unsafe_members_do_not_install_scene(tmp_path, kind):
    path = tmp_path / 'scene_148.tar.gz'
    if kind == 'traversal':
        member = tarfile.TarInfo('scenes/scene_148/../../../outside')
    else:
        member = tarfile.TarInfo('scenes/scene_148/escape')
        member.type = tarfile.SYMTYPE
        member.linkname = '/tmp'
    record = make_archive(path, member)
    with pytest.raises(ValueError, match='Unsafe archive member'):
        download.extract_scene(path, record, tmp_path / 'scenes')
    assert not (tmp_path / 'scenes/scene_148').exists()
    assert not (tmp_path / 'outside').exists()


def test_modified_archive_is_rejected(tmp_path):
    path = tmp_path / 'scene_148.tar.gz'
    record = make_archive(path)
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match='checksum differs'):
        download.verify_archive(path, record)
