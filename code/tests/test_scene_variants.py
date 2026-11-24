"""Check task-scoped exclusions across geometry, facts, and runtime metadata."""
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from roam_bench.protocol.taskset import spec_from_dict
from roam_bench.scene_loader import load_scene
from roam_bench.scene_facts import load_facts
from roam_bench.scene_variants import prepare_scene_variant, validate_variant, relation_source, SCHEMA


def hashes(path):
    return {p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob('*') if p.is_file()}


@pytest.fixture
def scene(tmp_path):
    root = tmp_path / 'source'
    combined = root / 'combined_house_after_furniture'
    combined.mkdir(parents=True)
    sdf = root / 'mesh.sdf'
    sdf.write_text('<sdf><model name="fixture"><link name="base_link"><visual name="v">'
                   '<geometry><mesh><uri>mesh.gltf</uri></mesh></geometry></visual></link></model></sdf>')
    (root / 'mesh.gltf').write_text('{}')
    def obj(name, kind='manipuland', z=0):
        return {'object_id': name, 'name': name, 'object_type': kind, 'sdf_path': str(sdf),
                'transform': {'translation': [0, 0, z], 'rotation_wxyz': [1,0,0,0]},
                'bbox_min': [-.1,-.1,0], 'bbox_max': [.1,.1,.2], 'support_surfaces': []}
    objects = {'target': obj('target'), 'distractor': obj('distractor'),
               'table': obj('table', 'furniture'), 'child': obj('child', z=1),
               'tray': obj('tray')}
    objects['table']['support_surfaces'] = [{'surface_id': 'table_top'}]
    objects['child']['placement_info'] = {'parent_surface_id': 'table_top'}
    objects['tray']['metadata'] = {'composite_type': 'filled_container',
        'member_model_names': ['tray_container', 'tray_card']}
    state = {'layout': {'rooms': [{'id':'r','type':'room','width':4,'length':4}],
                        'placed_rooms': [{'room_id':'r','width':4,'depth':4}]},
             'rooms': {'r': {'objects': objects}}}
    (combined / 'house_state.json').write_text(json.dumps(state))
    directives = [{'add_frame': {'name': 'room_r_frame', 'X_PF': {'base_frame':'world','translation':[0,0,0]}}}]
    for name in ['r_target','r_distractor','r_table','tray_container','tray_card']:
        directives.extend([{'add_model': {'name':name,'file':str(sdf)}},
            {'add_weld': {'parent':'room_r_frame','child':name+'::base_link','X_PC':{'translation':[0,0,0]}}}])
    (combined / 'house_furniture_welded.dmd.yaml').write_text(yaml.safe_dump({'directives':directives}))
    return root


def variant(*aliases, reposition=()):
    return {'schema': SCHEMA, 'exclude': list(aliases), 'reposition': list(reposition)}


def test_exclusion_is_shared_by_meshes_and_facts_and_preserves_source(scene, tmp_path):
    before = hashes(scene)
    path = prepare_scene_variant(scene, variant('r_distractor'), tmp_path / 'cache')
    assert 'r_distractor' not in {o.name for o in load_scene(path).instances}
    facts = load_facts(path)
    assert 'r_distractor' not in facts.by_alias
    assert 'r_target' in facts.by_alias
    assert hashes(scene) == before
    assert 'r_distractor' in load_facts(str(scene)).by_alias
    assert prepare_scene_variant(scene, variant('r_distractor'), tmp_path / 'cache') == path


def test_composite_members_do_not_reappear(scene, tmp_path):
    path = prepare_scene_variant(scene, variant('r_tray'), tmp_path / 'cache')
    names = {o.name for o in load_scene(path).instances}
    assert not names & {'r_tray','tray_container','tray_card'}
    assert 'r_tray' not in load_facts(path).by_alias


def test_support_removal_requires_explicit_reposition(scene, tmp_path):
    with pytest.raises(ValueError, match='still holds'):
        prepare_scene_variant(scene, variant('r_table'), tmp_path / 'cache')
    moved = {'alias':'r_child','translation':[0,0,.002],'support':'floor'}
    path = prepare_scene_variant(scene, variant('r_table', reposition=[moved]), tmp_path / 'cache')
    state = json.loads((Path(path)/'combined_house_after_furniture/house_state.json').read_text())
    assert state['rooms']['r']['objects']['child']['transform']['translation'] == [0,0,.002]
    assert 'r_table' not in load_facts(path).by_alias
    assert 'r_child' in load_facts(path).by_alias


def test_variants_and_source_changes_have_distinct_cache_paths(scene, tmp_path):
    cache = tmp_path / 'cache'
    one = prepare_scene_variant(scene, variant('r_distractor'), cache)
    two = prepare_scene_variant(scene, variant('r_tray'), cache)
    assert one != two
    state = scene / 'combined_house_after_furniture/house_state.json'
    state.write_text(state.read_text() + '\n')
    assert prepare_scene_variant(scene, variant('r_distractor'), cache) != one
    assert prepare_scene_variant(scene, {}, cache) == str(scene.resolve())


def test_tampered_cache_and_invalid_aliases_are_rejected(scene, tmp_path):
    cache = tmp_path / 'cache'
    path = prepare_scene_variant(scene, variant('r_distractor'), cache)
    state = Path(path)/'combined_house_after_furniture/house_state.json'
    state.write_text(state.read_text() + '\n')
    with pytest.raises(ValueError, match='modified'):
        prepare_scene_variant(scene, variant('r_distractor'), cache)
    with pytest.raises(ValueError, match='absent'):
        prepare_scene_variant(scene, variant('missing'), cache)
    with pytest.raises(ValueError, match='outside'):
        prepare_scene_variant(scene, variant('r_tray'), scene/'cache')


def test_required_objects_are_protected_and_variant_survives_serialization():
    raw = {'task_id':'test','instruction':'Find the target.','family':'room_localization',
           'referents':['r_target'],'scene_variant':variant('r_distractor')}
    spec = spec_from_dict(raw)
    assert spec_from_dict(spec.to_dict()).scene_variant == spec.scene_variant
    raw['scene_variant'] = variant('r_target')
    with pytest.raises(ValueError, match='task-required'):
        spec_from_dict(raw)
    with pytest.raises(ValueError, match='repositions'):
        validate_variant(variant('r_distractor', reposition=[
            {'alias':'r_target','translation':[0,0,0],'support':'floor'}]), ['r_target'])


def test_source_relation_provenance_requires_unmodified_generated_state(scene, tmp_path):
    path = Path(prepare_scene_variant(scene, variant('r_distractor'), tmp_path / 'cache'))
    assert path.name == scene.name
    state = path/'combined_house_after_furniture/house_state.json'
    digest = hashlib.sha256(state.read_bytes()).hexdigest()
    name, original_digest = relation_source(path, state, digest)
    assert name == scene.name
    assert original_digest == hashlib.sha256((scene/'combined_house_after_furniture/house_state.json').read_bytes()).hexdigest()
    state.write_text(state.read_text() + '\n')
    with pytest.raises(ValueError, match='modified'):
        relation_source(path, state, hashlib.sha256(state.read_bytes()).hexdigest())
