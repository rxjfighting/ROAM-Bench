"""Check an eligible same-template room witness as one complete observation."""
from copy import deepcopy
POLICY_NAME = 'roam-room-template-corroboration-v1'

def template_sufficiency(policy, spec, trace, necessary, sufficient, peers):
    if str(policy._get(spec, 'family', '')) != 'room_localization' or necessary.get('status') != policy.PASS or sufficient.get('status') == policy.PASS:
        return sufficient
    truth = policy._truth(spec)
    target = str(truth.get('alias') or '')
    room = str(truth.get('room_id') or truth.get('room') or '')
    kind = str(truth.get('kind') or '')
    if not target or not room or (not kind):
        return sufficient
    candidates = sorted({str(row['alias']) for row in peers if row.get('alias') and row['alias'] != target and (row.get('room') == room) and (row.get('kind') == kind) and row.get('visual_template_sha256')})
    for alias in candidates:
        rows = policy._object_rows(trace, alias)
        if not any((str(row.get('kind') or '') == kind for _, row in rows)):
            continue
        binding = policy._room_binding({alias: room}, trace)
        clear = policy._clear_all({alias}, trace)
        if binding.get('status') == policy.PASS and clear.get('status') == policy.PASS:
            corroborated = policy._combine([deepcopy(necessary), binding, clear])
            corroborated.update(witness_policy=POLICY_NAME, original_target=target, accepted_witness=alias, expected_room=room, witness_basis='verified_exact_loaded_template', rgb_recognition_certified=False)
            return corroborated
    return sufficient

def judge(policy, spec, trace, answer_success, peers):
    original = policy.judge(spec, trace, answer_success)
    corroborated = template_sufficiency(policy, spec, trace, original['necessary_evidence'], original['sufficient_evidence'], peers)
    result = deepcopy(original)
    result['sufficient_evidence'] = corroborated
    result['policy_version'] = str(original['policy_version']) + '+' + POLICY_NAME
    return result

def install(policy, peer_lookup):
    if getattr(policy, '_room_localization_copy_sufficiency_installed_v1', False):
        raise ValueError('Duplicate room-of copy sufficiency installation')
    original = policy._family

    def family(spec, trace):
        necessary, sufficient = original(spec, trace)
        return (necessary, template_sufficiency(policy, spec, trace, necessary, sufficient, peer_lookup(spec, trace)))
    policy._family = family
    policy.POLICY_VERSION += '+' + POLICY_NAME
    policy._room_localization_copy_sufficiency_installed_v1 = True
