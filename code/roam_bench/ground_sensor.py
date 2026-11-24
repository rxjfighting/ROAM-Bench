"""Resolve image-point navigation against the completed decision observation."""
from copy import deepcopy

def state_token(backend):
    return (id(getattr(backend, '_episode', None)), str(getattr(backend, '_loaded_scene_root', None)), deepcopy(getattr(backend, '_pose', None)), getattr(backend, '_step_idx', None), deepcopy(getattr(backend, '_held', None)), deepcopy(getattr(backend, '_open_state', None)), deepcopy(getattr(backend, '_toggled_state', None)), deepcopy(getattr(backend, '_position_overrides', None)))

def install(backend_type, referer_type, resolve_navpoint, no_segmentation_hit):
    if getattr(backend_type, '_ground_sensor_integrity_installed_v1', False):
        raise RuntimeError('Ground sensor contract already installed')
    original_ground = backend_type.go_to_ground_at_pixel
    original_observe = backend_type.observe
    original_reset = backend_type.reset
    original_navpoint = referer_type.navpoint_at
    field = '_ground_sensor_transaction_v1'
    decision_field = '_ground_decision_sensor_v1'

    def observe(backend, *args, **kwargs):
        setattr(backend, decision_field, None)
        result = original_observe(backend, *args, **kwargs)
        setattr(backend, decision_field, dict(token=state_token(backend), packet=deepcopy(backend._seg_and_aliases())))
        return result

    def reset(backend, *args, **kwargs):
        setattr(backend, decision_field, None)
        return original_reset(backend, *args, **kwargs)

    def ground(backend, u, v):
        if getattr(backend, field, None) is not None:
            raise RuntimeError('Nested ground sensor transaction')
        decision = getattr(backend, decision_field, None)
        if decision is None:
            raise RuntimeError('Ground click requires a completed observation')
        if decision['token'] != state_token(backend):
            raise RuntimeError('Ground click observation is stale; observe again')
        snapshot = deepcopy(decision['packet'])
        setattr(backend, decision_field, None)
        transaction = dict(pixel=(int(u), int(v)), packet=snapshot, consumed=False)
        setattr(backend, field, transaction)
        try:
            result = original_ground(backend, u, v)
            return result
        finally:
            delattr(backend, field)

    def navpoint(referer, u, v):
        transaction = getattr(referer._b, field, None)
        if transaction is None:
            return original_navpoint(referer, u, v)
        if transaction['consumed'] or transaction['pixel'] != (int(u), int(v)):
            raise RuntimeError('Ground click packet used by a different resolution')
        transaction['consumed'] = True
        packet = transaction['packet']
        transaction['packet'] = None
        if packet is None:
            return no_segmentation_hit()
        return resolve_navpoint(*packet, int(u), int(v), world_from_pixel=referer._raycast_hit, floor_z=referer._floor_z)
    backend_type.go_to_ground_at_pixel = ground
    backend_type.observe = observe
    backend_type.reset = reset
    referer_type.navpoint_at = navpoint
    backend_type._ground_sensor_integrity_installed_v1 = True

def install_accepted_harness():
    from roam_bench.backend import ROAMBackend
    from roam_bench.pixel_refer import PixelReferer, PixelHit, R_NO_SEG, resolve_pixel_to_navpoint

    def no_segmentation():
        return PixelHit('unresolved', reason='No segmentation frame yet (render first).', reason_code=R_NO_SEG)
    install(ROAMBackend, PixelReferer, resolve_pixel_to_navpoint, no_segmentation)
