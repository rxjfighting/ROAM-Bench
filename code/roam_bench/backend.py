"""Construct Isaac Sim scenes and execute ROAM-Bench navigation and interaction skills."""
from __future__ import annotations
import math
import os
import time
from dataclasses import replace
from typing import Optional
import numpy as np
from roam_bench.protocol.types import ActionResult, Episode, NameRef, Obs, PixelRef, Pose
from roam_bench.protocol import visibility as visibility_policy
from . import seg_alias
from . import visible_gate
from .embodiment import DEFAULT_EMBODIMENT
VISIBILITY_DISTANCE = 1.5
VISIBLE_GATE_MIN_PIXEL_FRAC = None
VISIBLE_GATE_MIN_PIXELS = visibility_policy.MIN_VISIBLE_PIXELS
ARRIVE_TOL = 0.25
_STRUCTURAL = {'wall'}
_DEFAULT_COLLISION_MOVE = True
GROUND_WALK_STABILIZED_HORIZON_DEG = 10.0

def _capsule_sample_heights(radius: float, height: float) -> tuple[float, float, float]:
    radius = float(radius)
    height = float(height)
    if radius <= 0.0 or height < 2.0 * radius:
        raise ValueError('capsule height must be at least twice its positive radius')
    eps = min(0.01, max(0.0, (height - 2.0 * radius) / 4.0))
    return (radius + eps, height / 2.0, height - radius - eps)

def _assert_render_deps_resolved() -> None:
    from . import imgcheck

class _RenderSkipped:
    __slots__ = ()
    _MSG = 'Obs.rgb is RENDER_SKIPPED: this frame was never rendered because ROAMBackend.set_render_skip(True) is on (a verdict-only fast path that discards the frame). No model/agent may consume this image. If you are on the model-eval path, render-skip has leaked here -- do not skip it, render the frame (call set_render_skip(False)).'

    def __array__(self, dtype=None):
        raise RuntimeError(self._MSG)

    def __getitem__(self, key):
        raise RuntimeError(self._MSG)

    def __iter__(self):
        raise RuntimeError(self._MSG)

    def __len__(self):
        raise RuntimeError(self._MSG)

    def __bool__(self):
        raise RuntimeError(self._MSG)

    def __getattr__(self, name):
        raise RuntimeError(self._MSG)

    def __repr__(self):
        return '<RENDER_SKIPPED>'
RENDER_SKIPPED = _RenderSkipped()
_SEG_ANNOTATOR = 'instance_id_segmentation_fast'
SEG_WARMUP = 2
ROOM_LIGHT_INTENSITY_OVERRIDES = {'wine_cellar': 38400.0}
PLACE_MODES = ('receptacle', 'point')

def _ref_alias(ref):
    try:
        from roam_bench.protocol.types import NameRef
        if isinstance(ref, NameRef):
            return ref.name
    except Exception:
        pass
    if isinstance(ref, str):
        return ref
    if getattr(ref, 'kind', None) == 'object':
        return getattr(ref, 'alias', None) or None
    return getattr(ref, 'name', None)

class ROAMBackend:

    def __init__(self, headless: bool=True, renderer: str='RayTracedLighting', width: int=1280, height: int=720):
        os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
        os.environ.setdefault('DISPLAY', ':99')
        from . import scratch_env
        scratch_env.redirect()
        from isaacsim import SimulationApp
        self._width = width
        self._height = height
        self._renderer = renderer
        self._app = SimulationApp({'headless': headless, 'renderer': renderer, 'width': width, 'height': height})
        import omni.usd
        import omni.replicator.core as rep
        from pxr import UsdGeom, Usd
        self._omni_usd = omni.usd
        self._rep = rep
        self._UsdGeom = UsdGeom
        self._Usd = Usd
        self._stage = omni.usd.get_context().get_stage()
        self._render_product = None
        self._render_antialiasing = None
        self._rgb_annot = None
        self._depth_annot = None
        self._seg_annot = None
        self._cam_path = '/World/EvalCamera'
        self._gltf_usd_cache = {}
        self._pose: Optional[Pose] = None
        self._episode: Optional[Episode] = None
        self._scene = None
        self._loaded_scene_root: Optional[str] = None
        self._facts = None
        self._nav = None
        self._step_idx = 0
        self._seen: set = set()
        self._held: Optional[str] = None
        self._held_visual_aabb = None
        self._placed_on: dict[str, str] = {}
        self._placed_points: dict[str, tuple[float, float, float]] = {}
        self._placement_events: list[dict] = []
        self._room_overrides: dict[str, str] = {}
        self._open_state: dict[str, bool] = {}
        self._toggled_state: dict[str, bool] = {}
        self._position_overrides: dict[str, tuple[float, float, float]] = {}
        self._world_model = None
        self._alias_prims: dict[str, list[str]] = {}
        self._alias_prim_matrices: dict[str, list[tuple[str, np.ndarray]]] = {}
        self._dark_check = True
        self._render_warmup = 60
        self._render_subframes = 16
        self._film_subframes = 16
        self._film_rate_multiplier = 1
        self._skip_render = False
        self._seg_warmup = SEG_WARMUP
        self._frame_checks: list = []
        self._evidence_observations: list[dict] = []
        self._evidence_opened: set[str] = set()
        self._evidence_mutations: list[dict] = []
        self._evidence_approaches: list[dict] = []
        self._evidence_interactions: list[dict] = []
        self._collision_move = _DEFAULT_COLLISION_MOVE
        self._collision_furniture = True
        self._physics_ready = False
        self._structural_physics_ready = False
        self._planner = None
        self._oracle_planner = None
        self._visible_gate = True
        self._interact_enabled = False
        self._articulations: dict = {}
        self._artic_specs: dict = {}
        self._pivot_reports: dict = {}
        self._donor_registry = None
        self._sim_ctx = None
        self._interact_settle = 2000
        self._aim_pixel = None
        self._aim_world = None
        self._pixel_referer = None
        self._embodiment = DEFAULT_EMBODIMENT
        self._humanoid_collision_boxes = None
        self._place_mode = 'receptacle'
        self._fixture_light_position = 'bottom'
        self._robustness_profile = None
        self.on_frame = None
        self.on_label = None
        self.on_aim = None
        self._film_outer_action = False
        self._setup_stage_defaults()
        from .runtime_integrity import install_candidate
        install_candidate(self)

    def _setup_stage_defaults(self):
        from pxr import UsdGeom, Gf
        stage = self._stage
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        world = UsdGeom.Xform.Define(stage, '/World')
        stage.SetDefaultPrim(world.GetPrim())

    def _clear_loaded_scene_stage(self) -> None:
        for path in ('/World/objs', '/World/ExteriorSeals', '/World/Ceilings', '/World/RoomLights', '/World/RoomFillLights', '/World/FixtureGlow', '/World/Looks/FixtureGlow', '/World/phys'):
            prim = self._stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                self._stage.RemovePrim(path)
        self._physics_ready = False
        self._structural_physics_ready = False
        self._room_light_positions = []
        self._humanoid_collision_boxes = None
        for _ in range(2):
            self._app.update()

    def add_lighting(self, dome_intensity: float=400.0, sun_intensity: float=1500.0):
        from pxr import UsdGeom, UsdLux, Gf
        stage = self._stage
        dome = UsdLux.DomeLight.Define(stage, '/World/DomeLight')
        dome.GetIntensityAttr().Set(dome_intensity)
        sun = UsdLux.DistantLight.Define(stage, '/World/Sun')
        sun.GetIntensityAttr().Set(sun_intensity)
        sun.GetAngleAttr().Set(1.0)
        UsdGeom.XformCommonAPI(sun).SetRotate(Gf.Vec3f(-50.0, 10.0, 0.0))

    def set_robustness(self, profile=None) -> None:
        if getattr(self, '_loaded_scene_root', None) is not None:
            raise RuntimeError('set_robustness must be called before the first reset')
        if profile is None:
            self._robustness_profile = None
            return
        from .robustness import validate_profile
        validate_profile(profile)
        self._robustness_profile = dict(profile)

    def _robustness_params(self) -> dict:
        profile = getattr(self, '_robustness_profile', None) or {}
        return dict(profile.get('params') or {})

    def _apply_robustness_scene(self, scene) -> None:
        profile = getattr(self, '_robustness_profile', None)
        if not profile or profile.get('level') == 'identity':
            return
        axis, params = (profile['axis'], profile['params'])
        seed = int(profile['seed'])
        if axis == 'layout' and float(params.get('layout_jitter_m', 0.0)) > 0:
            from .robustness import deterministic_gaussian
            protected = set(profile.get('protected_aliases') or ())
            facts = getattr(self, '_facts', None)
            for alias, entries in sorted(self._alias_prim_matrices.items()):
                fact = facts.by_alias.get(alias) if facts is not None else None
                if alias in protected or not getattr(fact, 'pickupable', False):
                    continue
                limit = float(params['layout_jitter_m'])
                dx = max(-limit, min(limit, deterministic_gaussian(seed, 0, alias + ':x', limit / 2)))
                dy = max(-limit, min(limit, deterministic_gaussian(seed, 0, alias + ':y', limit / 2)))
                for path, baseline in entries:
                    shifted = np.array(baseline, dtype=float, copy=True)
                    shifted[0, 3] += dx
                    shifted[1, 3] += dy
                    xform = self._UsdGeom.Xformable(self._stage.GetPrimAtPath(path))
                    ops = xform.GetOrderedXformOps()
                    if ops:
                        ops[0].Set(self._np_to_gf_matrix(shifted))
        if axis == 'material' and float(params.get('material_strength', 0.0)) > 0:
            import hashlib
            from pxr import Gf, Sdf, UsdShade
            strength = float(params['material_strength'])
            root = '/World/RobustnessMaterials'
            self._UsdGeom.Xform.Define(self._stage, root)
            facts = getattr(self, '_facts', None)
            scene_kinds = {inst.name: inst.kind for inst in getattr(scene, 'instances', ())}
            for alias, paths in sorted(self._alias_prims.items()):
                fact = facts.by_alias.get(alias) if facts is not None else None
                if scene_kinds.get(alias) == 'geometry' or getattr(fact, 'kind', '') == 'geometry':
                    continue
                raw = hashlib.sha256(f'{seed}:{alias}'.encode()).digest()
                if raw[3] / 255.0 >= strength:
                    continue
                tint = tuple((0.25 + 0.5 * raw[i] / 255.0 for i in range(3)))
                mat = UsdShade.Material.Define(self._stage, f'{root}/{_safe(alias)}')
                shader = UsdShade.Shader.Define(self._stage, f'{root}/{_safe(alias)}/Shader')
                shader.CreateIdAttr('UsdPreviewSurface')
                shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*tint))
                shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.55)
                mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
                for path in paths:
                    prim = self._stage.GetPrimAtPath(path)
                    if prim and prim.IsValid():
                        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    def add_room_lighting(self, scene, intensity: float=24000.0, radius: float=0.4, per_room_intensity=None, fixture_position: str | None=None, fixture_shape: str='disk', ambient_fill_intensity: float=3600.0):
        from collections import defaultdict
        from pxr import Usd, UsdLux, UsdGeom, UsdShade, Sdf, Gf
        from .ceiling_lighting import fixture_light_position, fixture_light_radius, room_fill_grid, room_fill_intensity, room_main_light_budget
        stage = self._stage
        mode = fixture_position or self._fixture_light_position
        if mode not in ('center', 'bottom'):
            raise ValueError("fixture_position must be 'center' or 'bottom'")
        if fixture_shape not in ('sphere', 'disk'):
            raise ValueError("fixture_shape must be 'sphere' or 'disk'")
        overrides = ROOM_LIGHT_INTENSITY_OVERRIDES if per_room_intensity is None else per_room_intensity
        groups = defaultdict(list)
        bounds_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        for anchor in tuple(getattr(scene, 'light_anchors', ())):
            lo = hi = None
            ranges = []
            for path in self._alias_prims.get(anchor.alias, ()):
                prim = stage.GetPrimAtPath(path)
                if prim and prim.IsValid():
                    ranges.append(bounds_cache.ComputeWorldBound(prim).ComputeAlignedRange())
            if ranges:
                lo = tuple((min((float(r.GetMin()[axis]) for r in ranges)) for axis in range(3)))
                hi = tuple((max((float(r.GetMax()[axis]) for r in ranges)) for axis in range(3)))
            emitter_radius = fixture_light_radius(lo, hi, fixture_shape)
            point = fixture_light_position(anchor.position, lo, hi, mode)
            if mode == 'bottom' and fixture_shape == 'sphere' and (lo is not None):
                point = (point[0], point[1], point[2] - emitter_radius)
            groups[anchor.room_id].append((anchor.alias, point, bool(anchor.fallback), emitter_radius))
        authored = bool(groups)
        if not groups:
            for inst in scene.instances:
                if inst.kind == 'geometry':
                    groups[inst.room].append(('room_center', inst.transform[:3, 3], True, radius))
        n = 0
        self._room_light_positions = []
        glow_material = None
        if authored and fixture_shape == 'disk':
            UsdGeom.Xform.Define(stage, '/World/FixtureGlow')
            glow_material = UsdShade.Material.Define(stage, '/World/Looks/FixtureGlow')
            shader = UsdShade.Shader.Define(stage, '/World/Looks/FixtureGlow/PreviewSurface')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.92, 0.72))
            glow_scale = max(0.0, float(intensity) / 24000.0)
            shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(4.0 * glow_scale, 3.6 * glow_scale, 2.8 * glow_scale))
            shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.3)
            glow_material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        slabs_by_room = {slab.room_id: slab for slab in tuple(getattr(scene, 'ceiling_slabs', ()))}
        for room, entries in groups.items():
            if not authored:
                pts = np.asarray([entry[1] for entry in entries])
                c = pts.mean(0)
                entries = [('room_center', (float(c[0]), float(c[1]), float(max(c[2], getattr(scene, 'wall_height', 3.4)) - 0.25)), True, radius)]
            slab = slabs_by_room.get(room)
            if room in overrides or slab is None:
                room_budget = float(overrides.get(room, intensity))
            else:
                room_budget = room_main_light_budget(slab.size[0], slab.size[1], intensity)
            each = room_budget / max(1, len(entries))
            for index, (alias, point, fallback, emitter_radius) in enumerate(entries):
                shape = 'sphere' if fallback else fixture_shape
                light_cls = UsdLux.SphereLight if shape == 'sphere' else UsdLux.DiskLight
                light = light_cls.Define(stage, f'/World/RoomLights/{_safe(room)}_{index}')
                light.GetIntensityAttr().Set(each)
                light.GetRadiusAttr().Set(emitter_radius)
                light.GetNormalizeAttr().Set(True)
                light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.98, 0.95))
                UsdGeom.XformCommonAPI(light).SetTranslate(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
                if shape == 'disk' and (not fallback) and (glow_material is not None):
                    glow = UsdGeom.Cylinder.Define(stage, f'/World/FixtureGlow/room_geometry_{_safe(room)}_{index}')
                    glow.CreateAxisAttr(UsdGeom.Tokens.z)
                    glow.CreateRadiusAttr().Set(float(emitter_radius) * 0.85)
                    glow.CreateHeightAttr().Set(0.004)
                    UsdGeom.XformCommonAPI(glow).SetTranslate(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]) + 0.012))
                    UsdShade.MaterialBindingAPI.Apply(glow.GetPrim()).Bind(glow_material)
                self._room_light_positions.append({'room': room, 'alias': alias, 'position': tuple(map(float, point)), 'mode': mode if not fallback else 'fallback', 'shape': shape, 'radius': emitter_radius, 'intensity': each, 'normalized': True})
                n += 1
        if ambient_fill_intensity > 0:
            for slab in tuple(getattr(scene, 'ceiling_slabs', ())):
                width, depth = (float(slab.size[0]), float(slab.size[1]))
                offsets = room_fill_grid(width, depth)
                cell_span = min(width / max(1, math.ceil(width / 4.2)), depth / max(1, math.ceil(depth / 4.0)))
                fill_radius = min(0.8, max(0.35, cell_span * 0.15))
                if width * depth <= 5.0:
                    fill_intensity = room_fill_intensity(width, depth, ambient_fill_intensity)
                else:
                    fill_intensity = float(ambient_fill_intensity)
                ceiling_z = float(slab.center[2] - slab.size[2] / 2.0)
                fill_z = max(1.5, ceiling_z * 0.68)
                for fill_index, (offset_x, offset_y) in enumerate(offsets):
                    fill = UsdLux.SphereLight.Define(stage, f'/World/RoomFillLights/{_safe(slab.room_id)}_{fill_index}')
                    fill.GetIntensityAttr().Set(fill_intensity)
                    fill.GetNormalizeAttr().Set(True)
                    fill.GetRadiusAttr().Set(fill_radius)
                    fill.GetColorAttr().Set(Gf.Vec3f(1.0, 0.98, 0.95))
                    position = (float(slab.center[0]) + offset_x, float(slab.center[1]) + offset_y, fill_z)
                    UsdGeom.XformCommonAPI(fill).SetTranslate(Gf.Vec3d(*position))
                    self._room_light_positions.append({'room': slab.room_id, 'alias': f'ambient_fill_{fill_index}', 'position': position, 'mode': 'room_fill', 'shape': 'sphere', 'radius': fill_radius, 'intensity': fill_intensity, 'reference_intensity': float(ambient_fill_intensity), 'normalized': True})
        print(f'[backend] add_room_lighting: {n} fixture-anchored ceiling lights' + (f' ({len(overrides)} per-room override)' if overrides else ''))
        return n

    def set_fixture_light_position(self, mode: str='bottom') -> None:
        if mode not in ('center', 'bottom'):
            raise ValueError("fixture light position must be 'center' or 'bottom'")
        self._fixture_light_position = mode

    def add_ceilings(self, scene) -> int:
        slabs = tuple(getattr(scene, 'ceiling_slabs', ()))
        if not slabs:
            return 0
        from pxr import Gf, Sdf, UsdGeom
        from .exterior_seals import seal_box_mesh
        root = '/World/Ceilings'
        UsdGeom.Xform.Define(self._stage, root)
        for slab in slabs:
            mesh = UsdGeom.Mesh.Define(self._stage, f'{root}/room_geometry_{_safe(slab.room_id)}_ceiling')
            points, counts, indices, uv = seal_box_mesh(slab.size)
            mesh.CreatePointsAttr(points)
            mesh.CreateFaceVertexCountsAttr(counts)
            mesh.CreateFaceVertexIndicesAttr(indices)
            mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set([Gf.Vec2f(u, v) for u, v in uv])
            UsdGeom.XformCommonAPI(mesh).SetTranslate(Gf.Vec3d(*slab.center))
            if not self._bind_room_wall_material(mesh.GetPrim(), slab.room_id):
                mesh.CreateDisplayColorAttr([Gf.Vec3f(0.82, 0.82, 0.8)])
        print(f'[backend] add_ceilings: {len(slabs)} render-only room slabs', flush=True)
        return len(slabs)

    def _gltf_to_usd(self, gltf_path: str) -> Optional[str]:
        if gltf_path in self._gltf_usd_cache:
            return self._gltf_usd_cache[gltf_path]
        usd_path = os.path.splitext(gltf_path)[0] + '.isaac.usd'
        if os.path.exists(usd_path):
            self._gltf_usd_cache[gltf_path] = usd_path
            return usd_path
        try:
            import omni.kit.asset_converter as ac
            import asyncio
            task_mgr = ac.get_instance()
            ctx = ac.AssetConverterContext()
            task = task_mgr.create_converter_task(gltf_path, usd_path, None, ctx)
            ok = asyncio.get_event_loop().run_until_complete(task.wait_until_finished())
            if not ok:
                print(f'[backend] asset_converter failed for {gltf_path}: {task.get_error_message()}')
                self._gltf_usd_cache[gltf_path] = None
                return None
        except Exception as e:
            print(f'[backend] asset_converter exception for {gltf_path}: {e}')
            self._gltf_usd_cache[gltf_path] = None
            return None
        self._gltf_usd_cache[gltf_path] = usd_path
        return usd_path

    def _np_to_gf_matrix(self, m: np.ndarray):
        from pxr import Gf
        mt = m.T
        return Gf.Matrix4d(*[float(x) for x in mt.flatten()])

    def _should_author_articulation(self, model_name: str) -> bool:
        return bool(self._interact_enabled and model_name in self._artic_specs)

    def _pivot_fixed_plan(self, plan, model_name: str, *, apply_pivot: bool):
        if not apply_pivot:
            return plan
        from .hinge_geometry import align_pivots
        plan_out, report = align_pivots(plan, self._scene.root, enable=True, scene_model_key=model_name, donor_registry=getattr(self, '_donor_registry', None))
        self._pivot_reports[model_name] = report
        return plan_out

    def add_scene(self, scene, kinds=('geometry', 'furniture'), use_asset_converter: bool=False, limit: Optional[int]=None):
        from pxr import UsdGeom
        stage = self._stage
        added, failed = (0, 0)
        insts = [i for i in scene.instances if i.kind in kinds]
        if limit:
            insts = insts[:limit]
        authored_artic = set()
        for idx, inst in enumerate(insts):
            prim_path = f'/World/objs/{inst.room}/{_safe(inst.name)}_{idx}'
            if self._should_author_articulation(inst.name):
                if inst.name in authored_artic:
                    continue
                try:
                    self._author_articulation(inst, prim_path, *self._artic_specs[inst.name], apply_pivot=True)
                    self._remember_alias_prim(inst.name, prim_path, inst.model_world)
                    authored_artic.add(inst.name)
                    added += 1
                except Exception as e:
                    print(f'Invalid scene geometry. {inst.name}: {e!r}', flush=True)
                    failed += 1
                continue
            xform = UsdGeom.Xform.Define(stage, prim_path)
            ref_target = inst.gltf_path
            if use_asset_converter:
                usd = self._gltf_to_usd(inst.gltf_path)
                if usd is None:
                    failed += 1
                    continue
                ref_target = usd
            ok = xform.GetPrim().GetReferences().AddReference(ref_target)
            xform.MakeMatrixXform().Set(self._np_to_gf_matrix(inst.transform))
            self._remember_alias_prim(inst.name, prim_path, inst.transform)
            added += 1
        print(f'[backend] add_scene: added={added} failed={failed} (of {len(insts)} in kinds={kinds})')
        return (added, failed)

    def add_exterior_seals(self, scene) -> int:
        seals = tuple(getattr(scene, 'exterior_seals', ()))
        if not seals:
            return 0
        from pxr import Gf, Sdf, UsdGeom
        root = '/World/ExteriorSeals'
        UsdGeom.Xform.Define(self._stage, root)
        for index, seal in enumerate(seals):
            from .exterior_seals import seal_box_mesh
            mesh = UsdGeom.Mesh.Define(self._stage, f'{root}/{_safe(seal.opening_id)}_{index}')
            points, counts, indices, uv = seal_box_mesh(seal.size)
            mesh.CreatePointsAttr(points)
            mesh.CreateFaceVertexCountsAttr(counts)
            mesh.CreateFaceVertexIndicesAttr(indices)
            mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set([Gf.Vec2f(u, v) for u, v in uv])
            xf = UsdGeom.Xformable(mesh.GetPrim())
            xf.AddTransformOp().Set(self._np_to_gf_matrix(seal.transform))
            if not self._bind_room_wall_material(mesh.GetPrim(), seal.room_id):
                mesh.CreateDisplayColorAttr([Gf.Vec3f(0.72, 0.72, 0.7)])
        return len(seals)

    def _bind_room_wall_material(self, target_prim, room_id: str) -> bool:
        try:
            from pxr import Usd, UsdGeom, UsdShade
            room_root = self._stage.GetPrimAtPath(f'/World/objs/{_safe(room_id)}')
            if not room_root or not room_root.IsValid():
                return False
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            for prim in Usd.PrimRange(room_root):
                if 'room_geometry' not in str(prim.GetPath()):
                    continue
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                size = cache.ComputeWorldBound(prim).ComputeAlignedRange().GetSize()
                if float(size[2]) < 1.0:
                    continue
                material, _relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                if material and material.GetPrim().IsValid():
                    UsdShade.MaterialBindingAPI.Apply(target_prim).Bind(material)
                    return True
        except Exception:
            return False
        return False

    def _remember_alias_prim(self, alias: str, prim_path: str, matrix) -> None:
        if not hasattr(self, '_alias_prims'):
            self._alias_prims = {}
        if not hasattr(self, '_alias_prim_matrices'):
            self._alias_prim_matrices = {}
        self._alias_prims.setdefault(alias, []).append(prim_path)
        if matrix is not None:
            self._alias_prim_matrices.setdefault(alias, []).append((prim_path, np.asarray(matrix, dtype=float).copy()))

    def _set_alias_visible(self, alias: str, visible: bool) -> None:
        try:
            from pxr import UsdGeom
            for path in self._alias_prims.get(alias, ()):
                imageable = UsdGeom.Imageable(self._stage.GetPrimAtPath(path))
                if visible:
                    imageable.MakeVisible()
                else:
                    imageable.MakeInvisible()
        except Exception:
            pass

    def _place_alias_at(self, alias: str, destination: str) -> None:
        facts = self._facts
        centroids = getattr(facts, 'centroids', {}) if facts else {}
        source = centroids.get(alias)
        target = centroids.get(destination)
        if source is None or target is None:
            self._set_alias_visible(alias, True)
            return
        delta = np.eye(4)
        delta[:3, 3] = (target[0] - source[0], target[2] - source[2], target[1] + 0.5 - source[1])
        entries = tuple(self._alias_prim_matrices.get(alias, ()))
        try:
            if entries:
                from pxr import UsdGeom
            for path, baseline in entries:
                xform = UsdGeom.Xformable(self._stage.GetPrimAtPath(path))
                ops = xform.GetOrderedXformOps()
                if ops:
                    ops[0].Set(self._np_to_gf_matrix(delta @ baseline))
        except Exception:
            pass
        self._set_alias_visible(alias, True)

    def _placement_point_from_pixel(self, u: int, v: int):
        world = self._film_world_from_depth(int(u), int(v))
        return None if world is None else tuple((float(q) for q in world))

    def _placement_normal_from_pixel(self, u: int, v: int, point, alias: str):
        p = np.asarray(point, dtype=float)
        eye = None
        try:
            eye, _target = self._pose_to_camera(self._pose)
            eye = np.asarray(eye, dtype=float)
        except Exception:
            pass
        for radius in (1, 2, 3, 4, 6, 8):
            pixels = ((int(u) - radius, int(v)), (int(u) + radius, int(v)), (int(u), int(v) - radius), (int(u), int(v) + radius))
            try:
                if any((self.instance_at_pixel(su, sv) != alias for su, sv in pixels)):
                    continue
            except Exception:
                continue
            left = self._film_world_from_depth(*pixels[0])
            right = self._film_world_from_depth(*pixels[1])
            top = self._film_world_from_depth(*pixels[2])
            bottom = self._film_world_from_depth(*pixels[3])
            if any((q is None for q in (left, right, top, bottom))):
                continue
            samples = [np.asarray(q, dtype=float) for q in (left, right, top, bottom)]
            if max((float(np.linalg.norm(q - p)) for q in samples)) > 0.25:
                continue
            du = samples[1] - samples[0]
            dv = samples[3] - samples[2]
            normal = np.cross(du, dv)
            length = float(np.linalg.norm(normal))
            if length <= 1e-07:
                continue
            normal /= length
            if eye is not None and float(np.dot(normal, eye - p)) < 0.0:
                normal = -normal
            return tuple((float(q) for q in normal))
        facts = self._facts
        target = (getattr(facts, 'centroids', {}) or {}).get(alias) if facts else None
        if target is not None:
            target_stage = np.asarray((target[0], target[2], target[1]), dtype=float)
            normal = p - target_stage
            length = float(np.linalg.norm(normal))
            if length > 1e-07:
                normal /= length
                return tuple((float(q) for q in normal))
        return (0.0, 0.0, 1.0)

    @staticmethod
    def _translated_aabb(lo, hi, source_center, target_center):
        delta = np.asarray(target_center, dtype=float) - np.asarray(source_center, dtype=float)
        return (np.asarray(lo, dtype=float) + delta, np.asarray(hi, dtype=float) + delta)

    @staticmethod
    def _aabb_overlap(a_lo, a_hi, b_lo, b_hi, epsilon: float=1e-05) -> bool:
        return bool(np.all(np.minimum(a_hi, b_hi) - np.maximum(a_lo, b_lo) > epsilon))

    def _alias_visual_aabb(self, alias: str):
        paths = tuple(getattr(self, '_alias_prims', {}).get(alias, ()))
        if not paths or not hasattr(self, '_stage'):
            return None
        try:
            from pxr import Usd, UsdGeom
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            lows, highs = ([], [])
            for path in paths:
                prim = self._stage.GetPrimAtPath(path)
                if not prim or not prim.IsValid():
                    continue
                aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                lo = np.asarray(aligned.GetMin(), dtype=float)
                hi = np.asarray(aligned.GetMax(), dtype=float)
                if lo.shape == (3,) and hi.shape == (3,) and np.isfinite(lo).all() and np.isfinite(hi).all() and np.all(hi > lo):
                    lows.append(lo)
                    highs.append(hi)
            if not lows:
                return None
            return (np.min(np.stack(lows), axis=0), np.max(np.stack(highs), axis=0))
        except Exception:
            return None

    def _alias_visual_cells(self, alias: str):
        paths = tuple(getattr(self, '_alias_prims', {}).get(alias, ()))
        if not paths or not hasattr(self, '_stage'):
            return ()
        try:
            from pxr import Gf, Usd, UsdGeom
            xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
            cells = []
            pad = 0.001
            for root_path in paths:
                root = self._stage.GetPrimAtPath(root_path)
                if not root or not root.IsValid():
                    continue
                for prim in Usd.PrimRange(root):
                    if not prim.IsA(UsdGeom.Mesh):
                        continue
                    imageable = UsdGeom.Imageable(prim)
                    if imageable.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
                        continue
                    mesh = UsdGeom.Mesh(prim)
                    points = mesh.GetPointsAttr().Get()
                    counts = mesh.GetFaceVertexCountsAttr().Get()
                    indices = mesh.GetFaceVertexIndicesAttr().Get()
                    if not points or not counts or (not indices):
                        continue
                    world = xforms.GetLocalToWorldTransform(prim)
                    vertices = np.asarray([tuple(world.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))) for p in points], dtype=float)
                    cursor = 0
                    for raw_count in counts:
                        count = int(raw_count)
                        face = [int(q) for q in indices[cursor:cursor + count]]
                        cursor += count
                        if count < 3:
                            continue
                        anchor = vertices[face[0]]
                        for index in range(1, count - 1):
                            tri = np.asarray((anchor, vertices[face[index]], vertices[face[index + 1]]), dtype=float)
                            if not np.isfinite(tri).all():
                                continue
                            cross = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                            length = float(np.linalg.norm(cross))
                            upward = bool(length > 1e-09 and abs(float(cross[2])) / length >= 0.5)
                            cells.append((tri.min(axis=0) - pad, tri.max(axis=0) + pad, upward))
            return tuple(cells)
        except Exception:
            return ()

    def _placement_cells(self, alias: str):
        cells = self._alias_visual_cells(alias)
        if cells:
            return (cells, 'usd_mesh_faces')
        packed = self._placement_aabb(alias)
        if packed is None:
            return ((), None)
        lo, hi, source = packed
        return (((np.asarray(lo, float), np.asarray(hi, float), True),), source)

    def _placement_aabb(self, alias: str):
        visual = self._alias_visual_aabb(alias)
        if visual is not None:
            return (visual[0], visual[1], 'usd_visual')
        facts = getattr(self, '_facts', None)
        box = (getattr(facts, 'aabbs', {}) or {}).get(alias) if facts else None
        if box is None:
            return None
        return (np.asarray(box[0], float), np.asarray(box[1], float), 'scene_facts')

    def _set_alias_stage_center(self, alias: str, center, source_stage) -> bool:
        delta = np.eye(4)
        delta[:3, 3] = np.asarray(center, dtype=float) - np.asarray(source_stage, dtype=float)
        entries = tuple(self._alias_prim_matrices.get(alias, ()))
        try:
            if entries:
                from pxr import UsdGeom
            for path, baseline in entries:
                xform = UsdGeom.Xformable(self._stage.GetPrimAtPath(path))
                ops = xform.GetOrderedXformOps()
                if ops:
                    ops[0].Set(self._np_to_gf_matrix(delta @ baseline))
        except Exception:
            return False
        self._set_alias_visible(alias, True)
        return True

    def _point_release_outcome(self, alias: str, destination: str, point, normal):
        facts = self._facts
        held_box = getattr(self, '_held_visual_aabb', None)
        if held_box is not None and held_box[0] == alias:
            lo0 = np.asarray(held_box[1], dtype=float)
            hi0 = np.asarray(held_box[2], dtype=float)
            source_kind = 'usd_visual'
        else:
            packed = self._placement_aabb(alias)
            if packed is None:
                return None
            lo0, hi0, source_kind = packed
        if not np.all(hi0 > lo0):
            return None
        source_stage = (lo0 + hi0) * 0.5
        n = np.asarray(normal, dtype=float)
        nlen = float(np.linalg.norm(n))
        if nlen <= 1e-07 or not np.isfinite(n).all():
            return None
        n /= nlen
        p = np.asarray(point, dtype=float)
        offsets = np.asarray([(x, y, z) for x in (lo0[0] - source_stage[0], hi0[0] - source_stage[0]) for y in (lo0[1] - source_stage[1], hi0[1] - source_stage[1]) for z in (lo0[2] - source_stage[2], hi0[2] - source_stage[2])], dtype=float)
        release_clearance = 0.05
        contact_clearance = 0.005
        centre = p + n * (-float(np.min(offsets @ n)) + release_clearance)
        obstacles = []
        bounds_sources = {}
        destination_inst = facts.by_alias.get(destination)
        destination_room = getattr(destination_inst, 'room_id', None)
        for other in getattr(facts, 'aabbs', {}) or {}:
            if other == alias:
                continue
            inst = facts.by_alias.get(other)
            if inst is None or getattr(inst, 'room_id', None) != destination_room:
                continue
            if str(getattr(inst, 'kind', '')).lower() == 'geometry':
                continue
            packed = self._placement_aabb(other)
            if packed is None:
                continue
            blo, bhi, kind = packed
            provisional_lo, provisional_hi = self._translated_aabb(lo0, hi0, source_stage, centre)
            if min(float(provisional_hi[0]), float(bhi[0])) - max(float(provisional_lo[0]), float(blo[0])) <= -0.1 or min(float(provisional_hi[1]), float(bhi[1])) - max(float(provisional_lo[1]), float(blo[1])) <= -0.1:
                continue
            cells, cell_source = self._placement_cells(other)
            for cell_lo, cell_hi, upward in cells:
                obstacles.append((other, cell_lo, cell_hi, upward))
            bounds_sources[other] = cell_source or kind
        clearance_shift = 0.0
        for _ in range(12):
            lo, hi = self._translated_aabb(lo0, hi0, source_stage, centre)
            overlaps = [(other, blo, bhi) for other, blo, bhi, _upward in obstacles if self._aabb_overlap(lo, hi, blo, bhi)]
            if not overlaps:
                break
            candidates = []
            for _other, blo, bhi in overlaps:
                for axis in range(3):
                    if n[axis] > 1e-05:
                        candidates.append(float(bhi[axis] + release_clearance - lo[axis]) / float(n[axis]))
                    elif n[axis] < -1e-05:
                        candidates.append(float(blo[axis] - release_clearance - hi[axis]) / float(n[axis]))
            positive = [value for value in candidates if value > 0.0 and math.isfinite(value)]
            if not positive:
                return None
            step = min(positive) + release_clearance
            centre = centre + n * step
            clearance_shift += step
        else:
            return None
        lo_after_clearance, _hi_after_clearance = self._translated_aabb(lo0, hi0, source_stage, centre)
        floor_clearance_shift = max(0.0, release_clearance - float(lo_after_clearance[2]))
        centre[2] += floor_clearance_shift
        release_center = centre.copy()
        release_lo, release_hi = self._translated_aabb(lo0, hi0, source_stage, release_center)
        support_alias = None
        support_z = 0.0
        for other, blo, bhi, upward in obstacles:
            if not upward:
                continue
            overlap_x = min(release_hi[0], bhi[0]) - max(release_lo[0], blo[0])
            overlap_y = min(release_hi[1], bhi[1]) - max(release_lo[1], blo[1])
            top_z = float(bhi[2])
            if overlap_x > 0.005 and overlap_y > 0.005 and (top_z <= float(release_lo[2]) + 0.015) and (top_z > support_z):
                support_z, support_alias = (top_z, other)
        final_bottom = support_z + contact_clearance
        fall_distance = max(0.0, float(release_lo[2]) - final_bottom)
        final_center = release_center.copy()
        final_center[2] -= fall_distance
        if not self._set_alias_stage_center(alias, release_center, source_stage):
            return None
        self._film_frame(repeat=3, rt_subframes=4, rerender=True)
        if fall_distance > 0.0001:
            frames = max(8, min(36, int(math.ceil(math.sqrt(2.0 * fall_distance / 9.81) * 24.0))))
            if getattr(self, 'on_frame', None) is not None:
                frames *= getattr(self, '_film_rate_multiplier', 1)
            for index in range(1, frames + 1):
                fraction = (index / frames) ** 2
                at = release_center.copy()
                at[2] -= fall_distance * fraction
                if not self._set_alias_stage_center(alias, at, source_stage):
                    return None
                self._film_frame(rt_subframes=4, sampled=True)
        self._set_alias_stage_center(alias, final_center, source_stage)
        self._film_frame(repeat=6, rt_subframes=4, rerender=True)
        final_visual = self._alias_visual_aabb(alias)
        measured_gap = float(final_visual[0][2]) - support_z if final_visual is not None else None
        self._placed_points[alias] = tuple((float(q) for q in p))
        if not hasattr(self, '_position_overrides'):
            self._position_overrides = {}
        self._position_overrides[alias] = (float(final_center[0]), float(final_center[2]), float(final_center[1]))
        return {'surface_normal': [float(q) for q in n], 'release_xyz': [float(q) for q in release_center], 'final_xyz': [float(q) for q in final_center], 'fall_distance_m': float(fall_distance), 'support': support_alias, 'clearance_shift_m': float(clearance_shift), 'floor_clearance_shift_m': float(floor_clearance_shift), 'bounds_source': source_kind, 'support_bounds_source': bounds_sources.get(support_alias), 'release_clearance_m': float(release_clearance), 'contact_clearance_m': float(contact_clearance), 'measured_final_vertical_gap_m': measured_gap, 'solver': 'point-gravity-mesh-v3'}

    def _place_alias_at_point(self, alias: str, point) -> bool:
        target = np.asarray(point, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            return False
        facts = self._facts
        source = (getattr(facts, 'centroids', {}) or {}).get(alias) if facts else None
        aabb = (getattr(facts, 'aabbs', {}) or {}).get(alias) if facts else None
        if source is None:
            return False
        source_stage = np.asarray((source[0], source[2], source[1]), dtype=float)
        source_bottom = float(aabb[0][2]) if aabb is not None else float(source_stage[2])
        delta = np.eye(4)
        delta[:3, 3] = (target[0] - source_stage[0], target[1] - source_stage[1], target[2] + 0.005 - source_bottom)
        entries = tuple(self._alias_prim_matrices.get(alias, ()))
        try:
            if entries:
                from pxr import UsdGeom
            for path, baseline in entries:
                xform = UsdGeom.Xformable(self._stage.GetPrimAtPath(path))
                ops = xform.GetOrderedXformOps()
                if ops:
                    ops[0].Set(self._np_to_gf_matrix(delta @ baseline))
        except Exception:
            return False
        if not hasattr(self, '_placed_points'):
            self._placed_points = {}
        self._placed_points[alias] = tuple((float(q) for q in target))
        self._set_alias_visible(alias, True)
        return True

    def _restore_pick_place_state(self) -> None:
        try:
            from pxr import UsdGeom
            for alias, entries in self._alias_prim_matrices.items():
                for path, baseline in entries:
                    xform = UsdGeom.Xformable(self._stage.GetPrimAtPath(path))
                    ops = xform.GetOrderedXformOps()
                    if ops:
                        ops[0].Set(self._np_to_gf_matrix(baseline))
                self._set_alias_visible(alias, True)
        except Exception:
            pass

    def _author_articulation(self, inst, prim_path, sdf_path, specs, *, apply_pivot: bool=True):
        from pxr import UsdGeom
        from .joint_articulation import plan_articulation
        from .sdf_link_meshes import parse_link_visuals
        from . import artic_stage
        from .openable_joint_exclusions import apply_openable_joint_exclusions
        import numpy as np
        plan = plan_articulation(prim_path, specs)
        plan = self._pivot_fixed_plan(plan, inst.name, apply_pivot=apply_pivot)
        plan = apply_openable_joint_exclusions(plan, inst.name)
        visuals = parse_link_visuals(sdf_path)
        drives = artic_stage.author_articulation(self._stage, plan, visuals)
        if inst.model_world is None:
            raise ValueError('Scene validation: %r' % inst.name)
        model_world = np.asarray(inst.model_world, dtype=float)
        root_xf = UsdGeom.Xform.Get(self._stage, plan.instance_root)
        if root_xf and root_xf.GetPrim().IsValid():
            root_xf.MakeMatrixXform().Set(self._np_to_gf_matrix(model_world))
        R = model_world[:3, :3]
        R = np.stack([R[:, k] / (np.linalg.norm(R[:, k]) + 1e-12) for k in range(3)], axis=1)
        self._articulations[inst.name] = {'plan': plan, 'drives': drives, 'specs': specs, 'inst_rot': R}

    def set_camera_lookat(self, eye, target, up=(0, 0, 1), focal_length: float=18.0):
        from pxr import UsdGeom, Gf
        stage = self._stage
        cam = UsdGeom.Camera.Define(stage, self._cam_path)
        cam.GetFocalLengthAttr().Set(focal_length)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 1000.0))
        eye = Gf.Vec3d(*[float(v) for v in eye])
        target = Gf.Vec3d(*[float(v) for v in target])
        up = Gf.Vec3d(*[float(v) for v in up])
        m = Gf.Matrix4d()
        m.SetLookAt(eye, target, up)
        cam_xform = m.GetInverse()
        cam.MakeMatrixXform().Set(cam_xform)
        return cam

    def set_embodiment(self, mode: str=DEFAULT_EMBODIMENT) -> None:
        from .embodiment import MODES
        if mode not in MODES:
            raise ValueError(f'unknown embodiment {mode!r}; choose one of {MODES}')
        old_radius = self._agent_collision_radius()
        self._embodiment = mode
        self._humanoid_collision_boxes = None
        if getattr(self, '_facts', None) is not None and self._agent_collision_radius() != old_radius:
            from .navmesh import NavMesh
            self._nav = NavMesh(self._facts, agent_radius=self._agent_collision_radius())
            self._planner = None
            self._oracle_planner = None
        if getattr(self, '_pose', None) is not None:
            self._sync_embodiment_pose()

    def set_place_mode(self, mode: str='receptacle') -> None:
        if mode not in PLACE_MODES:
            raise ValueError(f'unknown place mode {mode!r}; choose one of {PLACE_MODES}')
        self._place_mode = mode
        if getattr(self, '_render_product', None) is not None:
            self._ensure_film_depth_annotator()

    def _sync_embodiment_pose(self) -> None:
        if self._pose is None or not hasattr(self, '_embodiment'):
            return
        from .embodiment import sync
        sync(self._stage, self._pose, self._embodiment)

    def _humanoid_mesh_boxes(self):
        if getattr(self, '_embodiment', DEFAULT_EMBODIMENT) != 'humanoid':
            return ()
        if not hasattr(self, '_stage'):
            return ()
        cached = getattr(self, '_humanoid_collision_boxes', None)
        if cached is not None:
            return cached
        from .embodiment import ROOT_PATH, extract_humanoid_collision_boxes
        boxes = extract_humanoid_collision_boxes(self._stage, ROOT_PATH)
        if boxes:
            self._humanoid_collision_boxes = boxes
            prim = self._stage.GetPrimAtPath(ROOT_PATH)
            if prim and prim.IsValid():
                prim.SetCustomDataByKey('roam_bench:collisionMeshBoxCount', len(boxes))
        return boxes

    def _ensure_render_product(self):
        _assert_render_deps_resolved()
        if self._render_product is not None:
            self._ensure_film_depth_annotator()
            return
        if not self._stage.GetPrimAtPath(self._cam_path).IsValid():
            raise RuntimeError('render product cannot attach: sensor path %r has no prim on the stage. The camera was never authored -- usually because set_render_skip(True) short-circuited observe() before its set_camera_lookat(), which is what Define()s the camera. Call set_camera_lookat(eye, target) to author %r before rendering (it is pure USD authoring, no sim step, so it is safe on the render-skip fast path).' % (self._cam_path, self._cam_path))
        rep = self._rep
        if getattr(self, '_renderer', 'RayTracedLighting') == 'RayTracedLighting' and getattr(self, '_render_antialiasing', None) != 'FXAA':
            rep.settings.set_render_rtx_realtime(antialiasing='FXAA')
            self._render_antialiasing = 'FXAA'
        self._render_product = rep.create.render_product(self._cam_path, (self._width, self._height))
        self._rgb_annot = rep.AnnotatorRegistry.get_annotator('rgb')
        self._rgb_annot.attach([self._render_product])
        self._seg_annot = rep.AnnotatorRegistry.get_annotator(_SEG_ANNOTATOR)
        self._seg_annot.attach([self._render_product])
        self._ensure_film_depth_annotator()

    def _ensure_film_depth_annotator(self):
        needs_depth = bool(getattr(getattr(self, '_episode', None), 'give_depth', False)) or getattr(self, 'on_frame', None) is not None or getattr(self, '_place_mode', 'receptacle') == 'point'
        if not needs_depth or getattr(self, '_depth_annot', None) is not None or self._render_product is None:
            return
        self._depth_annot = self._rep.AnnotatorRegistry.get_annotator('distance_to_camera')
        self._depth_annot.attach([self._render_product])

    @staticmethod
    def _render_hb(phase: str, i: int, n: int, rt_subframes: int, t0: float, step_dt: float=-1.0) -> None:
        dt = f' step_dt={step_dt:.2f}s' if step_dt >= 0 else ''
        print(f'[hb] render {phase} {i}/{n} rt_subframes={rt_subframes} t={time.monotonic() - t0:.1f}s{dt}', flush=True)

    def set_render_quality(self, warmup: int=60, rt_subframes: int=16) -> None:
        if warmup < 1 or rt_subframes < 1:
            raise ValueError('render warmup and subframes must both be positive')
        self._render_warmup = int(warmup)
        self._render_subframes = int(rt_subframes)

    def set_film_quality(self, rt_subframes: int=16) -> None:
        """Set the minimum rendering subframes for each video pose."""
        if rt_subframes < 1:
            raise ValueError('video rendering subframes must be positive')
        self._film_subframes = int(rt_subframes)

    def set_film_rate(self, fps: int=24) -> None:
        """Match video sampling and holds to an integer multiple of 12 FPS."""
        if fps < 12 or fps % 12:
            raise ValueError('video frame rate must be a positive multiple of 12')
        self._film_rate_multiplier = int(fps // 12)

    def render(self, warmup: int | None=None, rt_subframes: int | None=None) -> np.ndarray:
        warmup = self._render_warmup if warmup is None else int(warmup)
        rt_subframes = self._render_subframes if rt_subframes is None else int(rt_subframes)
        self._ensure_render_product()
        rep = self._rep
        _hb_t0 = time.monotonic()
        for _hb_i in range(1, warmup + 1):
            self._render_hb('warmup-enter', _hb_i, warmup, rt_subframes, _hb_t0)
            _hb_step = time.monotonic()
            rep.orchestrator.step(rt_subframes=rt_subframes)
            self._render_hb('warmup-exit', _hb_i, warmup, rt_subframes, _hb_t0, step_dt=time.monotonic() - _hb_step)
        self._render_hb('getdata-enter', warmup, warmup, rt_subframes, _hb_t0)
        data = np.asarray(self._rgb_annot.get_data())
        if data.ndim == 3 and data.shape[2] == 4:
            data = data[:, :, :3]
        data = data.astype(np.uint8)
        if getattr(self, '_dark_check', True):
            from .imgcheck import warn_if_dark, frame_stats
            try:
                room = self.agent_room_name() or '?'
            except Exception:
                room = '?'
            ctx = f'room={room}'
            ok, _mean = warn_if_dark(data, context=ctx)
            st = frame_stats(data)
            self._frame_checks.append({'context': ctx, 'mean': st['mean'], 'nonblack_frac': st['nonblack_frac'], 'ok': bool(ok)})
        return data

    def _render_seg_at_pose(self, warmup_seg: int=SEG_WARMUP, rt_subframes: int=1) -> None:
        if self._pose is None:
            raise RuntimeError('_render_seg_at_pose() before reset(): no agent pose')
        eye, target = self._pose_to_camera(self._pose)
        self._sync_embodiment_pose()
        self.set_camera_lookat(eye, target)
        self._ensure_render_product()
        for _ in range(max(1, warmup_seg)):
            self._rep.orchestrator.step(rt_subframes=rt_subframes)

    def frame_checks(self) -> list:
        return list(self._frame_checks)

    def dark_frames(self) -> list:
        return [f for f in self._frame_checks if not f['ok']]

    def placement_events(self) -> list:
        return [dict(event) for event in getattr(self, '_placement_events', ())]

    def seg_frame(self):
        if self._seg_annot is None:
            return None
        return self._seg_annot.get_data()

    def list_objects(self):
        from pxr import UsdGeom
        out = []
        objs_root = self._stage.GetPrimAtPath('/World/objs')
        if not objs_root:
            return out
        for prim in objs_root.GetAllChildren():
            for obj in prim.GetAllChildren():
                xf = UsdGeom.Xformable(obj)
                m = xf.ComputeLocalToWorldTransform(self._Usd.TimeCode.Default())
                t = m.ExtractTranslation()
                out.append({'path': str(obj.GetPath()), 'pos': (t[0], t[1], t[2])})
        return out

    def save_png(self, arr: np.ndarray, path: str):
        from PIL import Image
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(arr).save(path)

    def _cutout_shot(self, alias: str, eye, target, *, pose: str):
        from .goal_image import alias_mask_from_seg
        self.set_camera_lookat(eye, target)
        rgb = self.render()
        packed = self._seg_and_aliases()
        if packed is None:
            return None
        seg, labels, safe2alias = packed
        mask = alias_mask_from_seg(seg, labels, safe2alias, alias)
        total = int(mask.size)
        pixels = int(mask.sum())
        if not total:
            return None
        cover = np.zeros(mask.shape, dtype=bool)
        for child in self._cutout_descendants(alias):
            cover |= alias_mask_from_seg(seg, labels, safe2alias, child)
        return {'rgb': rgb, 'mask': mask, 'cover': cover, 'mask_px': pixels, 'frac': pixels / float(total), 'eye': tuple(map(float, eye)), 'target': tuple(map(float, target)), 'pose': pose}

    def _cutout_probe(self, alias: str, eye, target, *, pose: str):
        from .goal_image import alias_mask_from_seg
        self.set_camera_lookat(eye, target)
        self._ensure_render_product()
        for _ in range(max(1, self._seg_warmup)):
            self._rep.orchestrator.step(rt_subframes=1)
        packed = self._seg_and_aliases()
        if packed is None:
            return None
        seg, labels, safe2alias = packed
        mask = alias_mask_from_seg(seg, labels, safe2alias, alias)
        if not mask.size:
            return None
        pixels = int(mask.sum())
        clipped = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
        ys, xs = np.nonzero(mask)
        bbox_w = int(xs.max() - xs.min() + 1) if len(xs) else 0
        bbox_h = int(ys.max() - ys.min() + 1) if len(ys) else 0
        return {'mask_px': pixels, 'frac': pixels / float(mask.size), 'clipped': clipped, 'bbox_w': bbox_w, 'bbox_h': bbox_h, 'eye': tuple(map(float, eye)), 'target': tuple(map(float, target)), 'pose': pose}

    def _cutout_isolation(self, alias: str):
        from contextlib import contextmanager
        from pxr import UsdGeom
        from .goal_image import isolation_hide_paths

        @contextmanager
        def transaction():
            target_roots = tuple(self._alias_prims.get(alias, ()))
            if not target_roots:
                raise KeyError(f'no rendered prims for alias {alias!r}')
            object_roots = []
            objects = self._stage.GetPrimAtPath('/World/objs')
            if objects and objects.IsValid():
                for room in objects.GetChildren():
                    object_roots.extend((str(child.GetPath()) for child in room.GetChildren()))
            paths = isolation_hide_paths(object_roots, target_roots, ('/World/EvalAgent', '/World/ExteriorSeals', '/World/Ceilings', '/World/FixtureGlow'))
            saved = []
            try:
                for path in paths:
                    prim = self._stage.GetPrimAtPath(path)
                    if not prim or not prim.IsValid() or (not prim.IsA(UsdGeom.Imageable)):
                        continue
                    attr = UsdGeom.Imageable(prim).GetVisibilityAttr()
                    saved.append((attr, attr.HasAuthoredValueOpinion(), attr.Get()))
                    attr.Set(UsdGeom.Tokens.invisible)
                yield
            finally:
                for attr, authored, value in reversed(saved):
                    if authored:
                        attr.Set(value)
                    else:
                        attr.Clear()
        return transaction()

    def _cutout_descendants(self, alias: str) -> tuple[str, ...]:
        children = {}
        for inst in self.inventory():
            for parent in tuple(getattr(inst, 'on', ()) or ()):
                children.setdefault(parent, set()).add(inst.alias)
        found, frontier = (set(), list(sorted(children.get(alias, ()))))
        while frontier:
            child = frontier.pop(0)
            if child in found:
                continue
            found.add(child)
            frontier.extend(sorted(children.get(child, ())))
        return tuple(sorted(found))

    def cutout(self, alias: str, fill: float=0.7):
        try:
            with self._cutout_isolation(alias):
                return self._cutout_isolated(alias, fill)
        except KeyError as exc:
            return (None, str(exc))

    def verification_view(self, alias: str, *, reveal_hidden: bool=False):
        if self._facts is None or self._pose is None:
            return (None, 'scene not loaded')
        box = self._facts.aabbs.get(alias)
        if box is None or alias not in self._facts.by_alias:
            return (None, 'no such rendered alias')
        center = (np.asarray(box[0], dtype=float) + np.asarray(box[1], dtype=float)) / 2.0
        restore_eye, restore_target = self._pose_to_camera(self._pose)
        revealed_parent = None
        if reveal_hidden:
            relation = (getattr(self._facts, 'placement_relations', {}) or {}).get(alias, {})
            parent = str(relation.get('host') or '') if relation.get('relation') == 'inside' else ''
            parent_inst = self._facts.by_alias.get(parent) if parent else None
            was_open = self._open_state.get(parent, getattr(parent_inst, 'is_open', None)) if parent else None
            if parent and was_open is not True:
                opened = self._drive_articulation(parent, 'open')
                if opened.success:
                    revealed_parent = parent
        probes = []
        try:
            candidates = [(restore_eye[0], restore_eye[1], restore_eye[2], 'agent-current')]
            if self._nav is not None:
                stand = self._nav.nearest_standable(alias, (self._pose.x, self._pose.z), prefer_within=2.5)
                if stand is not None:
                    (x, z), _distance = stand
                    candidates.insert(0, (x, z, 1.5, 'agent-near-target'))
                inst = self._facts.by_alias.get(alias)
                centroid = (getattr(self._facts, 'centroids', {}) or {}).get(alias)
                if inst is not None and centroid is not None:
                    ox, oz = (float(centroid[0]), float(centroid[2]))
                    sectors = {}
                    for x, z in self._nav.walkable_cells(inst.room_id):
                        distance = math.dist((x, z), (ox, oz))
                        if not 0.45 <= distance <= 2.75:
                            continue
                        angle = (math.atan2(z - oz, x - ox) + 2 * math.pi) % (2 * math.pi)
                        sector = int(angle / (2 * math.pi) * 12) % 12
                        rank = (abs(distance - 1.5), distance, x, z)
                        if sector not in sectors or rank < sectors[sector][0]:
                            sectors[sector] = (rank, x, z)
                    for sector, (_rank, x, z) in sorted(sectors.items()):
                        candidates.append((x, z, 1.5, f'reachable-orbit-{sector:02d}'))
            unique_candidates = []
            seen_eyes = set()
            for row in candidates:
                key = tuple((round(float(value), 3) for value in row[:3]))
                if key not in seen_eyes:
                    seen_eyes.add(key)
                    unique_candidates.append(row)
            for x, z, height, label in unique_candidates:
                probe = self._cutout_probe(alias, (x, z, height), center, pose=label)
                if probe is not None:
                    probes.append(probe)
            if not probes:
                return (None, 'target is not visible from a contextual camera')
            chosen = max(probes, key=lambda row: (not bool(row.get('clipped')), min(int(row.get('bbox_w', 0)), int(row.get('bbox_h', 0))), int(row.get('mask_px', 0))))
            shot = self._cutout_shot(alias, chosen['eye'], center, pose=str(chosen.get('pose', 'agent')))
            if shot is None or int(shot.get('mask_px', 0)) <= 0:
                return (None, 'target did not reproduce in the verification render')
            rgb = np.asarray(shot['rgb']).copy()
            mask = np.asarray(shot['mask'], dtype=bool)
            ys, xs = np.nonzero(mask)
            if not len(xs):
                return (None, 'target has no verification pixels')
            x0, x1, y0, y1 = (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))
            thickness = max(2, min(rgb.shape[0], rgb.shape[1]) // 240)
            colour = np.asarray((235, 55, 45), dtype=np.uint8)
            rgb[max(0, y0 - thickness):min(rgb.shape[0], y0 + thickness + 1), max(0, x0 - thickness):min(rgb.shape[1], x1 + thickness + 1), :3] = colour
            rgb[max(0, y1 - thickness):min(rgb.shape[0], y1 + thickness + 1), max(0, x0 - thickness):min(rgb.shape[1], x1 + thickness + 1), :3] = colour
            rgb[max(0, y0 - thickness):min(rgb.shape[0], y1 + thickness + 1), max(0, x0 - thickness):min(rgb.shape[1], x0 + thickness + 1), :3] = colour
            rgb[max(0, y0 - thickness):min(rgb.shape[0], y1 + thickness + 1), max(0, x1 - thickness):min(rgb.shape[1], x1 + thickness + 1), :3] = colour
            return (rgb, {'alias': alias, 'pose': chosen.get('pose'), 'mask_px': int(shot['mask_px']), 'bbox': [x0, y0, x1, y1], 'post_grade': True, 'temporarily_opened_parent': revealed_parent})
        finally:
            self.set_camera_lookat(restore_eye, restore_target)
            if revealed_parent is not None:
                self._drive_articulation(revealed_parent, 'close')

    def _cutout_isolated(self, alias: str, fill: float=0.7):
        del fill
        from .goal_image import CUTOUT_MIN_FRAC, CUTOUT_ORBIT_ENOUGH, best_shot, compose_cutout, natural_projection_is_clear, orbit_radii
        from . import audit_transforms
        if self._facts is None or self._scene is None:
            return (None, 'scene not loaded')
        inst = self._facts.by_alias.get(alias)
        box = self._facts.aabbs.get(alias)
        if inst is None or box is None:
            return (None, 'no such rendered alias')
        lo, hi = (np.asarray(box[0], dtype=float), np.asarray(box[1], dtype=float))
        center, size = ((lo + hi) / 2.0, hi - lo)
        extent = max(float(np.max(size)), 0.05)
        shots = []
        stand = self._nav.nearest_standable(alias, (self._pose.x, self._pose.z), prefer_within=2.5) if self._nav else None
        if stand is not None:
            (x, z), _distance = stand
            natural = self._cutout_probe(alias, (x, z, 1.5), center, pose='agent')
            if natural is not None:
                shots.append(natural)
                if natural_projection_is_clear(natural):
                    chosen = natural
                else:
                    chosen = None
            else:
                chosen = None
        else:
            chosen = None
        if chosen is None:
            rooms = audit_transforms.room_boxes_from_geometry(self._scene)
            room_box = rooms.get(inst.room_id)
            heights = (1.5, min(float(center[2] + extent), float(room_box[1][2] - 0.1)) if room_box else 2.5)
            for radius in orbit_radii(extent):
                for height in heights:
                    for index in range(8):
                        angle = 2.0 * math.pi * index / 8.0
                        eye = np.asarray((center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle), height), dtype=float)
                        shot = self._cutout_probe(alias, eye, center, pose='orbit')
                        if shot is not None:
                            shots.append(shot)
                            if shot['frac'] >= CUTOUT_ORBIT_ENOUGH and (not shot.get('clipped', False)) and natural_projection_is_clear(shot):
                                break
                    if shots and shots[-1]['frac'] >= CUTOUT_ORBIT_ENOUGH and (not shots[-1].get('clipped', False)) and natural_projection_is_clear(shots[-1]):
                        break
                if shots and shots[-1]['frac'] >= CUTOUT_ORBIT_ENOUGH and (not shots[-1].get('clipped', False)) and natural_projection_is_clear(shots[-1]):
                    break
            chosen = best_shot(shots)
        if chosen is None:
            best_px = max((shot['mask_px'] for shot in shots), default=0)
            return (None, f'no clear view ({best_px}px)')
        chosen = self._cutout_shot(alias, chosen['eye'], center, pose=chosen['pose'])
        if chosen is None or chosen['frac'] < CUTOUT_MIN_FRAC:
            pixels = 0 if chosen is None else chosen['mask_px']
            return (None, f'selected view did not reproduce ({pixels}px)')
        image, measurements = compose_cutout(chosen['rgb'], chosen['mask'], chosen['cover'])
        if image is None:
            return (None, measurements)
        if measurements.get('clipped'):
            return (None, 'isolated target touches the image boundary')
        from .imgcheck import frame_stats
        light = frame_stats(image)
        distance = float(np.linalg.norm(np.asarray(chosen['eye']) - center))
        info = {'mask_px': chosen['mask_px'], 'frac': round(float(chosen['frac']), 4), 'dist': round(distance, 2), 'fov': None, 'own_px': chosen['mask_px'], 'parts': 0, 'pose': chosen['pose'], **measurements, 'eye': [round(value, 4) for value in chosen['eye']], 'target': [round(float(value), 4) for value in center], 'luminance_mean': light['mean'], 'lit_frac': light['nonblack_frac']}
        return (image, info)

    def _pose_to_camera(self, pose: Pose):
        yaw = math.radians(pose.yaw_deg)
        h = math.radians(pose.horizon_deg)
        camera_offset = float(self._robustness_params().get('camera_height_offset_m', 0.0))
        eye = (pose.x, pose.z, pose.y + camera_offset)
        fwd = (math.cos(h) * math.sin(yaw), math.cos(h) * math.cos(yaw), -math.sin(h))
        target = (eye[0] + fwd[0], eye[1] + fwd[1], eye[2] + fwd[2])
        return (eye, target)

    def _default_pose(self) -> Pose:
        if self._scene is not None and self._scene.instances:
            pts = np.array([i.transform[:3, 3] for i in self._scene.instances if i.kind == 'geometry'] or [i.transform[:3, 3] for i in self._scene.instances])
            c = pts.mean(0)
            return Pose(x=float(c[0]), y=1.5, z=float(c[1]), yaw_deg=0.0)
        return Pose(x=0.0, y=1.5, z=0.0, yaw_deg=0.0)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def visibility_min_pixel_fraction(self) -> float:
        return visibility_policy.for_sim(self)

    def _pixel_ref(self):
        if self._pixel_referer is None:
            from .pixel_refer import PixelReferer
            self._pixel_referer = PixelReferer(self)
        return self._pixel_referer

    def reset(self, episode: Episode) -> Obs:
        from . import scene_loader, scene_facts
        self._restore_pick_place_state()
        self._restore_articulations_for_reset()
        if episode.scene != self._loaded_scene_root:
            if self._loaded_scene_root is not None:
                self._clear_loaded_scene_stage()
            self._alias_prims = {}
            self._alias_prim_matrices = {}
            scene = scene_loader.load_scene(episode.scene)
            self._scene = scene
            self._facts = scene_facts.load_facts(episode.scene)
            from . import navmesh
            self._nav = navmesh.NavMesh(self._facts, agent_radius=self._agent_collision_radius())
            self._planner = None
            self._oracle_planner = None
            self._structural_physics_ready = False
            self._artic_specs = {}
            self._articulations = {}
            self._pivot_reports = {}
            if self._interact_enabled:
                from .sdf_joints import parse_scene_joints
                from .joint_articulation import is_articulated
                for alias, sdf, specs in parse_scene_joints(episode.scene):
                    if is_articulated(specs):
                        self._artic_specs[alias] = (sdf, specs)
                print('Scene validation: %d' % len(self._artic_specs), flush=True)
            self.add_scene(scene, kinds=('geometry', 'furniture', 'object', 'manipuland', 'wall_mounted', 'ceiling_mounted'))
            self._apply_robustness_scene(scene)
            n_seals = self.add_exterior_seals(scene)
            if n_seals:
                print(f'[backend] sealed {n_seals} exterior door opening(s)', flush=True)
            self.add_ceilings(scene)
            light_scale = float(self._robustness_params().get('lighting_scale', 1.0))
            if light_scale == 1.0:
                self.add_lighting()
                self.add_room_lighting(scene)
            else:
                scaled_overrides = {key: float(value) * light_scale for key, value in ROOM_LIGHT_INTENSITY_OVERRIDES.items()}
                self.add_lighting(dome_intensity=400.0 * light_scale, sun_intensity=1500.0 * light_scale)
                self.add_room_lighting(scene, intensity=24000.0 * light_scale, ambient_fill_intensity=3600.0 * light_scale, per_room_intensity=scaled_overrides)
            self._loaded_scene_root = episode.scene
        else:
            self._apply_robustness_scene(self._scene)
        self._episode = episode
        self._pose = episode.agent_pose or self._default_pose()
        if self._collision_move:
            self._ensure_physics()
        self._sync_embodiment_pose()
        self._step_idx = 0
        self._evidence_observations = []
        self._evidence_opened = set()
        self._evidence_mutations = []
        self._evidence_approaches = []
        self._evidence_interactions = []
        self._seen = set()
        self._held = None
        self._held_visual_aabb = None
        self._placed_on = {}
        self._placed_points = {}
        self._placement_events = []
        self._room_overrides = {}
        instances = tuple(getattr(self._facts, 'instances', ()))
        self._open_state = {i.alias: bool(i.is_open) for i in instances if getattr(i, 'is_open', None) is not None}
        self._toggled_state = {i.alias: bool(i.is_toggled) for i in instances if getattr(i, 'is_toggled', None) is not None}
        self._position_overrides = {}
        self._apply_initial_states(episode)
        self._world_model = None
        if instances and hasattr(self._facts, 'room_types'):
            from roam_bench.protocol.firewall import WorldModel
            self._world_model = WorldModel(partial_obs=episode.partial_obs, furniture_prior=episode.furniture_prior, room_prior=episode.room_prior, strict_object_discovery=episode.strict_object_discovery, propose_feedback=episode.propose_feedback, furniture_prior_kinds_only=episode.furniture_prior_kinds_only, world_text=episode.world_text, hand_only=getattr(episode, 'hand_only', False))
            self._world_model.prime(rooms=tuple((self._facts.room_names.get(r, r) for r in self._facts.room_types)), furniture=tuple((i.alias for i in instances if not i.pickupable)), objects=tuple((i.alias for i in instances if i.pickupable)))
        self._observe_room()
        return self.observe()

    def _restore_articulations_for_reset(self) -> None:
        from . import artic_stage
        from .joint_interact import plan_interact, pose_within
        from .relation_rules import restore_articulations
        restore_articulations(self, artic_stage.set_action, plan_interact, pose_within)

    def set_render_skip(self, on: bool=True) -> None:
        self._skip_render = bool(on)

    def observe(self) -> Obs:
        if self._pose is None:
            raise RuntimeError('observe() before reset(): no agent pose')
        give_pose = bool(self._episode and self._episode.give_pose)
        if self._skip_render:
            return Obs(rgb=RENDER_SKIPPED, depth=None, pose=self._pose if give_pose else None, step_idx=self._step_idx)
        eye, target = self._pose_to_camera(self._pose)
        self._sync_embodiment_pose()
        self.set_camera_lookat(eye, target)
        rgb = self.render()
        self._render_seg_at_pose(self._seg_warmup)
        self._perceive_world_from_last_frame()
        depth = None
        if self._episode and self._episode.give_depth:
            if self._depth_annot is None:
                raise RuntimeError('give_depth=True but the depth annotator is unavailable')
            depth = np.asarray(self._depth_annot.get_data(), dtype=np.float32)
        return Obs(rgb=rgb, depth=depth, pose=self._pose if give_pose else None, step_idx=self._step_idx)

    def agent_pose(self) -> Pose:
        if self._pose is None:
            raise RuntimeError('agent_pose() before reset()')
        return self._pose

    def set_collision_move(self, on: bool=True, furniture: bool=True) -> None:
        self._collision_move = bool(on)
        self._collision_furniture = bool(furniture)

    def set_interact(self, on: bool=True) -> None:
        self._interact_enabled = bool(on)

    def _apply_initial_states(self, episode: Episode) -> None:
        for alias, prop, value in episode.initial_states or ():
            if not self._facts or alias not in self._facts.by_alias:
                raise KeyError(f'initial_states names unknown object {alias!r}')
            value = bool(value)
            if prop == 'open':
                if alias not in self._articulations:
                    raise RuntimeError(f'could not set {alias} open={value}: no articulation')
                result = self._drive_articulation(alias, 'open' if value else 'close')
                if not result.success:
                    raise RuntimeError(f'could not set {alias} open={value}: {result.reason}')
            elif prop == 'toggled':
                inst = self._facts.by_alias[alias]
                if not bool(getattr(inst, 'toggleable', False)):
                    raise RuntimeError(f'could not set {alias} toggled={value}: not toggleable')
                self._toggled_state[alias] = value
            else:
                raise ValueError(f"initial_states property must be 'open' or 'toggled', got {prop!r}")

    def _ensure_physics(self) -> None:
        if self._physics_ready:
            return
        from . import physics_colliders as pcol
        pcol.setup_physics_scene(self._stage)
        n_wall = pcol.add_wall_colliders(self._stage, self._loaded_scene_root)
        n_furn = 0
        if self._collision_furniture:
            stats = pcol.add_furniture_colliders(self._stage, self._loaded_scene_root)
            n_furn = stats['pieces']
        import omni.physx
        omni.physx.get_physx_interface().force_load_physics_from_usd()
        for _ in range(5):
            self._app.update()
        self._physics_ready = True
        print(f'[backend] physics cooked: {n_wall} wall boxes + {n_furn} furniture hulls', flush=True)

    def _ensure_structural_physics(self) -> None:
        if self._structural_physics_ready:
            return
        from . import physics_colliders as pcol
        pcol.setup_physics_scene(self._stage)
        n_wall = pcol.add_wall_colliders(self._stage, self._loaded_scene_root)
        import omni.physx
        omni.physx.get_physx_interface().force_load_physics_from_usd()
        for _ in range(5):
            self._app.update()
        self._structural_physics_ready = True
        print(f'Invalid scene geometry. {n_wall} room_geometry boxes (walls + floor, render-excluded)', flush=True)

    def _capsule_sweep_distance(self, x, y, dir_world, dist, body_forward=None):
        import omni.physx, carb
        from .physics_colliders import AGENT_HEIGHT
        from .embodiment import collision_forward_offsets
        radius = self._agent_collision_radius()
        sq = omni.physx.get_physx_scene_query_interface()
        d = carb.Float3(float(dir_world[0]), float(dir_world[1]), float(dir_world[2]))
        best = None
        if body_forward is None:
            body_forward = dir_world
        fx, fy = (float(body_forward[0]), float(body_forward[1]))
        norm = math.hypot(fx, fy)
        if norm > 1e-09:
            fx, fy = (fx / norm, fy / norm)
        boxes = self._humanoid_mesh_boxes()
        if boxes:
            angle = math.atan2(-fx, fy)
            c, s = (math.cos(angle), math.sin(angle))
            rot = carb.Float4(0.0, 0.0, math.sin(angle * 0.5), math.cos(angle * 0.5))
            for centre, half, _path in boxes:
                cx, cy, cz = centre
                pos = carb.Float3(float(x) + c * cx - s * cy, float(y) + s * cx + c * cy, cz)
                hit = sq.sweep_box_closest(carb.Float3(*half), pos, rot, d, float(dist), False)
                got = hit.get('hit') if isinstance(hit, dict) else getattr(hit, 'hit', False)
                if got:
                    hd = hit.get('distance') if isinstance(hit, dict) else getattr(hit, 'distance')
                    best = hd if best is None else min(best, hd)
            return best
        for forward_offset in collision_forward_offsets(getattr(self, '_embodiment', DEFAULT_EMBODIMENT)):
            ox = float(x) + forward_offset * fx
            oy = float(y) + forward_offset * fy
            for h in _capsule_sample_heights(radius, AGENT_HEIGHT):
                o = carb.Float3(ox, oy, h)
                hit = sq.sweep_sphere_closest(radius, o, d, float(dist), False)
                got = hit.get('hit') if isinstance(hit, dict) else getattr(hit, 'hit', False)
                if got:
                    hd = hit.get('distance') if isinstance(hit, dict) else getattr(hit, 'distance')
                    best = hd if best is None else min(best, hd)
        return best

    def _humanoid_overlap_at(self, x: float, y: float, yaw_deg: float) -> bool:
        boxes = self._humanoid_mesh_boxes()
        if not boxes:
            return False
        import omni.physx, carb
        angle = -math.radians(float(yaw_deg))
        c, s = (math.cos(angle), math.sin(angle))
        rot = carb.Float4(0.0, 0.0, math.sin(angle * 0.5), math.cos(angle * 0.5))
        sq = omni.physx.get_physx_scene_query_interface()
        for centre, half, _path in boxes:
            cx, cy, cz = centre
            pos = carb.Float3(float(x) + c * cx - s * cy, float(y) + s * cx + c * cy, cz)
            if sq.overlap_box_any(carb.Float3(*half), pos, rot):
                return True
        return False

    def _safe_turn_amount(self, x: float, y: float, start_yaw: float, requested_deg: float) -> tuple[float, bool]:
        if getattr(self, '_embodiment', DEFAULT_EMBODIMENT) != 'humanoid' or not self._humanoid_mesh_boxes() or abs(requested_deg) < 1e-09:
            return (float(requested_deg), False)
        n = max(1, int(math.ceil(abs(float(requested_deg)) / 3.0)))
        safe = 0.0
        for i in range(1, n + 1):
            amount = float(requested_deg) * i / n
            yaw = (float(start_yaw) - amount) % 360.0
            if self._humanoid_overlap_at(x, y, yaw):
                return (safe, True)
            safe = amount
        return (safe, False)

    def _agent_collision_radius(self) -> float:
        from .embodiment import collision_radius
        return collision_radius(getattr(self, '_embodiment', DEFAULT_EMBODIMENT))

    def move(self, distance_m: float) -> ActionResult:
        requested_distance = float(distance_m)
        params = self._robustness_params()
        sigma = float(params.get('move_sigma_m', 0.0))
        if sigma:
            from .robustness import deterministic_gaussian
            noisy = requested_distance + deterministic_gaussian(int((self._robustness_profile or {}).get('seed', 0)), self._step_idx, 'move', sigma)
            distance_m = math.copysign(max(0.0, abs(noisy)), requested_distance)
        self._film_begin('move', requested_distance)
        p = self._pose
        yaw = math.radians(p.yaw_deg)
        if not self._collision_move:
            self._pose = replace(p, x=p.x + distance_m * math.sin(yaw), z=p.z + distance_m * math.cos(yaw))
            final_pose = self._pose
            self._film_translation(p, final_pose)
            self._pose = final_pose
            self._step_idx += 1
            return ActionResult(success=True, reason=f'You moved {distance_m:.2f} m.', moved_m=distance_m, requested_m=requested_distance, collided=False)
        self._ensure_physics()
        sgn = 1.0 if distance_m >= 0 else -1.0
        req = abs(distance_m)
        dir_world = (sgn * math.sin(yaw), sgn * math.cos(yaw), 0.0)
        body_forward = (math.sin(yaw), math.cos(yaw), 0.0)
        hit_d = self._capsule_sweep_distance(p.x, p.z, dir_world, req, body_forward)
        skin = 0.02
        if hit_d is not None and hit_d - skin < req:
            moved_abs = max(0.0, hit_d - skin)
            collided = True
        else:
            moved_abs = req
            collided = False
        moved = sgn * moved_abs
        self._pose = replace(p, x=p.x + moved * math.sin(yaw), z=p.z + moved * math.cos(yaw))
        final_pose = self._pose
        self._film_translation(p, final_pose)
        self._pose = final_pose
        self._step_idx += 1
        reason = f'You moved {moved_abs:.2f} m and were blocked.' if collided else f'You moved {moved_abs:.2f} m.'
        return ActionResult(success=True, reason=reason, moved_m=moved, requested_m=requested_distance, collided=collided)

    def strafe(self, distance_m: float) -> ActionResult:
        requested = float(distance_m)
        self._film_begin('strafe', requested)
        p = self._pose
        yaw = math.radians(p.yaw_deg)
        sgn = 1.0 if requested >= 0 else -1.0
        direction = (sgn * math.cos(yaw), -sgn * math.sin(yaw), 0.0)
        amount = abs(requested)
        if self._collision_move:
            self._ensure_physics()
            hit_d = self._capsule_sweep_distance(p.x, p.z, direction, amount, (math.sin(yaw), math.cos(yaw), 0.0))
            skin = 0.02
            collided = hit_d is not None and hit_d - skin < amount
            moved_abs = max(0.0, hit_d - skin) if collided else amount
        else:
            collided, moved_abs = (False, amount)
        signed = math.copysign(moved_abs, requested)
        self._pose = replace(p, x=p.x + signed * math.cos(yaw), z=p.z - signed * math.sin(yaw))
        final_pose = self._pose
        self._film_translation(p, final_pose)
        self._pose = final_pose
        self._step_idx += 1
        side = 'right' if requested > 0 else 'left'
        reason = f'You moved {side} {moved_abs:.2f} m and were blocked.' if collided else f'You moved {side} {moved_abs:.2f} m.'
        return ActionResult(True, reason, moved_m=signed, requested_m=requested, collided=collided)

    def turn(self, degrees: float) -> ActionResult:
        requested_degrees = float(degrees)
        params = self._robustness_params()
        sigma = float(params.get('turn_sigma_deg', 0.0))
        if sigma:
            from .robustness import deterministic_gaussian
            noisy = requested_degrees + deterministic_gaussian(int((self._robustness_profile or {}).get('seed', 0)), self._step_idx, 'turn', sigma)
            degrees = math.copysign(max(0.0, abs(noisy)), requested_degrees)
        self._film_begin('turn', requested_degrees)
        p = self._pose
        applied = float(degrees)
        collided = False
        if self._collision_move:
            self._ensure_physics()
            applied, collided = self._safe_turn_amount(p.x, p.z, p.yaw_deg, applied)
        self._pose = replace(p, yaw_deg=(p.yaw_deg - applied) % 360.0)
        final_pose = self._pose
        self._film_rotation(p, yaw_delta=-applied, pitch_delta=0.0)
        self._pose = final_pose
        self._step_idx += 1
        side = 'left' if degrees >= 0 else 'right'
        reason = f'You turned {side} {abs(applied):.0f} degrees and were blocked.' if collided else f'You turned {side} {abs(applied):.0f} degrees.'
        return ActionResult(success=True, reason=reason, turned_deg=applied, collided=collided)

    def look(self, delta_pitch_deg: float) -> ActionResult:
        self._film_begin('look', delta_pitch_deg)
        new_h = self._pose.horizon_deg - delta_pitch_deg
        new_h = max(-30.0, min(60.0, new_h))
        applied = self._pose.horizon_deg - new_h
        p = self._pose
        self._pose = replace(p, horizon_deg=new_h)
        final_pose = self._pose
        self._film_rotation(p, yaw_delta=0.0, pitch_delta=float(new_h - p.horizon_deg))
        self._pose = final_pose
        self._step_idx += 1
        updown = 'up' if delta_pitch_deg >= 0 else 'down'
        return ActionResult(success=True, reason=f'You looked {updown} {abs(applied):.0f} degrees.')

    def _perceive_world_from_last_frame(self):
        wm = getattr(self, '_world_model', None)
        if self._facts is None:
            return
        sa = self._seg_and_aliases()
        visibility_fraction = self.visibility_min_pixel_fraction()
        visible = seg_alias.visible_instances_from_seg(*sa, visibility_fraction) if sa is not None else set()
        rid = self._agent_room_id()
        pose = self._pose
        visible_objects = []
        segmentation_objects = []
        depth_summary = None
        if sa is not None:
            seg_data, id_to_labels, safe2alias = sa
            counts, total_pixels = seg_alias.alias_pixel_counts(*sa)
            for alias, pixel_count in sorted(counts.items()):
                inst = self._facts.by_alias.get(alias)
                segmentation_objects.append({'alias': alias, 'kind': getattr(inst, 'kind', None) if inst is not None else None, 'object_room_id': self._room_overrides.get(alias, getattr(inst, 'room_id', None) if inst else None), 'pixel_count': int(pixel_count), 'pixel_fraction': float(pixel_count) / float(total_pixels) if total_pixels else 0.0, 'passes_visibility_threshold': alias in visible})
            masks = seg_alias.alias_masks_from_seg(seg_data, id_to_labels, safe2alias, only=visible)
            depth_frame = None
            annot = getattr(self, '_depth_annot', None)
            if annot is not None:
                try:
                    depth_frame = np.squeeze(np.asarray(annot.get_data(), dtype=np.float64))
                except Exception:
                    depth_frame = None
            if depth_frame is not None:
                try:
                    sampled = np.asarray(depth_frame)[::16, ::16].reshape(-1)
                    valid_depth = sampled[np.isfinite(sampled) & (sampled > 0.0)]
                    depth_summary = {'sample_count': int(sampled.size), 'valid_count': int(valid_depth.size), 'valid_fraction': float(valid_depth.size) / float(sampled.size) if sampled.size else 0.0, 'median_m': float(np.median(valid_depth)) if valid_depth.size else None, 'p10_m': float(np.quantile(valid_depth, 0.1)) if valid_depth.size else None, 'p90_m': float(np.quantile(valid_depth, 0.9)) if valid_depth.size else None}
                except Exception:
                    depth_summary = None
            for alias in sorted(visible):
                mask = masks.get(alias)
                if mask is None:
                    continue
                ys, xs = np.nonzero(mask)
                if not len(xs):
                    continue
                centroid = seg_alias.instance_centroid_from_seg(seg_data, id_to_labels, safe2alias, alias)
                world_point = self._film_world_from_depth(*centroid, depth_data=depth_frame) if centroid is not None else None
                point_strategy = 'centroid' if world_point is not None else None
                primary_point_failure = None
                point_failure = None
                point_diagnostics = None
                if world_point is None:
                    primary_point_failure = self._film_depth_pixel_failure_reason(centroid, depth_frame)
                    world_point, point_strategy, point_failure, point_diagnostics = self._film_world_from_mask_depth(mask, centroid, depth_frame)
                depth_median = None
                if depth_frame is not None:
                    try:
                        values = depth_frame[mask]
                        values = values[np.isfinite(values) & (values > 0.0)]
                        if values.size:
                            depth_median = float(np.median(values))
                    except Exception:
                        depth_median = None
                inst = self._facts.by_alias.get(alias)
                object_room = self._room_overrides.get(alias, getattr(inst, 'room_id', None) if inst else None)
                relation = (getattr(self._facts, 'placement_relations', {}) or {}).get(alias, {})
                support = self._placed_on.get(alias)
                if support is None and inst is not None and (alias != self._held):
                    on = tuple(getattr(inst, 'on', ()) or ())
                    support = on[0] if on else relation.get('host') if relation.get('relation') == 'on' else None
                relative_bearing = None
                if centroid is not None:
                    try:
                        from .cam_visibility import intrinsics_px
                        fx, _fy = intrinsics_px(self._width, self._height)
                        relative_bearing = -math.degrees(math.atan((float(centroid[0]) + 0.5 - self._width / 2.0) / fx))
                    except Exception:
                        relative_bearing = None
                edge_world_points = {}
                extrema = {'left': np.flatnonzero(xs == xs.min()), 'right': np.flatnonzero(xs == xs.max()), 'top': np.flatnonzero(ys == ys.min()), 'bottom': np.flatnonzero(ys == ys.max())}
                for name, indices in extrema.items():
                    if not len(indices):
                        continue
                    chosen_i = int(indices[len(indices) // 2])
                    point = self._film_world_from_depth(int(xs[chosen_i]), int(ys[chosen_i]), depth_data=depth_frame)
                    if point is not None:
                        edge_world_points[name] = [float(q) for q in point]
                visible_objects.append({'alias': alias, 'kind': getattr(inst, 'kind', None) if inst is not None else None, 'object_room_id': object_room, 'pixel_count': int(counts.get(alias, 0)), 'pixel_fraction': float(counts.get(alias, 0)) / float(total_pixels) if total_pixels else 0.0, 'bbox': [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], 'centroid_pixel': list(centroid) if centroid is not None else None, 'median_depth': depth_median, 'relative_bearing_deg': relative_bearing, 'world_point_from_depth': list(world_point) if world_point is not None else None, 'world_point_from_depth_strategy': point_strategy, 'world_point_primary_failure_reason': primary_point_failure, 'world_point_failure_reason': point_failure, 'world_point_depth_diagnostics': point_diagnostics, 'edge_world_points_from_depth': edge_world_points, 'position_error_m': max(0.15, 2.0 / math.sqrt(max(1, int(counts.get(alias, 0))))), 'is_open': self._open_state.get(alias, getattr(inst, 'is_open', None)) if inst is not None else None, 'is_toggled': self._toggled_state.get(alias, getattr(inst, 'is_toggled', None)) if inst is not None else None, 'held': alias == self._held, 'support': support, 'placement_relation': dict(relation) if relation else None})
        self._evidence_observations.append({'observation_index': len(self._evidence_observations), 'step_idx': int(self._step_idx), 'room_id': rid, 'pose': {'x': float(pose.x), 'z': float(pose.z), 'yaw_deg': float(pose.yaw_deg), 'horizon_deg': float(pose.horizon_deg)} if pose else None, 'depth_summary': depth_summary, 'visibility_min_pixels': int(getattr(self._episode, 'view_min_pixels', VISIBLE_GATE_MIN_PIXELS)), 'visibility_min_pixel_fraction': visibility_fraction, 'segmentation_objects': segmentation_objects, 'visible_aliases': sorted(visible), 'visible_objects': visible_objects})
        if wm is None:
            return
        room = self._facts.room_names.get(rid, rid) if rid else None
        relations = tuple(((i.alias, 'on', host) for i in self.inventory() for host in tuple(i.on or ())))
        wm.observe(visible=visible, rooms=(room,) if room else (), relations=relations, holding=self._held)

    def exploration_evidence(self) -> dict:
        return {'observations': [dict(row) for row in self._evidence_observations], 'opened': sorted(self._evidence_opened), 'mutations': [dict(row) for row in self._evidence_mutations], 'approaches': [dict(row) for row in self._evidence_approaches], 'interaction_events': [dict(row) for row in self._evidence_interactions], 'placement_relations': {str(alias): dict(row) for alias, row in (getattr(self._facts, 'placement_relations', {}) or {}).items()}}

    def _record_approach(self, alias: str | None, result: ActionResult, verb: str) -> None:
        if not alias:
            return
        if not hasattr(self, '_evidence_approaches'):
            self._evidence_approaches = []
        self._evidence_approaches.append({'step_idx': int(getattr(self, '_step_idx', 0)), 'after_observation_index': len(getattr(self, '_evidence_observations', ())) - 1, 'alias': str(alias), 'verb': str(verb), 'success': bool(getattr(result, 'success', False)), 'collided': bool(getattr(result, 'collided', False)), 'moved_m': float(getattr(result, 'moved_m', 0.0) or 0.0), 'reason': str(getattr(result, 'reason', '') or '')})

    def world(self):
        from roam_bench.protocol.types import HELD_UNNAMED, WorldView
        wm = getattr(self, '_world_model', None)
        if wm is None:
            episode = getattr(self, '_episode', None)
            if episode is not None and (not getattr(episode, 'world_text', True)):
                return WorldView(blind=True)
            if episode is not None and getattr(episode, 'hand_only', False):
                return WorldView(hand_only=True, holding=HELD_UNNAMED if self._held is not None else None)
            f = self._facts
            if f is None:
                return WorldView(blind=True)
            rid = self._agent_room_id()
            visible = tuple((i.alias for i in f.instances if i.room_id == rid))
            return WorldView(rooms=tuple((f.room_names.get(r, r) for r in f.room_types)), furniture=tuple((i.alias for i in f.instances if not i.pickupable and i.alias in self._seen)), objects=tuple((i.alias for i in f.instances if i.pickupable and i.alias in self._seen)), relations=(), holding=HELD_UNNAMED if self._held else None, here=self.agent_room_name(), visible=visible)
        sa = self._seg_and_aliases()
        visible = seg_alias.visible_instances_from_seg(*sa, self.visibility_min_pixel_fraction()) if sa is not None else set()
        return wm.view(here=self.agent_room_name() or self.agent_room(), visible=visible)

    def propose(self, text: str):
        from roam_bench.protocol.types import ProposeResult
        wm = getattr(self, '_world_model', None)
        if self._facts is None:
            return ProposeResult(accepted=False, granted=text, reason='there is no scene loaded.', phantom=True)
        if wm is None:
            t = text.lower()
            rid = self._agent_room_id()
            for inst in self._facts.instances:
                key = (inst.kind or '').lower().replace('_', ' ')
                if inst.room_id == rid and key and (key in t):
                    return ProposeResult(True, inst.alias, 'granted.')
            return ProposeResult(False, text, 'nothing here matches that.', phantom=True)
        if not self._episode.strict_object_discovery:
            return ProposeResult(False, None, 'You do not need to name things in this task; you are told what you see.')
        visible = self.visible_instances()
        candidates = []
        for alias in visible:
            d = self._interaction_distance(alias)
            if d is not None and d <= self._episode.propose_max_distance:
                candidates.append((d, alias))
        from roam_bench.protocol.firewall import NONE, normalise
        wm.n_proposals += 1
        label = normalise(text)
        if not label:
            return ProposeResult(False, None, 'Say what you think you see.')
        if wm.feedback == NONE and wm.labels.get(label) is not None:
            return ProposeResult(True, label, 'Noted.')
        ordered = [a for _d, a in sorted(candidates)]
        same_kind = [a for a in ordered if normalise(self._facts.by_alias[a].kind) == label]
        fresh = next((a for a in same_kind if a not in wm.discovered), None)
        if fresh is not None:
            wm.discovered.add(fresh)
            granted = wm._label_for(fresh, text)
            return ProposeResult(True, granted, 'Noted.') if wm.feedback == NONE else ProposeResult(True, granted, f'Yes -- you can refer to it as {granted}.')
        if same_kind:
            return ProposeResult(True, wm._label_for(same_kind[0], text), 'Noted.')
        if wm.feedback == NONE:
            wm.labels.setdefault(label, None)
            wm.n_phantoms += 1
            return ProposeResult(True, label, 'Noted.', phantom=True)
        return ProposeResult(False, None, "You are not close enough to anything like that, or it is not there. Get within arm's reach of what you want to name.")

    def navigate_to_relative(self, deg: float, dist_m: float) -> ActionResult:
        self._film_begin('navigate_to_relative', (deg, dist_m))
        p = self._pose
        applied_turn = float(deg)
        collided = False
        if self._collision_move:
            self._ensure_physics()
            applied_turn, collided = self._safe_turn_amount(p.x, p.z, p.yaw_deg, applied_turn)
        yaw = (p.yaw_deg - applied_turn) % 360.0
        rad = math.radians(yaw)
        moved = 0.0 if collided else float(dist_m)
        if self._collision_move and (not collided) and (abs(moved) > 1e-09):
            sign = 1.0 if moved >= 0.0 else -1.0
            direction = (sign * math.sin(rad), sign * math.cos(rad), 0.0)
            body_forward = (math.sin(rad), math.cos(rad), 0.0)
            hit = self._capsule_sweep_distance(p.x, p.z, direction, abs(moved), body_forward)
            if hit is not None and hit - 0.02 < abs(moved):
                moved = sign * max(0.0, float(hit) - 0.02)
                collided = True
        self._pose = replace(p, x=p.x + moved * math.sin(rad), z=p.z + moved * math.cos(rad), yaw_deg=yaw)
        final_pose = self._pose
        self._film_rotation(p, yaw_delta=-applied_turn, pitch_delta=0.0)
        turned = replace(p, yaw_deg=yaw)
        self._film_translation(turned, final_pose)
        self._pose = final_pose
        self._step_idx += 1
        self._observe_room()
        reason = f'You moved {abs(moved):.2f} m and were blocked.' if collided else f'You moved {abs(moved):.2f} m.'
        return ActionResult(not collided, reason, moved_m=moved, turned_deg=-applied_turn, requested_m=dist_m, collided=collided)

    def _face_and_pitch(self, cx: float, cz: float, alias: str) -> Pose:
        eye_y = self._pose.y if self._pose else 1.5
        if alias is None:
            keep_yaw = self._pose.yaw_deg if self._pose else 0.0
            keep_h = self._pose.horizon_deg if self._pose else 0.0
            return Pose(x=cx, y=eye_y, z=cz, yaw_deg=keep_yaw, horizon_deg=keep_h)
        c = self._facts.centroids.get(alias) if self._facts else None
        ox, oz = (c[0], c[2]) if c else (cx, cz)
        yaw = math.degrees(math.atan2(ox - cx, oz - cz)) % 360.0
        horizon = 0.0
        if c is not None:
            dist = math.hypot(ox - cx, oz - cz)
            if dist > 1e-06:
                horizon = max(-30.0, min(60.0, math.degrees(math.atan2(eye_y - c[1], dist))))
        return Pose(x=cx, y=eye_y, z=cz, yaw_deg=yaw, horizon_deg=horizon)

    @staticmethod
    def _walk_hb(tag: str, i: int, n: int, walked: float, sweeps: int, t0: float) -> None:
        print(f'[hb] walk {tag} wp {i}/{n} walked={walked:.2f}m sweeps={sweeps} t={time.monotonic() - t0:.1f}s', flush=True)

    def _film_begin(self, kind, arg) -> None:
        if getattr(self, 'on_frame', None) is None or getattr(self, '_film_outer_action', False):
            return
        sink = getattr(self, 'on_label', None)
        if sink is not None:
            try:
                if isinstance(arg, tuple):
                    shown = ', '.join((str(x) for x in arg))
                elif isinstance(arg, float):
                    shown = f'{arg:.2f}'
                else:
                    shown = str(arg)
                sink(f'{kind}({shown})')
            except Exception:
                pass

    def _film_render_rgb(self, rt_subframes: int=1) -> np.ndarray:
        self._ensure_render_product()
        eye, target = self._pose_to_camera(self._pose)
        self._sync_embodiment_pose()
        self.set_camera_lookat(eye, target)
        subframes = max(1, int(rt_subframes), getattr(self, '_film_subframes', 16))
        self._rep.orchestrator.step(rt_subframes=subframes)
        data = np.asarray(self._rgb_annot.get_data())
        if data.ndim == 3 and data.shape[2] == 4:
            data = data[:, :, :3]
        return data.astype(np.uint8)

    def _aim_uv(self):
        pixel = getattr(self, '_aim_pixel', None)
        world = getattr(self, '_aim_world', None)
        if world is None:
            return pixel
        from .cam_visibility import intrinsics_px, project_px
        eye, target = self._pose_to_camera(self._pose)
        fx, fy = intrinsics_px(self._width, self._height)
        uv = project_px(eye, target, world, fx, fy, self._width, self._height)
        if uv is None:
            return None
        u, v = (float(uv[0]), float(uv[1]))
        if u < 0.0 or u >= self._width or v < 0.0 or (v >= self._height):
            return None
        return (u, v)

    def _film_world_from_depth(self, u: int, v: int, depth_data=None):
        annot = getattr(self, '_depth_annot', None)
        if depth_data is None and annot is None:
            return None
        try:
            depth = np.asarray(depth_data if depth_data is not None else annot.get_data())
            if depth.ndim > 2:
                depth = np.squeeze(depth)
            if not (0 <= int(u) < depth.shape[1] and 0 <= int(v) < depth.shape[0]):
                return None
            distance = float(depth[int(v), int(u)])
            if not math.isfinite(distance) or distance <= 0.0:
                return None
            from .cam_visibility import intrinsics_px, unit
            eye, target = self._pose_to_camera(self._pose)
            eye = np.asarray(eye, dtype=np.float64)
            fwd = unit(np.asarray(target, dtype=np.float64) - eye)
            right = unit(np.cross(fwd, np.asarray((0.0, 0.0, 1.0))))
            up = np.cross(right, fwd)
            fx, fy = intrinsics_px(self._width, self._height)
            x = (float(u) + 0.5 - self._width / 2.0) / fx
            y = -(float(v) + 0.5 - self._height / 2.0) / fy
            ray = unit(fwd + x * right + y * up)
            return tuple((float(q) for q in eye + distance * ray))
        except Exception:
            return None

    @staticmethod
    def _film_depth_pixel_failure_reason(pixel, depth_data):
        if pixel is None:
            return 'instance_centroid_unavailable'
        if depth_data is None:
            return 'depth_frame_unavailable'
        try:
            depth = np.asarray(depth_data)
        except Exception:
            return 'depth_frame_conversion_failed'
        if depth.ndim > 2:
            try:
                depth = np.squeeze(depth)
            except Exception:
                return 'depth_frame_squeeze_failed'
        if depth.ndim != 2:
            return 'depth_frame_not_2d'
        try:
            u, v = (int(pixel[0]), int(pixel[1]))
        except (TypeError, ValueError, IndexError):
            return 'invalid_centroid_pixel'
        if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
            return 'centroid_outside_depth_frame'
        try:
            distance = float(depth[v, u])
        except Exception:
            return 'centroid_depth_read_failed'
        if not math.isfinite(distance):
            return 'centroid_depth_non_finite'
        if distance <= 0.0:
            return 'centroid_depth_non_positive'
        return 'camera_unprojection_failed'

    def _film_world_from_mask_depth(self, mask, centroid, depth_data):
        diagnostics = {'mask_pixel_count': 0, 'valid_depth_pixel_count': 0, 'stable_valid_depth_pixel_count': 0, 'candidate_count': 0}
        if depth_data is None:
            return (None, None, 'depth_frame_unavailable', diagnostics)
        try:
            depth = np.asarray(depth_data, dtype=np.float64)
            if depth.ndim > 2:
                depth = np.squeeze(depth)
        except Exception:
            return (None, None, 'depth_frame_conversion_failed', diagnostics)
        if depth.ndim != 2:
            diagnostics['depth_shape'] = list(depth.shape)
            return (None, None, 'depth_frame_not_2d', diagnostics)
        try:
            object_mask = np.asarray(mask, dtype=bool)
        except Exception:
            return (None, None, 'instance_mask_conversion_failed', diagnostics)
        diagnostics['depth_shape'] = list(depth.shape)
        diagnostics['mask_shape'] = list(object_mask.shape)
        if object_mask.ndim != 2:
            return (None, None, 'instance_mask_not_2d', diagnostics)
        if object_mask.shape != depth.shape:
            return (None, None, 'segmentation_depth_shape_mismatch', diagnostics)
        diagnostics['mask_pixel_count'] = int(np.count_nonzero(object_mask))
        if not diagnostics['mask_pixel_count']:
            return (None, None, 'instance_mask_empty', diagnostics)
        valid = object_mask & np.isfinite(depth) & (depth > 0.0)
        diagnostics['valid_depth_pixel_count'] = int(np.count_nonzero(valid))
        if not diagnostics['valid_depth_pixel_count']:
            return (None, None, 'no_valid_depth_in_instance_mask', diagnostics)
        stable = valid.copy()
        stable[[0, -1], :] = False
        stable[:, [0, -1]] = False
        stable[1:-1, 1:-1] &= object_mask[:-2, 1:-1] & object_mask[2:, 1:-1] & object_mask[1:-1, :-2] & object_mask[1:-1, 2:]
        diagnostics['stable_valid_depth_pixel_count'] = int(np.count_nonzero(stable))
        median_depth = float(np.median(depth[valid]))
        diagnostics['valid_depth_median_m'] = median_depth
        cx = float(centroid[0]) if centroid is not None else (depth.shape[1] - 1) / 2.0
        cy = float(centroid[1]) if centroid is not None else (depth.shape[0] - 1) / 2.0
        centroid_xy = None if centroid is None else (int(centroid[0]), int(centroid[1]))
        candidates = []
        seen = set()

        def add_ranked(candidate_mask, strategy, limit):
            ys, xs = np.nonzero(candidate_mask)
            if not len(xs):
                return
            order = np.lexsort(((ys - cy) ** 2 + (xs - cx) ** 2, np.abs(depth[ys, xs] - median_depth)))
            for index in order[:limit]:
                xy = (int(xs[index]), int(ys[index]))
                if xy == centroid_xy or xy in seen:
                    continue
                seen.add(xy)
                candidates.append((xy, strategy))
        add_ranked(stable, 'stable_mask_median_depth', 16)
        add_ranked(valid, 'mask_median_depth', 16)
        diagnostics['candidate_count'] = len(candidates)
        for (u, v), strategy in candidates:
            point = self._film_world_from_depth(u, v, depth_data=depth)
            if point is not None:
                diagnostics['selected_pixel'] = [u, v]
                diagnostics['selected_depth_m'] = float(depth[v, u])
                return (point, strategy, None, diagnostics)
        return (None, None, 'all_valid_mask_pixels_failed_unprojection', diagnostics)

    def _film_frame(self, repeat: int=1, *, rt_subframes: int=1, rerender: bool=False, sampled: bool=False) -> None:
        sink = getattr(self, 'on_frame', None)
        if sink is None:
            return
        multiplier = 1 if sampled else getattr(self, '_film_rate_multiplier', 1)
        repeats = max(1, int(repeat)) * multiplier
        count = repeats if rerender else 1
        copies = 1 if rerender else repeats
        for _index in range(count):
            try:
                frame = self._film_render_rgb() if int(rt_subframes) == 1 else self._film_render_rgb(rt_subframes=rt_subframes)
            except Exception:
                return
            aim_sink = getattr(self, 'on_aim', None)
            if aim_sink is not None and (getattr(self, '_aim_world', None) is not None or getattr(self, '_aim_pixel', None) is not None):
                try:
                    aim_sink(self._aim_uv())
                except Exception:
                    pass
            try:
                for _ in range(copies):
                    sink(frame)
            except Exception:
                return

    def _film_translation(self, start: Pose, end: Pose, spacing: float=0.1) -> None:
        trace = getattr(self, '_model_motion_trace', None)
        if trace is not None:
            trace.append(('translation', start, end))
        if getattr(self, 'on_frame', None) is None:
            if trace is not None:
                self._pose = end
            return
        dist = math.hypot(end.x - start.x, end.z - start.z)
        n = max(1, int(math.ceil(dist / spacing))) * getattr(self, '_film_rate_multiplier', 1)
        for i in range(1, n + 1):
            t = i / n
            self._pose = replace(start, x=start.x + (end.x - start.x) * t, z=start.z + (end.z - start.z) * t, yaw_deg=end.yaw_deg, horizon_deg=end.horizon_deg)
            self._film_frame(sampled=True)

    def begin_model_motion_trace(self) -> None:
        self._model_motion_trace = []

    def finish_model_motion_trace(self):
        trace = list(getattr(self, '_model_motion_trace', ()) or ())
        self._model_motion_trace = None
        return trace

    def capture_model_motion_frames(self, trace, *, spacing_m: float=1.0, rotation_deg: float=30.0, maximum: int=8):
        from .intermediate_frames import sample_motion
        poses = sample_motion(trace, spacing_m=spacing_m, rotation_deg=rotation_deg, maximum=maximum)
        if not poses:
            return []
        final_pose = self._pose
        observations = self._evidence_observations
        world_model = self._world_model
        frames = []
        try:
            self._world_model = None
            for pose in poses:
                self._pose = pose
                self._evidence_observations = []
                rgb = self.observe().rgb.copy()
                frames.append((rgb, list(self._evidence_observations)))
        finally:
            self._pose = final_pose
            self._world_model = world_model
            self._evidence_observations = observations
        for _rgb, evidence in frames:
            for row in evidence:
                observations.append(dict(row, observation_index=len(observations)))
        return [rgb for rgb, _evidence in frames]

    def _film_rotation(self, start: Pose, yaw_delta: float, pitch_delta: float, increment: float=5.0) -> None:
        trace = getattr(self, '_model_motion_trace', None)
        if trace is not None:
            trace.append(('rotation', start, (yaw_delta, pitch_delta)))
        if getattr(self, 'on_frame', None) is None:
            if trace is not None:
                self._pose = replace(start, yaw_deg=(start.yaw_deg + yaw_delta) % 360.0, horizon_deg=start.horizon_deg + pitch_delta)
            return
        amount = max(abs(yaw_delta), abs(pitch_delta))
        n = max(1, int(math.ceil(amount / increment))) * getattr(self, '_film_rate_multiplier', 1)
        for i in range(1, n + 1):
            t = i / n
            self._pose = replace(start, yaw_deg=(start.yaw_deg + yaw_delta * t) % 360.0, horizon_deg=start.horizon_deg + pitch_delta * t)
            self._film_frame(sampled=True)

    @staticmethod
    def _film_yaw_delta(start_deg: float, end_deg: float) -> float:
        return (float(end_deg) - float(start_deg) + 180.0) % 360.0 - 180.0

    def _film_transition(self, start: Pose, end: Pose) -> None:
        if getattr(self, 'on_frame', None) is None and getattr(self, '_model_motion_trace', None) is None:
            return
        self._pose = start
        self._film_rotation(start, yaw_delta=self._film_yaw_delta(start.yaw_deg, end.yaw_deg), pitch_delta=float(end.horizon_deg - start.horizon_deg))
        turned = replace(start, yaw_deg=end.yaw_deg, horizon_deg=end.horizon_deg)
        self._film_translation(turned, end)

    def _film_walk_segment(self, start_xz, end_xz, template: Pose) -> None:
        if getattr(self, 'on_frame', None) is None and getattr(self, '_model_motion_trace', None) is None:
            return
        dx, dz = (end_xz[0] - start_xz[0], end_xz[1] - start_xz[1])
        if math.hypot(dx, dz) < 1e-09:
            return
        yaw = math.degrees(math.atan2(dx, dz)) % 360.0
        visual = getattr(self, '_pose', None) or template
        start = replace(visual, x=start_xz[0], z=start_xz[1])
        aimed = replace(start, yaw_deg=yaw)
        self._film_rotation(start, self._film_yaw_delta(start.yaw_deg, yaw), 0.0)
        self._film_translation(aimed, replace(aimed, x=end_xz[0], z=end_xz[1]))

    def _navigate_common(self, alias: str) -> ActionResult:
        p = self._pose
        got = self._nav.nearest_standable(alias, (p.x, p.z), prefer_within=self._navigation_standoff_radius(alias))
        if got is None:
            return ActionResult(False, 'You cannot find a way to stand next to it.')
        (cx, cz), d_obj = got
        if not self._collision_move:
            moved = math.hypot(cx - p.x, cz - p.z)
            self._pose = self._face_and_pitch(cx, cz, alias)
            final_pose = self._pose
            self._film_transition(p, final_pose)
            self._pose = final_pose
            self._step_idx += 1
            self._observe_room()
            if d_obj <= VISIBILITY_DISTANCE + 0.05:
                return ActionResult(True, f'You walked {moved:.2f} m and are now next to it.', moved_m=moved, collided=False)
            return ActionResult(False, f'You walked {moved:.2f} m but could not reach it.', moved_m=moved, collided=False)
        return self._navigate_walk(alias, (cx, cz), d_obj)

    def _navigation_standoff_radius(self, alias: str) -> float:
        return VISIBILITY_DISTANCE

    def _ensure_path_planner(self):
        if getattr(self, '_planner', None) is not None:
            return self._planner
        if self._nav is None or not getattr(self, '_loaded_scene_root', None):
            raise RuntimeError('cannot build a path planner before a scene is loaded')
        from .astar_path import PathPlanner
        from .physics_colliders import parse_wall_boxes
        body_radius = self._agent_collision_radius()
        self._planner = PathPlanner(self._nav, wall_boxes=parse_wall_boxes(self._loaded_scene_root), agent_radius=body_radius + 0.02, furniture_radius=body_radius)
        return self._planner

    def _ensure_oracle_path_planner(self):
        return self._ensure_path_planner()

    def _navigate_walk(self, alias: str, goal_xz, d_obj: float) -> ActionResult:
        self._ensure_physics()
        self._ensure_path_planner()
        p = self._pose
        plan = self._planner.plan((p.x, p.z), (goal_xz[0], goal_xz[1]))
        if plan is None:
            self._step_idx += 1
            return ActionResult(False, 'You cannot find a walkable path to stand next to it.', moved_m=0.0, collided=False)
        poly, _plan_len = plan
        walked = 0.0
        collided = False
        cur = (float(p.x), float(p.z))
        body_yaw = float(p.yaw_deg)
        skin = 0.02
        _hb_n, _hb_t0, _hb_sweeps = (len(poly) - 1, time.monotonic(), 0)
        for _hb_i, nxt in enumerate(poly[1:], 1):
            self._walk_hb(alias, _hb_i, _hb_n, walked, _hb_sweeps, _hb_t0)
            seg = math.hypot(nxt[0] - cur[0], nxt[1] - cur[1])
            if seg < 1e-06:
                continue
            ux, uz = ((nxt[0] - cur[0]) / seg, (nxt[1] - cur[1]) / seg)
            target_yaw = math.degrees(math.atan2(ux, uz)) % 360.0
            turn_request = (body_yaw - target_yaw + 180.0) % 360.0 - 180.0
            safe_turn, turn_blocked = self._safe_turn_amount(cur[0], cur[1], body_yaw, turn_request)
            body_yaw = (body_yaw - safe_turn) % 360.0
            if turn_blocked:
                collided = True
                break
            boxes = self._humanoid_mesh_boxes()
            _hb_sweeps += len(boxes) if boxes else 3
            hit_d = self._capsule_sweep_distance(cur[0], cur[1], (ux, uz, 0.0), seg, (ux, uz, 0.0))
            if hit_d is not None and hit_d - skin < seg:
                adv = max(0.0, hit_d - skin)
                walked += adv
                old_cur = cur
                cur = (cur[0] + adv * ux, cur[1] + adv * uz)
                self._film_walk_segment(old_cur, cur, p)
                collided = True
                break
            walked += seg
            old_cur = cur
            cur = (nxt[0], nxt[1])
            self._film_walk_segment(old_cur, cur, p)
        filmed_pose = self._pose
        self._pose = p
        desired_pose = self._face_and_pitch(cur[0], cur[1], alias)
        face_request = (body_yaw - desired_pose.yaw_deg + 180.0) % 360.0 - 180.0
        safe_face, face_blocked = self._safe_turn_amount(cur[0], cur[1], body_yaw, face_request)
        if face_blocked:
            collided = True
        final_pose = replace(desired_pose, yaw_deg=(body_yaw - safe_face) % 360.0)
        self._film_transition(filmed_pose, final_pose)
        self._pose = final_pose
        self._step_idx += 1
        self._observe_room()
        if collided:
            return ActionResult(False, f'You walked {walked:.2f} m but were blocked before reaching it.', moved_m=walked, collided=True)
        if d_obj <= VISIBILITY_DISTANCE + 0.05:
            return ActionResult(True, f'You walked {walked:.2f} m and are now next to it.', moved_m=walked, collided=False)
        return ActionResult(False, f'You walked {walked:.2f} m but could not reach it.', moved_m=walked, collided=False)

    def go_to_visible_object(self, ref, *, pixel_hit_visible: bool=False) -> ActionResult:
        alias, resolve_error = self._resolve_name_ref(ref)
        self._film_begin('go_to_visible_object', alias or ref)
        if resolve_error:
            return ActionResult(False, resolve_error)
        f, p = (self._facts, self._pose)
        if f is None or self._nav is None or p is None:
            return ActionResult(False, 'There is no scene loaded.')
        inst = f.by_alias.get(alias) if alias else None
        if inst is None:
            return ActionResult(False, 'There is no such object here.')
        if not self._visible_gate:
            if inst.room_id != self._agent_room_id():
                return ActionResult(False, 'You cannot see that from where you are standing.')
            return self._navigate_common(alias)
        visible = bool(pixel_hit_visible) or alias in self.visible_instances(VISIBLE_GATE_MIN_PIXEL_FRAC)
        reachable = self._nav.nearest_standable(alias, (p.x, p.z), prefer_within=VISIBILITY_DISTANCE) is not None
        decision = visible_gate.decide(visible=visible, reachable=reachable)
        if not decision.passed:
            return ActionResult(False, decision.reason)
        result = self._navigate_common(alias)
        if not getattr(self, '_film_outer_action', False):
            self._record_approach(alias, result, 'go_to_visible_object')
        return result

    def navigate_to_object(self, ref) -> ActionResult:
        alias, resolve_error = self._resolve_name_ref(ref)
        self._film_begin('navigate_to_object', alias or ref)
        if resolve_error:
            return ActionResult(False, resolve_error)
        f, p = (self._facts, self._pose)
        if f is None or self._nav is None or p is None:
            return ActionResult(False, 'There is no scene loaded.')
        inst = f.by_alias.get(alias) if alias else None
        if inst is None:
            return ActionResult(False, 'There is no such object here.')
        if (getattr(inst, 'kind', '') or '').lower() in _STRUCTURAL:
            return ActionResult(False, 'That is part of the room, not something to walk to.')
        if not self._nav.reachable(self._agent_room_id(), inst.room_id):
            return ActionResult(False, 'You cannot find a way to stand next to it.')
        result = self._navigate_common(alias)
        self._record_approach(alias, result, 'navigate_to_object')
        return result

    def oracle_path_to_object(self, ref, max_goal_distance_m=None, excluded_goal_xz=(), preferred_goal_xz=None):
        alias, error = self._resolve_name_ref(ref)
        p = self._pose
        if error or alias is None or p is None or (self._nav is None):
            return None
        planner = self._ensure_oracle_path_planner() if max_goal_distance_m is not None else self._ensure_path_planner()
        if max_goal_distance_m is None:
            inst = self._facts.by_alias.get(alias)
            if inst is None:
                return None
            distance_to_target = lambda cell: self._horizontal_distance_to_alias(alias, float(cell[0]), float(cell[1]))
            candidates = [cell for cell in self._nav.walkable_cells(inst.room_id) if distance_to_target(cell) <= VISIBILITY_DISTANCE]
            candidates.sort(key=lambda cell: math.dist(cell, (p.x, p.z)))
            planned = None
            for gx, gz in candidates:
                if not self.oracle_turn_clearance(gx, gz):
                    continue
                planned = planner.plan((p.x, p.z), (gx, gz))
                if planned is not None:
                    break
            if planned is None:
                return None
            _distance = distance_to_target((gx, gz))
        else:
            limit = VISIBILITY_DISTANCE if max_goal_distance_m is None else min(VISIBILITY_DISTANCE, max(0.0, float(max_goal_distance_m)))
            inst = self._facts.by_alias.get(alias)
            centre = self._facts.centroids.get(alias)
            if inst is None or centre is None:
                return None
            ox, oz = (float(centre[0]), float(centre[2]))
            distance_to_target = lambda cell: self._horizontal_distance_to_alias(alias, float(cell[0]), float(cell[1]))
            excluded = tuple(((float(q[0]), float(q[1])) for q in excluded_goal_xz))
            candidates = [cell for cell in self._nav.walkable_cells(inst.room_id) if distance_to_target(cell) <= limit and (not any((math.dist(cell, old) <= 0.45 for old in excluded)))]
            preferred_standoff = min(0.9, max(0.0, limit - 0.2))
            candidates.sort(key=lambda cell: (abs(distance_to_target(cell) - preferred_standoff), math.dist(cell, (p.x, p.z))))
            if preferred_goal_xz is not None:
                preferred = tuple(map(float, preferred_goal_xz))
                if not any((math.dist(preferred, old) <= 0.45 for old in excluded)):
                    candidates = [preferred] + [cell for cell in candidates if math.dist(cell, preferred) > 0.05]
            planned = None
            for gx, gz in candidates:
                planned = planner.plan((p.x, p.z), (gx, gz))
                if planned is None:
                    continue
                polyline = planned[0]
                incoming_yaw = float(p.yaw_deg)
                for start, end in reversed(tuple(zip(polyline[:-1], polyline[1:]))):
                    if math.dist(start, end) > 1e-06:
                        incoming_yaw = math.degrees(math.atan2(end[0] - start[0], end[1] - start[1])) % 360.0
                        break
                target_yaw = math.degrees(math.atan2(ox - float(gx), oz - float(gz))) % 360.0
                if self.oracle_turn_sweep_clearance(gx, gz, incoming_yaw, target_yaw):
                    break
                planned = None
            if planned is None:
                return None
            _distance = distance_to_target((gx, gz))
        polyline, length = planned
        return {'alias': alias, 'goal_xz': (float(gx), float(gz)), 'polyline': tuple(((float(x), float(z)) for x, z in polyline)), 'length_m': float(length), 'goal_distance_m': float(_distance)}

    def oracle_turn_clearance(self, x: float, z: float) -> bool:
        planner = self._ensure_path_planner()
        if getattr(self, '_embodiment', DEFAULT_EMBODIMENT) != 'humanoid':
            return True
        from .embodiment import HUMANOID_MESH_BBOX_MIN, HUMANOID_MESH_BBOX_MAX, HUMANOID_MESH_MARGIN
        radial = max((math.hypot(float(xv), float(yv)) for xv in (HUMANOID_MESH_BBOX_MIN[0], HUMANOID_MESH_BBOX_MAX[0]) for yv in (HUMANOID_MESH_BBOX_MIN[1], HUMANOID_MESH_BBOX_MAX[1]))) + float(HUMANOID_MESH_MARGIN)
        if not planner.point_clear((float(x), float(z)), radial):
            return False
        self._ensure_physics()
        return not any((self._humanoid_overlap_at(float(x), float(z), yaw) for yaw in range(0, 360, 30)))

    def oracle_turn_sweep_clearance(self, x: float, z: float, from_yaw_deg: float, to_yaw_deg: float) -> bool:
        if getattr(self, '_embodiment', DEFAULT_EMBODIMENT) != 'humanoid':
            return True
        self._ensure_physics()
        x, z = (float(x), float(z))
        start = float(from_yaw_deg) % 360.0
        if self._humanoid_overlap_at(x, z, start):
            return False
        request = (start - float(to_yaw_deg) + 180.0) % 360.0 - 180.0
        arcs = (request,) if abs(request) < 1e-09 else (request, request - math.copysign(360.0, request))
        for arc in arcs:
            safe, blocked = self._safe_turn_amount(x, z, start, arc)
            if not blocked and abs(float(safe) - float(arc)) <= 1e-06:
                return True
        return False

    def oracle_ground_pixel(self, x: float, z: float, tolerance_m: float=0.75, require_turn_clearance: bool=False, turn_from_yaw_deg: float | None=None, turn_to_yaw_deg: float | None=None, excluded_xz=()):
        if self._pose is None:
            return None
        from .cam_visibility import intrinsics_px, project_px
        self._ensure_structural_physics()
        eye, target = self._pose_to_camera(self._pose)
        fx, fy = intrinsics_px(self._width, self._height)
        projected = project_px(eye, target, (float(x), float(z), 0.0), fx, fy, self._width, self._height)
        if projected is None:
            return None
        u0, v0 = (int(round(float(projected[0]))), int(round(float(projected[1]))))
        local = [(u0 + du, v0 + dv) for du, dv in ((0, 0), (-3, 0), (3, 0), (0, -3), (0, 3), (-8, 0), (8, 0), (0, -8), (0, 8), (-16, 0), (16, 0), (0, -16), (0, 16))]
        grid = [(u, v) for v in range(self._height // 2, self._height - 4, max(8, self._height // 16)) for u in range(4, self._width - 4, max(8, self._width // 24))]
        best = None
        for u, v in local + grid:
            if not (0 <= u < self._width and 0 <= v < self._height):
                continue
            hit = self._pixel_ref().navpoint_at(u, v)
            if not hit.ok:
                continue
            hx, hz, _up = hit.xyz
            rid = self._facts.room_localization_point(float(hx), float(hz)) if self._facts else None
            cells = self._nav.walkable_cells(rid) if rid is not None else ()
            if not cells:
                continue
            nearest = min(cells, key=lambda cell: math.hypot(float(cell[0]) - float(hx), float(cell[1]) - float(hz)))
            if math.hypot(float(nearest[0]) - float(hx), float(nearest[1]) - float(hz)) > 0.35:
                continue
            if any((math.hypot(float(nearest[0]) - float(old[0]), float(nearest[1]) - float(old[1])) <= 0.1 for old in excluded_xz or ())):
                continue
            error = math.hypot(float(nearest[0]) - float(x), float(nearest[1]) - float(z))
            if require_turn_clearance:
                if turn_from_yaw_deg is not None and turn_to_yaw_deg is not None:
                    turn_clear = self.oracle_turn_sweep_clearance(*nearest, turn_from_yaw_deg, turn_to_yaw_deg)
                else:
                    turn_clear = self.oracle_turn_clearance(*nearest)
                if not turn_clear:
                    continue
            if require_turn_clearance:
                dx, dz = (float(nearest[0]) - self._pose.x, float(nearest[1]) - self._pose.z)
                distance = math.hypot(dx, dz)
                if distance > 1e-06:
                    direction = (dx / distance, dz / distance, 0.0)
                    hit_distance = self._capsule_sweep_distance(self._pose.x, self._pose.z, direction, distance, direction)
                    if hit_distance is not None and hit_distance - 0.02 < distance:
                        continue
            if error <= tolerance_m and (best is None or error < best[0]):
                best = (error, int(u), int(v))
        return None if best is None else (best[1], best[2])

    def record_oracle_approach(self, alias: str, result: ActionResult, verb: str='tier_legal_route') -> None:
        self._record_approach(str(alias), result, str(verb))

    def go_to_object_at_pixel(self, u: int, v: int) -> ActionResult:
        self._film_begin('go_to_object_at_pixel', (int(u), int(v)))
        outer = getattr(self, '_film_outer_action', False)
        self._film_outer_action = True
        try:
            f, p = (self._facts, self._pose)
            if f is None or self._nav is None or p is None:
                return ActionResult(False, 'There is no scene loaded.')
            hit = self._pixel_ref().object_at(int(u), int(v))
            if not hit.ok:
                return ActionResult(False, hit.reason)
            result = self.go_to_visible_object(hit.alias, pixel_hit_visible=True)
            self._record_approach(hit.alias, result, 'go_to_object_at_pixel')
            return result
        finally:
            self._film_outer_action = outer

    def go_to_ground_at_pixel(self, u: int, v: int) -> ActionResult:
        self._film_begin('go_to_ground_at_pixel', (int(u), int(v)))
        f, p = (self._facts, self._pose)
        if f is None or self._nav is None or p is None:
            return ActionResult(False, 'There is no scene loaded.')
        self._ensure_structural_physics()
        hit = self._pixel_ref().navpoint_at(int(u), int(v))
        if not hit.ok:
            return ActionResult(False, hit.reason)
        x, z, _up = hit.xyz
        return self._walk_to_ground(x, z)

    def _walk_to_ground(self, x: float, z: float, snap_tolerance: float=0.35) -> ActionResult:
        f, p = (self._facts, self._pose)
        rid = f.room_localization_point(x, z)
        if rid is None:
            return ActionResult(False, 'There is nowhere to stand near that point.')
        cells = self._nav.walkable_cells(rid)
        if not cells:
            return ActionResult(False, 'There is nowhere to stand near that point.')
        cx, cz = min(cells, key=lambda c: math.hypot(c[0] - x, c[1] - z))
        if math.hypot(cx - x, cz - z) > snap_tolerance:
            return ActionResult(False, 'There is no walkable ground near that point.')
        original_pose = p
        p = replace(p, horizon_deg=GROUND_WALK_STABILIZED_HORIZON_DEG)
        if abs(float(original_pose.horizon_deg) - float(p.horizon_deg)) > 1e-06:
            self._pose = original_pose
            self._film_rotation(original_pose, yaw_delta=0.0, pitch_delta=float(p.horizon_deg - original_pose.horizon_deg))
        self._pose = p
        if not self._collision_move:
            moved = math.hypot(cx - p.x, cz - p.z)
            self._pose = self._face_and_pitch(cx, cz, None)
            final_pose = self._pose
            self._film_transition(p, final_pose)
            self._pose = final_pose
            self._step_idx += 1
            self._observe_room()
            return ActionResult(True, f'You walked {moved:.2f} m and are now there.', moved_m=moved, collided=False)
        self._ensure_physics()
        self._ensure_path_planner()
        plan = self._planner.plan((p.x, p.z), (cx, cz))
        if plan is None:
            self._step_idx += 1
            return ActionResult(False, 'You cannot find a walkable path to that point.', moved_m=0.0, collided=False)
        poly, _len = plan
        walked, collided = (0.0, False)
        cur = (float(p.x), float(p.z))
        body_yaw = float(p.yaw_deg)
        skin = 0.02
        _hb_tag = f'ground({x:.2f},{z:.2f})'
        _hb_n, _hb_t0, _hb_sweeps = (len(poly) - 1, time.monotonic(), 0)
        for _hb_i, nxt in enumerate(poly[1:], 1):
            self._walk_hb(_hb_tag, _hb_i, _hb_n, walked, _hb_sweeps, _hb_t0)
            seg = math.hypot(nxt[0] - cur[0], nxt[1] - cur[1])
            if seg < 1e-06:
                continue
            ux, uz = ((nxt[0] - cur[0]) / seg, (nxt[1] - cur[1]) / seg)
            target_yaw = math.degrees(math.atan2(ux, uz)) % 360.0
            turn_request = (body_yaw - target_yaw + 180.0) % 360.0 - 180.0
            safe_turn, turn_blocked = self._safe_turn_amount(cur[0], cur[1], body_yaw, turn_request)
            body_yaw = (body_yaw - safe_turn) % 360.0
            if turn_blocked:
                collided = True
                break
            boxes = self._humanoid_mesh_boxes()
            _hb_sweeps += len(boxes) if boxes else 3
            hit_d = self._capsule_sweep_distance(cur[0], cur[1], (ux, uz, 0.0), seg, (ux, uz, 0.0))
            if hit_d is not None and hit_d - skin < seg:
                adv = max(0.0, hit_d - skin)
                walked += adv
                old_cur = cur
                cur = (cur[0] + adv * ux, cur[1] + adv * uz)
                self._film_walk_segment(old_cur, cur, p)
                collided = True
                break
            walked += seg
            old_cur = cur
            cur = (nxt[0], nxt[1])
            self._film_walk_segment(old_cur, cur, p)
        restore_request = (body_yaw - p.yaw_deg + 180.0) % 360.0 - 180.0
        safe_restore, restore_blocked = self._safe_turn_amount(cur[0], cur[1], body_yaw, restore_request)
        body_yaw = (body_yaw - safe_restore) % 360.0
        if restore_blocked:
            collided = True
        filmed_pose = self._pose
        self._pose = p
        final_pose = replace(self._face_and_pitch(cur[0], cur[1], None), yaw_deg=body_yaw)
        self._film_transition(filmed_pose, final_pose)
        self._pose = final_pose
        self._step_idx += 1
        self._observe_room()
        if collided:
            return ActionResult(False, f'You walked {walked:.2f} m but were blocked before reaching it.', moved_m=walked, collided=True)
        return ActionResult(True, f'You walked {walked:.2f} m and are now there.', moved_m=walked, collided=False)

    def _horizontal_distance_to_alias(self, alias: str, x: float, z: float) -> Optional[float]:
        facts = getattr(self, '_facts', None)
        if facts is None:
            return None
        box = (getattr(facts, 'aabbs', {}) or {}).get(alias)
        centre = (getattr(facts, 'centroids', {}) or {}).get(alias)
        override = getattr(self, '_position_overrides', {}).get(alias)
        if box is not None:
            lo, hi = (np.asarray(box[0], float).copy(), np.asarray(box[1], float).copy())
            if override is not None and centre is not None:
                lo[0] += float(override[0]) - float(centre[0])
                hi[0] += float(override[0]) - float(centre[0])
                lo[1] += float(override[2]) - float(centre[2])
                hi[1] += float(override[2]) - float(centre[2])
            dx = max(float(lo[0]) - x, 0.0, x - float(hi[0]))
            dz = max(float(lo[1]) - z, 0.0, z - float(hi[1]))
            return math.hypot(dx, dz)
        c = override if override is not None else centre
        return None if c is None else math.hypot(float(c[0]) - x, float(c[2]) - z)

    def _interaction_distance(self, alias: str, pixel_uv=None) -> Optional[float]:
        pose = getattr(self, '_pose', None)
        if pose is None:
            return None
        if pixel_uv is not None:
            point = self._placement_point_from_pixel(int(pixel_uv[0]), int(pixel_uv[1]))
            if point is not None:
                return math.hypot(float(point[0]) - pose.x, float(point[1]) - pose.z)
        return self._horizontal_distance_to_alias(alias, pose.x, pose.z)

    def oracle_interaction_distance(self, alias: str, pixel=None) -> Optional[float]:
        return self._interaction_distance(str(alias), pixel_uv=pixel)

    def oracle_interaction_pixel(self, alias: str, verb: str=''):
        ordinary = self.instance_centroid(alias)
        if str(verb).strip().lower() not in ('place', 'put', 'drop'):
            return ordinary
        sa = self._seg_and_aliases()
        annot = getattr(self, '_depth_annot', None)
        if sa is None or annot is None:
            return None
        seg, id_to_labels, safe2alias = sa
        ids = [sid for sid in np.unique(seg).tolist() if seg_alias.resolve_label(seg_alias._lookup_label(id_to_labels, sid), safe2alias) == alias]
        if not ids:
            return None
        mask = np.isin(np.asarray(seg), ids)
        interior = mask.copy()
        interior[1:-1, 1:-1] &= mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
        interior[[0, -1], :] = False
        interior[:, [0, -1]] = False
        ys, xs = np.nonzero(interior)
        if not len(ys):
            ys, xs = np.nonzero(mask)
        try:
            depth = np.asarray(annot.get_data())
            if depth.ndim > 2:
                depth = np.squeeze(depth)
            ds = depth[ys, xs].astype(np.float64)
            valid = np.isfinite(ds) & (ds > 0.0)
            ys, xs, ds = (ys[valid], xs[valid], ds[valid])
            if not len(ys):
                return None
            from .cam_visibility import intrinsics_px, unit
            eye, target = self._pose_to_camera(self._pose)
            eye = np.asarray(eye, dtype=np.float64)
            fwd = unit(np.asarray(target, dtype=np.float64) - eye)
            right = unit(np.cross(fwd, np.asarray((0.0, 0.0, 1.0))))
            up = np.cross(right, fwd)
            fx, fy = intrinsics_px(self._width, self._height)
            rx = (xs.astype(np.float64) + 0.5 - self._width / 2.0) / fx
            ry = -(ys.astype(np.float64) + 0.5 - self._height / 2.0) / fy
            rays = fwd + rx[:, None] * right + ry[:, None] * up
            rays /= np.linalg.norm(rays, axis=1)[:, None]
            points = eye[None, :] + ds[:, None] * rays
            horizontal = np.hypot(points[:, 0] - self._pose.x, points[:, 1] - self._pose.z)
            reachable = horizontal <= VISIBILITY_DISTANCE
            if not reachable.any():
                return None
            top = float(np.max(points[reachable, 2]))
            q95 = float(np.quantile(points[reachable, 2], 0.95))
            support = reachable & (points[:, 2] >= min(top - 0.05, q95))
            cy, cx = (float(ys.mean()), float(xs.mean()))
            choices = np.flatnonzero(support)
            order = choices[np.argsort((ys[choices] - cy) ** 2 + (xs[choices] - cx) ** 2)]
            if len(order) > 512:
                order = order[np.linspace(0, len(order) - 1, 512, dtype=int)]
            valid_support = []
            for index in order:
                point = tuple((float(q) for q in points[index]))
                normal = self._placement_normal_from_pixel(int(xs[index]), int(ys[index]), point, alias)
                if normal is not None and float(normal[2]) >= 0.85:
                    valid_support.append(index)
            if not valid_support:
                return None
            held_box = getattr(self, '_held_visual_aabb', None)
            if held_box is not None:
                held_lo = np.asarray(held_box[1], dtype=float)
                held_hi = np.asarray(held_box[2], dtype=float)
                held_half = np.maximum((held_hi - held_lo)[:2] * 0.5, 0.01)
                held_height = max(0.05, float(held_hi[2] - held_lo[2]))
            else:
                held_half = np.asarray((0.05, 0.05), dtype=float)
                held_height = 0.15
            facts = getattr(self, '_facts', None)
            destination = facts.by_alias.get(alias) if facts is not None else None
            room_id = getattr(destination, 'room_id', None)
            obstacles = []
            for other, packed in (getattr(facts, 'aabbs', {}) or {}).items() if facts is not None else ():
                if other in (alias, getattr(self, '_held', None)):
                    continue
                inst = facts.by_alias.get(other)
                if inst is None or getattr(inst, 'room_id', None) != room_id or str(getattr(inst, 'kind', '')).lower() == 'geometry':
                    continue
                blo, bhi = (np.asarray(packed[0], float), np.asarray(packed[1], float))
                obstacles.append((blo, bhi))

            def footprint_is_clear(index):
                point = points[index]
                margin = 0.04
                for blo, bhi in obstacles:
                    if float(bhi[2]) <= float(point[2]) + 0.01 or float(blo[2]) >= float(point[2]) + held_height + 0.1:
                        continue
                    if float(blo[0]) - held_half[0] - margin <= point[0] <= float(bhi[0]) + held_half[0] + margin and float(blo[1]) - held_half[1] - margin <= point[1] <= float(bhi[1]) + held_half[1] + margin:
                        return False
                return True
            clear_support = [index for index in valid_support if footprint_is_clear(index)]
            if not clear_support:
                return None
            best = clear_support[0]
            return (int(xs[best]), int(ys[best]))
        except Exception:
            return None

    def _resolve_name_ref(self, ref):
        alias = _ref_alias(ref)
        wm = getattr(self, '_world_model', None)
        if not isinstance(ref, NameRef) or wm is None:
            return (alias, None)
        resolved = wm.resolve(ref.name)
        if resolved is not None:
            return (resolved, None)
        if wm.is_phantom(ref.name):
            return (None, 'You reach for it and there is nothing there.')
        if wm.is_shown_kind(ref.name):
            return (None, 'You know it exists, but have not seen which one. Look around first.')
        return (None, 'You have not found anything by that name. Look around, and get close enough to name it.')

    def _drive_articulation(self, alias: str, action: str) -> ActionResult:
        art = self._articulations.get(alias)
        if art is None:
            return ActionResult(False, f'There is nothing to {action} on {alias}.')
        from . import artic_stage
        from .joint_interact import plan_interact, pose_within
        self._ensure_sim_ctx()
        artic_stage.set_action(art['drives'], art['plan'], action)
        for _ in range(self._interact_settle):
            self._sim_ctx.step(render=False)
        readouts = self._read_articulation(art, action, plan_interact, pose_within)
        n_reached = sum((1 for r in readouts if r['reached_target']))
        ok = bool(readouts) and n_reached == len(readouts)
        art['last_readouts'] = readouts
        if ok:
            art['last_action'] = action
            if not hasattr(self, '_open_state'):
                self._open_state = {}
            self._open_state[alias] = action == 'open'
        past = 'opened' if action == 'open' else 'closed'
        return ActionResult(ok, f'You {past} {alias}: {n_reached}/{len(readouts)} joints reached the {action} target.')

    def interact(self, verb: str, ref, into=None) -> ActionResult:
        v = (verb or '').strip().lower()
        canonical = 'pick' if v in {'pick', 'pickup', 'take'} else 'place' if v in {'place', 'put', 'drop'} else v
        held_before = getattr(self, '_held', None)
        open_before = dict(getattr(self, '_open_state', {}) or {})
        toggled_before = dict(getattr(self, '_toggled_state', {}) or {})
        placements_before = len(getattr(self, '_placement_events', ()) or ())
        result = self._interact_impl(verb, ref, into=into)
        target = _ref_alias(ref) or (str(ref) if isinstance(ref, str) else None)
        source = held_before if canonical == 'place' else None
        destination = None
        support = None
        stable = None
        placement_mode = None
        stability_basis = None
        if canonical == 'pick' and result.success:
            target = getattr(self, '_held', None) or target
            source = target
        elif canonical == 'place' and result.success:
            placements = getattr(self, '_placement_events', ()) or ()
            placed = placements[-1] if len(placements) > placements_before else {}
            source = placed.get('held', source)
            destination = placed.get('destination', target)
            support = placed.get('support', destination)
            stable = support == destination and support is not None
            placement_mode = placed.get('mode')
            stability_basis = placed.get('solver') or ('atomic_receptacle_relation' if placement_mode == 'receptacle' else None)
            target = destination
        elif canonical in {'open', 'close'} and result.success:
            changed = [alias for alias, value in self._open_state.items() if open_before.get(alias) != value]
            if len(changed) == 1:
                target = changed[0]
        elif canonical == 'toggle' and result.success:
            changed = [alias for alias, value in self._toggled_state.items() if toggled_before.get(alias) != value]
            if len(changed) == 1:
                target = changed[0]
        before = {'held': held_before, 'is_open': open_before.get(target), 'is_toggled': toggled_before.get(target)}
        after = {'held': getattr(self, '_held', None), 'is_open': getattr(self, '_open_state', {}).get(target), 'is_toggled': getattr(self, '_toggled_state', {}).get(target)}
        if not hasattr(self, '_evidence_interactions'):
            self._evidence_interactions = []
        event = {'step_idx': int(getattr(self, '_step_idx', 0)), 'after_observation_index': len(getattr(self, '_evidence_observations', ())) - 1, 'verb': canonical, 'target': target, 'source': source, 'destination': destination, 'success': bool(result.success), 'before': before, 'after': after}
        if canonical == 'place':
            event.update({'support': support, 'stable_after_settle': stable, 'placement_mode': placement_mode, 'stability_basis': stability_basis})
        self._evidence_interactions.append(event)
        return result

    def _interact_impl(self, verb: str, ref, into=None) -> ActionResult:
        self._film_begin(verb, _ref_alias(ref) or ref)
        if not self._interact_enabled:
            return ActionResult(False, 'interact is disabled (stage-3 hole; call set_interact).')
        v = (verb or '').strip().lower()
        pick_verbs = ('pick', 'pickup', 'take')
        place_verbs = ('place', 'put', 'drop')
        if v not in ('open', 'close', 'toggle', *pick_verbs, *place_verbs):
            return ActionResult(False, f"I don't know how to {verb!r} something.")
        f = self._facts
        if f is None:
            return ActionResult(False, 'There is no scene loaded.')
        pixel_ref = isinstance(ref, PixelRef)
        name_ref = isinstance(ref, NameRef)
        pixel_uv = (int(ref.u), int(ref.v)) if pixel_ref else None
        placement_point = None
        placement_normal = None
        if pixel_ref:
            hit = self._pixel_ref().object_at(int(ref.u), int(ref.v))
            if not hit.ok:
                return ActionResult(False, hit.reason)
            ref = hit
            if v in place_verbs and getattr(self, '_place_mode', 'receptacle') == 'point':
                placement_point = self._placement_point_from_pixel(*pixel_uv)
                if placement_point is None:
                    return ActionResult(False, 'That pixel has no recoverable 3D surface point.')
                pose = getattr(self, '_pose', None)
                if pose is not None and math.hypot(placement_point[0] - pose.x, placement_point[1] - pose.z) > VISIBILITY_DISTANCE:
                    return ActionResult(False, 'The selected 3D point is beyond manipulation reach.')
                placement_normal = self._placement_normal_from_pixel(*pixel_uv, placement_point, hit.alias)
        alias, resolve_error = self._resolve_name_ref(ref)
        if resolve_error:
            return ActionResult(False, resolve_error)
        if alias is None or f.by_alias.get(alias) is None:
            return ActionResult(False, 'There is no such object here.')
        inst = f.by_alias[alias]
        reach_alias = alias
        if v in place_verbs and into is not None:
            reach_alias, destination_error = self._resolve_name_ref(into)
            if destination_error:
                return ActionResult(False, destination_error)
            reach_alias = reach_alias or alias
        distance = self._interaction_distance(reach_alias, pixel_uv=pixel_uv)
        if distance is not None and distance > VISIBILITY_DISTANCE + 0.05:
            return ActionResult(False, f'{reach_alias} is too far away to reach.')
        if name_ref and getattr(self, '_world_model', None) is not None and (reach_alias not in self.visible_instances(VISIBLE_GATE_MIN_PIXEL_FRAC)):
            return ActionResult(False, 'You cannot see that from where you are standing.')
        if v in pick_verbs:
            if self._held is not None:
                return ActionResult(False, 'You are already holding something.')
            if not bool(getattr(inst, 'pickupable', False)):
                return ActionResult(False, f'{alias} cannot be picked up.')
            self._held = alias
            visual_box = self._alias_visual_aabb(alias)
            self._held_visual_aabb = (alias, np.asarray(visual_box[0], float), np.asarray(visual_box[1], float)) if visual_box is not None else None
            self._placed_on.pop(alias, None)
            getattr(self, '_placed_points', {}).pop(alias, None)
            self._room_overrides.pop(alias, None)
            self._film_frame(repeat=8)
            self._set_alias_visible(alias, False)
            self._film_frame()
            return ActionResult(True, f'You picked up {alias}.')
        if v in place_verbs:
            if self._held is None:
                return ActionResult(False, 'You are not holding anything.')
            destination = reach_alias if into is not None else alias
            if into is not None and alias != self._held:
                return ActionResult(False, f'You are not holding {alias}.')
            dest = f.by_alias.get(destination) if destination else None
            if dest is None:
                return ActionResult(False, 'There is no such destination here.')
            if not bool(getattr(dest, 'receptacle', False)):
                return ActionResult(False, f'{destination} cannot hold an object.')
            held = self._held
            self._film_frame(repeat=8)
            placement_outcome = None
            if placement_point is not None:
                placement_outcome = self._point_release_outcome(held, destination, placement_point, placement_normal)
                if placement_outcome is None:
                    return ActionResult(False, 'The object could not be released there without starting inside scene geometry.')
            else:
                self._place_alias_at(held, destination)
            actual_support = (placement_outcome or {}).get('support', destination)
            if actual_support is None:
                self._placed_on.pop(held, None)
            else:
                self._placed_on[held] = actual_support
            final_xyz = (placement_outcome or {}).get('final_xyz')
            room_lookup = getattr(self._facts, 'room_localization_point', None)
            room_id = room_lookup(final_xyz[0], final_xyz[1]) if final_xyz is not None and callable(room_lookup) else getattr(dest, 'room_id', None)
            if room_id is not None:
                self._room_overrides[held] = room_id
            self._held = None
            self._held_visual_aabb = None
            if not hasattr(self, '_placement_events'):
                self._placement_events = []
            event = {'mode': getattr(self, '_place_mode', 'receptacle'), 'held': held, 'destination': destination, 'pixel': list(pixel_uv) if pixel_uv is not None else None}
            if placement_point is not None:
                event['world_xyz'] = [float(q) for q in placement_point]
                event.update(placement_outcome)
            self._placement_events.append(event)
            self._film_frame()
            if placement_outcome is not None:
                support = placement_outcome.get('support')
                suffix = f'; it settled on {support}.' if support else '; it fell to the floor.'
                return ActionResult(True, f'You released {held} at the selected point{suffix}')
            return ActionResult(True, f'You placed {held} on {destination}.')
        if v == 'toggle':
            if not bool(getattr(inst, 'toggleable', False)):
                return ActionResult(False, f'{alias} cannot be toggled.')
            now = not bool(self._toggled_state.get(alias, getattr(inst, 'is_toggled', False)))
            self._toggled_state[alias] = now
            self._film_frame()
            return ActionResult(True, f"You toggled {alias} {('on' if now else 'off')}.")
        self._film_frame(repeat=12)
        result = self._drive_articulation(alias, v)
        if v == 'open' and result.success:
            if not hasattr(self, '_evidence_opened'):
                self._evidence_opened = set()
            self._evidence_opened.add(alias)
        self._film_frame()
        return result

    def _ensure_sim_ctx(self):
        if self._sim_ctx is not None:
            return
        from pxr import UsdPhysics
        from isaacsim.core.api import SimulationContext
        from . import artic_stage
        if not UsdPhysics.Scene.Get(self._stage, '/physicsScene'):
            sc = UsdPhysics.Scene.Define(self._stage, '/physicsScene')
            sc.CreateGravityMagnitudeAttr(0.0)
        self._sim_ctx = SimulationContext(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 120.0, stage_units_in_meters=1.0)
        try:
            self._sim_ctx.get_physics_context().set_gravity(0.0)
        except Exception as e:
            print('Scene validation: %r' % e, flush=True)
        self._sim_ctx.reset()
        for _ in range(self._interact_settle):
            self._sim_ctx.step(render=False)
        for a in self._articulations.values():
            a['closed_pose'] = {jp.joint_name: artic_stage.read_link_pose(self._stage, jp.body1_path) for jp in a['plan'].joints}

    def _read_articulation(self, art, action, plan_interact, pose_within):
        import numpy as np
        from . import artic_stage
        plan = art['plan']
        closed = art.get('closed_pose') or {}
        R = art.get('inst_rot')
        ip = plan_interact(plan, action)
        targets = {jp.joint_name: t.target for jp, t in zip(plan.joints, ip.targets)}
        out = []
        for jp in plan.joints:
            name = jp.joint_name
            now = artic_stage.read_link_pose(self._stage, jp.body1_path)
            base = closed.get(name, now)
            axis_w = artic_stage.world_axis(jp.params.local_rot_wxyz, jp.params.axis)
            if R is not None:
                axis_w = np.asarray(R) @ np.asarray(axis_w)
            (p0, q0), (p1, q1) = (base, now)
            if jp.params.drive == 'angular':
                val = artic_stage.signed_angle_about_deg(q0, q1, axis_w)
            else:
                val = float(np.dot(np.asarray(p1) - np.asarray(p0), axis_w))
            tgt = targets[name]
            out.append({'joint': name, 'drive': jp.params.drive, 'measured': round(val, 5), 'target': round(tgt, 5), 'reached_target': bool(pose_within(val, tgt))})
        return out

    def _seg_and_aliases(self):
        seg = self.seg_frame()
        if not isinstance(seg, dict) or seg.get('data') is None:
            return None
        id_to_labels = (seg.get('info') or {}).get('idToLabels') or {}
        s2a = seg_alias.build_safe2alias(self._facts.by_alias) if self._facts else {}
        return (seg['data'], id_to_labels, s2a)

    def visible_instances(self, min_pixel_frac: float | None=None) -> set:
        self._render_seg_at_pose(self._seg_warmup)
        sa = self._seg_and_aliases()
        if sa is None:
            return set()
        threshold = self.visibility_min_pixel_fraction() if min_pixel_frac is None else float(min_pixel_frac)
        return seg_alias.visible_instances_from_seg(*sa, threshold)

    def instance_at_pixel(self, u: int, v: int):
        sa = self._seg_and_aliases()
        if sa is None:
            return None
        return seg_alias.instance_at_pixel_from_seg(*sa, u, v)

    def instances_in_patch(self, u: int, v: int, k: int=2) -> set:
        sa = self._seg_and_aliases()
        if sa is None:
            return set()
        return seg_alias.instances_in_patch_from_seg(*sa, u, v, k)

    def aim_at_pixel(self, u=None, v=None):
        if u is None:
            self._aim_pixel = None
            self._aim_world = None
            return None
        self._aim_pixel = (float(u), float(v))
        self._aim_world = self._film_world_from_depth(int(u), int(v))
        if self._aim_world is None:
            try:
                hit = self._pixel_ref()._raycast_hit(int(u), int(v))
            except Exception:
                hit = None
            if hit is not None:
                self._aim_world = hit[0]
        return self._aim_world

    def instance_centroid(self, alias: str):
        sa = self._seg_and_aliases()
        if sa is None:
            return None
        return seg_alias.instance_centroid_from_seg(*sa, alias)

    def bearing_to(self, alias: str):
        f, p = (self._facts, self._pose)
        if f is None or p is None:
            return None
        c = getattr(self, '_position_overrides', {}).get(alias, f.centroids.get(alias))
        if c is None:
            return None
        px, pz = (c[0], c[2])
        want = math.degrees(math.atan2(px - p.x, pz - p.z))
        deg = (p.yaw_deg - want + 180.0) % 360.0 - 180.0
        return (deg, math.hypot(px - p.x, pz - p.z))

    def predicate(self, expr: str) -> bool:
        e = (expr or '').strip()
        if ' and ' in e:
            return all((self.predicate(part) for part in e.split(' and ')))

        def args_of(name):
            prefix = name + '('
            if not (e.startswith(prefix) and e.endswith(')')):
                return None
            body = e[len(prefix):-1]
            return tuple((part.strip().strip('\'"') for part in body.split(',')))
        try:
            args = args_of('holding')
            if args is not None:
                return len(args) == 1 and self._held == args[0]
            args = args_of('on')
            if args is not None:
                if len(args) != 2:
                    return False
                if self._placed_on.get(args[0]) == args[1]:
                    return True
                inst = self._facts.by_alias.get(args[0]) if self._facts else None
                return inst is not None and args[0] != self._held and (args[1] in tuple(getattr(inst, 'on', ()) or ()))
            args = args_of('in_room')
            if args is not None:
                if len(args) != 2:
                    return False
                inst = self._facts.by_alias.get(args[0]) if self._facts else None
                room = self._room_overrides.get(args[0], getattr(inst, 'room_id', None) if inst else None)
                return room == args[1]
            args = args_of('open') or args_of('is_open')
            if args is not None:
                states = getattr(self, '_open_state', {})
                fallback = getattr(self, '_articulations', {}).get(args[0], {}).get('last_action') == 'open'
                return len(args) == 1 and bool(states.get(args[0], fallback))
            args = args_of('toggled') or args_of('is_toggled')
            if args is not None:
                return len(args) == 1 and bool(getattr(self, '_toggled_state', {}).get(args[0], False))
            args = args_of('exists')
            if args is not None:
                return len(args) == 1 and bool(self._facts and args[0] in self._facts.by_alias)
            args = args_of('near')
            if args is not None and len(args) in (1, 2):
                radius = float(args[1]) if len(args) == 2 else VISIBILITY_DISTANCE
                d = self._interaction_distance(args[0])
                return d is not None and d <= radius
            args = args_of('visible')
            if args is not None and len(args) in (1, 2):
                frac = float(args[1]) if len(args) == 2 else VISIBLE_GATE_MIN_PIXEL_FRAC
                return args[0] in self.visible_instances(frac)
            if args_of('sliced') is not None or args_of('broken') is not None:
                return False
        except (AttributeError, TypeError, ValueError):
            return False
        return False

    def mutate(self, op: str, alias, arg) -> bool:
        alias = _ref_alias(alias) or alias
        if not self._facts or alias not in self._facts.by_alias:
            return False
        try:
            event_room = self._agent_room_id()
        except (AttributeError, TypeError, ValueError):
            event_room = None
        if op == 'open':
            desired = bool(arg)
            before = bool(self._open_state.get(alias, False))
            if before == desired:
                return False
            result = self._drive_articulation(alias, 'open' if desired else 'close')
            success = bool(result.success and self.predicate(f"is_open('{alias}')") == desired)
            if not hasattr(self, '_evidence_mutations'):
                self._evidence_mutations = []
            self._evidence_mutations.append({'step_idx': int(getattr(self, '_step_idx', 0)), 'after_observation_index': len(getattr(self, '_evidence_observations', ())) - 1, 'alias': str(alias), 'op': str(op), 'success': success, 'agent_room_id': event_room, 'before': before, 'after': bool(self._open_state.get(alias, before))})
            return success
        if op == 'move':
            try:
                target = tuple((float(q) for q in arg))
            except (TypeError, ValueError):
                return False
            if len(target) != 3:
                return False
            before = getattr(self, '_position_overrides', {}).get(alias, self._facts.centroids.get(alias))
            success = self._move_alias_to(alias, target)
            after = getattr(self, '_position_overrides', {}).get(alias, before)
            if not hasattr(self, '_evidence_mutations'):
                self._evidence_mutations = []
            self._evidence_mutations.append({'step_idx': int(getattr(self, '_step_idx', 0)), 'after_observation_index': len(getattr(self, '_evidence_observations', ())) - 1, 'alias': str(alias), 'op': str(op), 'success': bool(success), 'agent_room_id': event_room, 'before': list(before) if before is not None else None, 'after': list(after) if after is not None else None})
            return success
        return False

    def _move_alias_to(self, alias: str, target_thor) -> bool:
        source = self._facts.centroids.get(alias) if self._facts else None
        entries = tuple(getattr(self, '_alias_prim_matrices', {}).get(alias, ()))
        if source is None or not entries:
            return False
        target = tuple((float(q) for q in target_thor))
        before = getattr(self, '_position_overrides', {}).get(alias, source)
        if math.dist(tuple(before), target) <= 1e-06:
            return False
        delta = np.eye(4)
        delta[:3, 3] = (target[0] - source[0], target[2] - source[2], target[1] - source[1])
        try:
            from pxr import UsdGeom
            for path, baseline in entries:
                xf = UsdGeom.Xformable(self._stage.GetPrimAtPath(path))
                ops = xf.GetOrderedXformOps()
                if not ops:
                    return False
                wanted = delta @ baseline
                wanted_gf = self._np_to_gf_matrix(wanted)
                ops[0].Set(wanted_gf)
                got = np.asarray(ops[0].Get(), dtype=float)
                expected = np.asarray(wanted_gf, dtype=float)
                if got.shape != (4, 4) or not np.allclose(got, expected, atol=1e-05):
                    return False
        except Exception:
            return False
        self._position_overrides[alias] = target
        support = self._support_at(target, exclude=alias)
        if support is None:
            self._placed_on.pop(alias, None)
        else:
            self._placed_on[alias] = support
        return True

    def _support_at(self, target_thor, exclude=None):
        if not self._facts:
            return None
        x, y, z = target_thor
        candidates = []
        for alias, box in self._facts.aabbs.items():
            if alias == exclude or box is None:
                continue
            inst = self._facts.by_alias.get(alias)
            if not inst or not bool(getattr(inst, 'receptacle', False)):
                continue
            lo, hi = box
            if lo[0] <= x <= hi[0] and lo[1] <= z <= hi[1]:
                candidates.append((abs(float(hi[2]) - y), alias))
        return min(candidates)[1] if candidates else None

    def inventory(self) -> tuple:
        if not self._facts:
            return ()
        out = []
        for inst in self._facts.instances:
            on = (self._placed_on[inst.alias],) if inst.alias in self._placed_on else tuple(inst.on or ())
            detached = inst.alias in self._placed_on or inst.alias == self._held
            out.append(replace(inst, on=on, inside=() if detached else tuple(inst.inside or ()), is_open=self._open_state.get(inst.alias, inst.is_open) if inst.openable or inst.alias in self._open_state else inst.is_open, is_toggled=self._toggled_state.get(inst.alias, inst.is_toggled) if inst.toggleable or inst.alias in self._toggled_state else inst.is_toggled))
        return tuple(out)

    def room_labels(self) -> tuple:
        return self._facts.room_labels() if self._facts else ()

    def _agent_room_id(self):
        f, p = (self._facts, self._pose)
        if f is None or p is None:
            return None
        return f.room_localization_point(p.x, p.z)

    def agent_room(self):
        rid = self._agent_room_id()
        return self._facts.room_types.get(rid) if rid else None

    def agent_room_name(self):
        rid = self._agent_room_id()
        return self._facts.room_names.get(rid) if rid else None

    def _observe_room(self):
        f = self._facts
        rid = self._agent_room_id()
        if f is None or rid is None:
            return
        for i in f.instances:
            if i.room_id == rid:
                self._seen.add(i.alias)

    def footprint(self, alias: str):
        return self._facts.footprints.get(alias) if self._facts else None

    def room_sizes(self) -> dict:
        return dict(self._facts.room_area) if self._facts else {}

    def room_graph(self, probe_spawns=True) -> dict:
        if not self._facts:
            return {}
        authored = getattr(self._facts, 'room_graph', {}) or {}
        rooms = set(getattr(self._facts, 'room_bounds', {}) or {}) | set(authored)
        for neighbours in authored.values():
            rooms.update(neighbours or {})
        empty = {rid: () for rid in sorted(rooms)}
        if self._nav is None or not getattr(self, '_loaded_scene_root', None):
            return empty
        return self._ensure_path_planner().measured_room_graph()

    def standing_room(self) -> dict:
        if not self._facts or not self._nav:
            return {}
        out = {}
        for inst in self._facts.instances:
            c = self._facts.centroids.get(inst.alias)
            if c is None or not inst.room_id:
                continue
            out[inst.alias] = sum((1 for x, z in self._nav.walkable_cells(inst.room_id) if math.hypot(x - c[0], z - c[2]) <= VISIBILITY_DISTANCE))
        return out

    def close(self):
        try:
            self._app.close()
        except Exception:
            pass

def _safe(name: str) -> str:
    import re
    return re.sub('[^A-Za-z0-9_]', '_', name)
