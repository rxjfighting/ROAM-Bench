"""Apply the shelf light and remove its layer when changing scenes."""
import json
from pathlib import Path
CONFIGURATION = json.loads(Path(__file__).with_suffix('.json').read_text())
LIGHT_ROOT = '/World/ROAMLighting'

def apply(backend):
    if Path(str(backend._loaded_scene_root)).name != CONFIGURATION['scene']:
        return
    from pxr import Gf, UsdGeom, UsdLux
    if backend._stage.GetPrimAtPath(LIGHT_ROOT).IsValid():
        return
    UsdGeom.Xform.Define(backend._stage, LIGHT_ROOT)
    for index, row in enumerate(CONFIGURATION['lights']):
        light = UsdLux.RectLight.Define(backend._stage, LIGHT_ROOT + '/Panel_' + str(index))
        for name in ('Width', 'Height', 'Intensity', 'Normalize', 'Exposure', 'Diffuse', 'Specular'):
            getattr(light, 'Get' + name + 'Attr')().Set(row[name.lower()])
        light.GetColorAttr().Set(Gf.Vec3f(*row['color']))
        UsdLux.ShadowAPI.Apply(light.GetPrim()).CreateShadowEnableAttr(row['shadow_enable'])
        shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
        shaping.CreateShapingConeAngleAttr(row['cone_angle'])
        shaping.CreateShapingConeSoftnessAttr(row['cone_softness'])
        UsdGeom.Xformable(light.GetPrim()).AddTransformOp().Set(Gf.Matrix4d(*[value for values in row['world_transform_readback'] for value in values]))

def install(backend_type):
    original_render = backend_type.render
    original_clear = backend_type._clear_loaded_scene_stage
    def render(backend, *args, **kwargs):
        if getattr(backend, '_loaded_scene_root', None):
            apply(backend)
        return original_render(backend, *args, **kwargs)
    def clear(backend):
        if backend._stage.GetPrimAtPath(LIGHT_ROOT).IsValid():
            backend._stage.RemovePrim(LIGHT_ROOT)
        return original_clear(backend)
    backend_type.render = render
    backend_type._clear_loaded_scene_stage = clear
