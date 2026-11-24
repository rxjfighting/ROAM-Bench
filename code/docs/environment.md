# Assets and environment construction

The release includes scene sources, construction code, and task annotations. Isaac Sim and official Unitree G1 assets are obtained separately under their own licenses.

Expected input layout:

```text
scenes/
  scene_148/
    package.xml
    house_layout.json
    combined_house/
      house.dmd.yaml
      house_state.json
    room_<room_id>/...
    materials/...
```

Meshes, SDF files, glTF buffers and textures referenced by these files must be present. Identifiers in `house_state.json` must match the task set.

`scripts/build_scene_bundle.py` exports the scene layout, object state, directives, and their SDF/glTF/mesh/texture dependency graph. It stores portable local references and a scene manifest. Supply `--shared-materials-root` if the input uses a separate material directory. Use `--verify-only` to check dependency links in an exported bundle. The exporter does not generate scenes.

`scripts/build_robot_asset.py` accepts the official G1 directory containing `g1.usd` and its Inspire hand layers. It copies affected layers and their dependency ancestors, removes four wrist references whose internal targets are absent, and produces `g1_inspire.usda`. Other resources remain referenced from the supplied official directory, which must remain accessible. Derived references are relative to their layer locations. Move the official and derived directories together if relocating the environment.

The backend instantiates scene meshes, collision geometry, the robot body, the egocentric camera, and navigation/interaction skills. The robot uses a fixed evaluation pose and Inspire hands. Navigation and manipulation are high-level platform skills. Carrying attaches an object to the hand; placement settles the object and records destination support readback. Generated results retain these events as described in [episode outputs](episode.md).

Inspect scene metadata on CPU with `build_environment.py --inspect`. Stage construction and evaluation require the Isaac Sim runtime and a compatible NVIDIA GPU. Use Isaac Sim 5.1.0 with its Python 3.11 environment. The official [installation guide](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/index.html) lists supported systems and GPU requirements.

For a task with `scene_variant`, the evaluator constructs a content-addressed scene directory under `--scene-cache`. It retains the original scene identifier, rewrites only directives and object metadata, and references the supplied assets. Excluded composite members and declared support relationships are processed together. Cache identity includes the source metadata, directives, and variant configuration. Existing checked support relations retain their source provenance only when the generated metadata and the exact relation still match.

Inspect or export the same task-specific environment used for evaluation:

```bash
python scripts/build_environment.py --dataset ../dataset --assets ../scenes --scene scene_105 --task-id relocation__bedroom_digital_alarm_clock_0__to__kitchen_breakfast_table_0 --inspect
```

Omit `--task-id` to inspect the full scene. Keep the supplied assets accessible while using cached variants; the cache contains metadata and links, not a second copy of the asset bundle.
