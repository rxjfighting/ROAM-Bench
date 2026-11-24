# Task set

The released set contains 250 tasks across 27 scenes. `manifest.json` indexes each task by scene, identifier, family, group, display type, and charged-action budget. `tasksets/scene_XXX.json` contains instructions, reference-image captions, spawn poses, answer rules, and task-specific observation/action conditions. References use relative paths to `references/`.

| Group | Spatial ability | Tasks | Display types |
|---|---|---:|---|
| G1 | Semantic grounding | 49 | Surface contents; room localization |
| G2 | Exhaustive counting | 38 | Object count; room count |
| G3 | Metric geometry | 77 | Object distance; nearer-object comparison; room area; gap width |
| G4 | Perspective and path integration | 56 | Relative direction; path integration |
| G5 | Object relocation | 30 | Relocation |

There are 11 display types. Two executable families, `room_localization` and `room_comparison`, belong to the room-localization display type. The manifest keeps both executable family and display type explicit.

Tasks are generated from scene structure and geometry, then manually checked for instruction/target consistency, reference identity, answer rules, and task-specific N/S conditions. Numeric tasks retain units and tolerances. Path-integration truth remains pose-dependent: the answer is computed from the initial and final agent poses rather than flattened into a fixed answer. Relocation tasks retain ordered pick/place requirements and destination support checks.

For gap annotations, `truth.reference_gap_cm` records the reference measurement between visual-mesh outlines. The rounded answer and `truth.cm` use the same value. The oracle reader uses the reference measurement and retains the endpoint-observation requirement.

The JSON includes scoring truth for reproducibility. The model sees the instruction and permitted reference images, followed by its observations and feedback; answer labels, object aliases, room maps, and scoring rules are not inserted into its input. Reference images serve the roles specified by their captions, including exact-instance references and category examples.

The set contains 388 unique reference PNGs. Their names are their SHA-256 hashes.

Task specifications may include evaluator-only `scene_variant` records containing excluded aliases and explicitly positioned non-target objects. The evaluation script applies these settings before loading the scene, so rendering, scene facts, collision geometry, navigation, and interaction use the same environment state. These settings are not supplied to the model.

Run the following from the package root to validate the task set without a simulator or model:

```bash
python code/scripts/evaluate.py --dataset dataset --dry-run
python -m pip install -e './code[test]'
python -m pytest -q code/tests/test_dataset.py
```
