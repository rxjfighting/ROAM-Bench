# ROAM-Bench code

This directory contains the simulator backend, environment builders, evaluation harness, model interface, and CPU tests. Run these commands from this directory. The task set is in `../dataset` and scene sources are in `../scenes`.

## Installation

Use Python 3.11 and NVIDIA Isaac Sim 5.1.0 for simulation. Install this source checkout into the Python environment used by Isaac Sim:

```bash
python -m pip install -e '.[geometry,media,test]'
python -m roam_bench.cli --dataset ../dataset --dry-run
python -m pytest -q
```

The dry run and tests do not start Isaac Sim or call a model. Simulation additionally requires a compatible NVIDIA GPU and the Isaac Sim Python runtime. Build the provided Dockerfile from the package root with `docker build -f code/Dockerfile -t roam-bench .`. The container requires NVIDIA GPU support and mounted assets. Review the NVIDIA image license before use.

## Download scene sources

Scene sources are distributed separately in the [Hugging Face environment repository](https://huggingface.co/datasets/xjran/ROAM-Bench-scenes). Run from `code/`:

```bash
python -m pip install -e '.[hub]'
python scripts/download_scenes.py --repo-id xjran/ROAM-Bench-scenes --scene scene_148 --output ../scenes
```

Use `--all` for all scenes. Local archives can be read with `--source-dir /path/to/environment-repository`. Downloads are checked against `scene_archives.json` before safe extraction; existing scenes are not overwritten. See the root README for storage requirements.

## Environment construction

The scene sources retain the instance identifiers required by the task set. Each scene contains `combined_house/` directives and `house_state.json`, referenced SDF/mesh/texture resources, and room layout files. Obtain official robot assets separately. See [assets and construction](docs/environment.md).

```bash
python scripts/build_robot_asset.py --source official_G1 --output ../robot
python scripts/build_scene_bundle.py --output ../scenes --verify-only
python scripts/build_environment.py --dataset ../dataset --assets ../scenes --scene scene_148 --inspect
python scripts/build_environment.py --dataset ../dataset --assets ../scenes --scene scene_148 --robot-asset ../robot/g1_inspire.usda --output ../stages/scene_148.usda
```

The scene exporter can also collect dependencies from another compatible source tree with `--source-root SOURCE --output OUTPUT`. It defaults to all 27 scenes listed in the task manifest. The robot builder creates derived layers and an Inspire-hand entry without editing the supplied official assets. Run USD construction with the Isaac Sim Python environment.

## Evaluation

Use a vision-capable endpoint implementing the Chat Completions interface. Set credentials in your shell; no credentials are included in this package.

```bash
export ROAM_BENCH_API_KEY=YOUR_API_KEY
python scripts/evaluate.py --dataset ../dataset --assets ../scenes --robot-asset ../robot/g1_inspire.usda --model YOUR_MODEL --base-url https://api.openai.com/v1 --output ../results/model.json
python scripts/summarize.py ../results/model.json
```

Filter with `--scene`, `--task-id`, `--group`, or `--limit`. Explicitly pass `--dataset` when installing the Python package outside this checkout: the dataset is distributed alongside the source, rather than embedded in a wheel. The evaluation defaults to temperature 0, top-p 1, and maximum reasoning effort. The client sends `max` for reasoning effort by default; if the endpoint uses another value for its highest reasoning level, supply that value with `--reasoning-effort`. Model names and endpoint choices remain explicit.

Task-specific scene variants are selected automatically. `--scene-cache` sets their cache directory, which defaults to `.roam-cache/scenes`. Variants reuse the supplied assets and write separate scene directives and object metadata; shared scenes are never edited. Running another task in the same scene selects its own full scene or variant.

Results are saved after each completed task. `--resume` skips completed tasks only when the code, task set, reference images, model, and settings match. Per-round checkpoints retain interrupted episode state; resume restores this state without repeating completed decision rounds. Checkpoints use JSON with explicit encoded state types.

```bash
python scripts/evaluate.py --dataset ../dataset --assets ../scenes --robot-asset ../robot/g1_inspire.usda --model YOUR_MODEL --base-url https://api.openai.com/v1 --output ../results/model.json --resume
```

The main protocol supplies RGB, self-odometry, and action/hand/contact feedback. Scene semantics and scoring truth remain inside the evaluator. It does not supply depth images to the main benchmark model. See [protocol details](docs/protocol.md) for budgets, visual memory, retrieval, retries, and metric definitions.

## License

Code and task annotations use the [MIT license](../LICENSE). See [third-party notices](../THIRD_PARTY_NOTICES.md) for scene-asset provenance and external runtime licenses.
