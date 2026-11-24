"""Resolve portable asset, output, and cache locations."""
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT
OUTPUT_ROOT = Path(os.environ.get('ROAM_BENCH_OUTPUT_ROOT', PROJECT_ROOT / 'results')).expanduser()
CACHE_ROOT = Path(os.environ.get('ROAM_BENCH_CACHE_ROOT', PROJECT_ROOT / '.cache')).expanduser()
DATA_ROOT = Path(os.environ.get('ROAM_BENCH_DATA_ROOT', PROJECT_ROOT / 'assets')).expanduser()
SCENESMITH_ROOT = Path(os.environ.get('SCENESMITH_ROOT', DATA_ROOT)).expanduser()
DEFAULT_SCENE = Path(os.environ.get('ROAM_BENCH_SCENE', DATA_ROOT / 'scene_100')).expanduser()
G1_EXPERIMENT_ROOT = CACHE_ROOT / 'robot'
