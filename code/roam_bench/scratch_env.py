"""Allocate a process-local temporary directory and configurable simulation caches."""
import atexit
import os
from pathlib import Path
import shutil
import tempfile
from .paths import CACHE_ROOT

_RUN_DIR = None
_REPORT = []

def cleanup():
    global _RUN_DIR
    if _RUN_DIR is not None:
        shutil.rmtree(_RUN_DIR, ignore_errors=True)
        _RUN_DIR = None

def redirect():
    global _RUN_DIR, _REPORT
    targets = {'OMNI_CACHE_DIR': CACHE_ROOT / 'omni', 'XDG_CACHE_HOME': CACHE_ROOT / 'xdg', 'CUDA_CACHE_PATH': CACHE_ROOT / 'cuda'}
    _REPORT = []
    for name, default in targets.items():
        existing = name in os.environ
        value = Path(os.environ.setdefault(name, str(default))).expanduser()
        value.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(value)
        _REPORT.append({'var': name, 'effective': str(value), 'source': 'preexisting' if existing else 'default'})
    if _RUN_DIR is None:
        root = Path(os.environ.get('TMPDIR', CACHE_ROOT / 'tmp')).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        _RUN_DIR = tempfile.mkdtemp(prefix='roam_run_', dir=root)
        atexit.register(cleanup)
    os.environ['TMPDIR'] = _RUN_DIR
    return {name: os.environ[name] for name in (*targets, 'TMPDIR')}

def cache_redirect_report():
    return list(_REPORT)
