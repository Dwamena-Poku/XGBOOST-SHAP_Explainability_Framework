from __future__ import annotations
import ast, json, re, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ANCHOR = 'REM-ALBERT_DWAMENA_POKU-2026-08-26'
SEED = 42


def root_from_script(file: str) -> Path:
    return Path(file).resolve().parent


def dirs(root: Path):
    d = {
        'root': root,
        'artifacts': root/'artifacts',
        'results': root/'results',
        'figures': root/'figures',
        'logs': root/'logs',
        'data': root/'data',
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def sha256(path: Path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()


def require_no_fillins(mapping, context='configuration'):
    bad=[]
    def walk(x, key=''):
        if isinstance(x, dict):
            for k,v in x.items(): walk(v, f'{key}.{k}' if key else str(k))
        elif isinstance(x, list):
            for i,v in enumerate(x): walk(v, f'{key}[{i}]')
        elif isinstance(x, str) and '<<< FILL IN >>>' in x:
            bad.append(key)
    walk(mapping)
    if bad:
        raise RuntimeError(f'{context} contains unresolved <<< FILL IN >>> fields: {bad}. Refusing to guess.')


def parse_score_list(cell):
    """Parse plain floats or legacy np.float64(...) strings. Never substitutes values."""
    if isinstance(cell, (list, tuple, np.ndarray)):
        return [float(x) for x in cell]
    if cell is None or (isinstance(cell,float) and np.isnan(cell)):
        raise ValueError('empty score list')
    s=str(cell).strip()
    # tolerate legacy numpy scalar representation only by stripping wrapper
    s=re.sub(r'np\.float(?:32|64)\(([-+0-9.eE]+)\)', r'\1', s)
    try:
        val=ast.literal_eval(s)
    except Exception as e:
        raise ValueError(f'unparseable score list: {cell!r}') from e
    if not isinstance(val,(list,tuple)):
        raise ValueError(f'expected list/tuple, got {type(val).__name__}')
    return [float(x) for x in val]


def resolve_data_path(root: Path, value: str) -> Path:
    p=Path(value)
    if not p.is_absolute(): p=(root/p).resolve()
    return p

def load_table(path: Path):
    path=Path(path)
    if path.suffix.lower() in ('.xlsx','.xls'):
        return pd.read_excel(path)
    if path.suffix.lower()=='.csv':
        # auto-detect delimiter, including semicolon-delimited UCI exports
        return pd.read_csv(path, sep=None, engine='python')
    raise ValueError(f'Unsupported dataset type: {path.suffix}')
