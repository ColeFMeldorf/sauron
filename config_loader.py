import yaml
from pathlib import Path


def deep_merge(base, override):
    """Return a new dict: `base` with `override` laid on top of it.

    - Nested dicts are merged key by key.
    - Anything else (numbers, strings, lists) in `override` replaces the base value outright.
    - A value of null (None) in `override` DELETES that key from the result.
    """
    merged = dict(base)  # copy so we don't modify the parent in place
    for key, val in override.items():
        if val is None:
            merged.pop(key, None)
        elif isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_config(path, _seen=None):
    """Load a YAML config, resolving PARENT_CONFIG (a path or list of paths) recursively."""
    path = Path(path).expanduser().resolve()

    # Guard against A -> B -> A loops, which would otherwise recurse forever.
    _seen = set() if _seen is None else _seen
    if path in _seen:
        raise ValueError(f"Circular PARENT_CONFIG chain involving {path}")
    _seen = _seen | {path}

    with open(path, "r") as f:
        config = yaml.safe_load(f) or {}

    parent = config.pop("PARENT_CONFIG", None)  # pop so it never looks like a "survey"
    if parent is None:
        return config

    parents = [parent] if isinstance(parent, str) else parent
    base = {}
    for p in parents:  # later parents override earlier ones
        p = Path(p).expanduser()
        if not p.is_absolute():
            p = path.parent / p  # relative paths are relative to the child file
        base = deep_merge(base, load_config(p, _seen))

    return deep_merge(base, config)