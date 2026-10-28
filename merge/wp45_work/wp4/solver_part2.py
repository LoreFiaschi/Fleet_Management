# ---------------------------------------------------------------------------
# Result files: YAML and JSON
# ---------------------------------------------------------------------------
def _save_results(result: dict, path) -> None:
    """Save solver results to a file (YAML, JSON, or HDF5), chosen by the extension."""
    path = Path(path)
    file_format = _result_format(path)

    if file_format == "yaml":
        _save_yaml(result, path)
    elif file_format == "json":
        _save_json(result, path)
    else:
        _save_hdf5(result, path)


def _to_plain(value):
    """Convert NumPy values and nested containers to plain data; NaN and infinity become None."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_plain(value.tolist())
    if isinstance(value, np.generic):
        return _to_plain(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (bool, int, str)):
        return value
    raise TypeError(
        f"cannot write a value of type {type(value).__name__} to a result file."
    )


def _text_output(result: dict) -> dict:
    """Build the mapping of a YAML or JSON result file: the keys of RESULT_KEYS in order."""
    output = {}
    for key in RESULT_KEYS:
        value = result.get(key)
        if value is None and key != "objective":
            continue
        output[key] = _to_plain(value)
    return output


def _save_yaml(result: dict, path: Path) -> None:
    output = _text_output(result)
    with open(path, "w", encoding="utf-8") as f:
        # safe_dump emits only plain YAML (no !!python/object tags), so the file
        # round-trips through yaml.safe_load; it also fails loudly if any NumPy
        # object slipped through _to_plain.
        yaml.safe_dump(
            output, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )


def _save_json(result: dict, path: Path) -> None:
    output = _text_output(result)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, allow_nan=False)
