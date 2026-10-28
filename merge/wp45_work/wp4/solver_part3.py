# ---------------------------------------------------------------------------
# Result files: HDF5
# ---------------------------------------------------------------------------
def _save_hdf5(result: dict, path: Path) -> None:
    """Write the flat HDF5 layout of the spec: numbers and strings as attributes,
    arrays of numbers as float64 datasets, everything else as JSON text."""
    with h5py.File(path, "w") as f:
        for key in RESULT_KEYS:
            value = result.get(key)
            if value is None:
                continue  # a key without a value is not stored
            if key in _HDF5_DATASET_KEYS:
                f.create_dataset(key, data=np.asarray(value, dtype=np.float64))
            elif key in _HDF5_JSON_KEYS or (
                key in _SELECTOR_KEYS and not isinstance(value, str)
            ):
                f.attrs[key] = json.dumps(_to_plain(value), allow_nan=False)
            elif key == "H":
                f.attrs[key] = np.asarray(value, dtype=np.int64)
            else:
                f.attrs[key] = value.item() if isinstance(value, np.generic) else value
