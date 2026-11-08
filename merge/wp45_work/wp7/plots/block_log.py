def _read_presolve_log(input_file: Path) -> dict[str, int]:
    """Presolved counts from the Gurobi log ``runner.out`` beside the result file.

    Empty if there is no such file or it has no presolve lines (spec, "Summary panel").
    """
    log_path = input_file.parent / "runner.out"
    if not log_path.is_file():
        return {}
    return _parse_gurobi_presolve(
        log_path.read_text(encoding="utf-8", errors="replace")
    )
