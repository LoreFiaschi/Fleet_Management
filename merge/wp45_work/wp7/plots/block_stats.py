def _component_action_label(
    view: dict[str, Any], i: int, l: int, k: int, M: int
) -> tuple[str, str]:
    """Return the mark of cell (i, l) on day index k: the first rule that applies."""
    if view["r"][i, l, k] > 0.5:
        return "P", "replacement"
    if view["m"][i, l, k] > 0.5:
        return "R", "repair"
    assigned = np.flatnonzero(view["x"][i, 1:M + 1, k] > 0.5)
    if assigned.size:
        return f"M{int(assigned[0]) + 1}", "mission"
    if view["x"][i, 0, k] > 0.5:
        return "D", "depot"
    return "I", "idle"


def _draw_statistics(ax, view: dict[str, Any]) -> None:
    data, performance = view["raw"], view["raw"].get("performance") or {}
    status = data.get("status")
    if status is None:
        status_text, status_colour = "unknown", "#374151"
    else:
        status_text = str(status).replace("_", " ")
        status_colour = "#15803d" if status == "optimal" else "#b45309"
    lines = [
        ("Status", status_text, status_colour),
        ("Objective", _number(data.get("objective")), "#111827"),
        ("Dimensions", f"F={view['F']}, M={view['M']}, L={view['L']}", "#111827"),
        ("Horizon", f"H1={view['H1']}, H2={view['H2']}, T={view['T']}", "#111827"),
        ("Models", ", ".join(_unique_values(view["models"].tolist())), "#111827"),
        ("Repair laws", ", ".join(_unique_values(data.get("repair_model"))) or "-",
         "#111827"),
    ]
    if data.get("objective_mode") is not None:
        lines += [("Objective mode", _compact(data.get("objective_mode")), "#111827"),
                  ("J_op / H2", _number(data.get("J_op_average")), "#111827")]
    if data.get("mip_gap") is not None:
        lines.append(("MIP gap", f"{100.0 * float(data['mip_gap']):.2f}%", "#111827"))
    lines.append(("Counts from", view["counts_source"], "#111827"))
    for label, key in (
        ("Variables", "variables"),
        ("Continuous variables", "continuous_variables"),
        ("Integer variables", "integer_variables"),
        ("Linear rows", "linear_rows"),
    ):
        value = view.get(key)
        if value is not None:
            lines.append((label, f"{int(value):,}", "#111827"))
    if performance.get("optimizer_call_seconds") is not None:
        lines.append(("Optimizer", _seconds(float(performance["optimizer_call_seconds"])), "#111827"))
    if performance.get("branch_and_bound_nodes") is not None:
        lines.append(("B&B nodes", f"{float(performance['branch_and_bound_nodes']):,.0f}", "#111827"))

    lines += [
        ("Repairs", str(view["repairs"]), "#111827"),
        ("Replacements", str(view["replacements"]), "#111827"),
    ]

    ax.set_axis_off()
    ax.set_title("Run summary", loc="left", fontweight="bold", pad=12)
    y = 0.96
    for label, value, colour in lines:
        ax.text(0.0, y, label, transform=ax.transAxes, fontsize=8.5,
                color="#6b7280", va="top")
        ax.text(0.70, y, value, transform=ax.transAxes, fontsize=8.5,
                color=colour, va="top", wrap=True,
                fontweight="bold" if label == "Status" else "normal")
        y -= 0.058


def _model_grid(data: dict[str, Any], F: int, L: int) -> np.ndarray:
    """The model name of each cell, from the key ``model``; "unknown" if it is missing."""
    value = data.get("model")
    if value is None:
        return np.full((F, L), "unknown", dtype=object)
    arr = np.asarray(value, dtype=object)
    if arr.shape != (F, L):
        raise ValueError(f"model has shape {arr.shape}; expected {(F, L)}")
    return arr


def _component_names(data: dict[str, Any], L: int) -> list[str]:
    names = data.get("component_names")
    if names is None:
        return [f"Component {l + 1}" for l in range(L)]
    if isinstance(names, str):
        names = [names]
    if len(names) != L:
        raise ValueError(f"component_names has length {len(names)}; expected {L}")
    return [str(name) for name in names]


def _component_row_label(view: dict[str, Any], i: int, l: int) -> str:
    component = view["component_names"][l]
    return f"Train {i + 1} — {component}\n{view['models'][i, l]}"


def _unique_values(value: Any) -> list[str]:
    """The distinct strings of a string or a nested list, in order of first appearance."""
    if value is None:
        return []
    if not isinstance(value, list):
        return [str(value)]
    seen: list[str] = []
    for item in value:
        for text in _unique_values(item):
            if text not in seen:
                seen.append(text)
    return seen
