from pathlib import Path
import copy
import yaml

input_path = Path("input/data.yaml")
output_path = Path("input/data_test.yaml")

with input_path.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

F = data["F"]

# These are 4D parameters expected to have shape:
# F x M x L x H
params_to_expand = ["mu", "v"]

for name in params_to_expand:
    if name not in data:
        print(f"Skipping {name}: not found")
        continue

    value = data[name]

    print(f"{name}: original top-level length = {len(value)}")

    if len(value) == 1 and F > 1:
        data[name] = [copy.deepcopy(value[0]) for _ in range(F)]
        print(f"{name}: expanded to top-level length = {len(data[name])}")
    elif len(value) == F:
        print(f"{name}: already matches F={F}")
    else:
        raise ValueError(
            f"{name} has top-level length {len(value)}, but F={F}. "
            f"Cannot safely patch automatically."
        )

with output_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, sort_keys=False)

print(f"\nWrote patched test file to {output_path}")