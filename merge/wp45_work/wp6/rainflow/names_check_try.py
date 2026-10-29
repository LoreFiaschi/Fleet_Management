"""Report global names that a function of base.py uses but nobody defines.

Usage: python names_check.py [path-to-base.py]
Prints one line per undefined name with the function that uses it.
Exit code 0 always; read the output. Names used only inside the Gamma block
(``_gamma_reachable_upper_bounds``, ``GammaCellBuilder``) are listed apart,
because WP5 leaves that block alone.
"""
import ast
import builtins
import pathlib
import symtable
import sys

DEFAULT = pathlib.Path(
    r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge"
    r"\src\fleet_management\degradation_model\base.py"
)
GAMMA_SCOPES = {"_gamma_reachable_upper_bounds", "GammaCellBuilder", "_add_binary_scaled_product"}


def module_names(tree):
    names = set(dir(builtins))
    top = list(tree.body)
    for node in tree.body:   # also a guarded import: try / except at module level
        if isinstance(node, ast.Try):
            top += node.body + [s for h in node.handlers for s in h.body]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node in tree.body:
                names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and node in top:
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
    for node in top:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
    return names


def walk(table, path, defined, out):
    for child in table.get_children():
        child_path = path + [child.get_name()]
        if child.get_type() == "function":
            for symbol in child.get_symbols():
                if symbol.is_global() and symbol.is_referenced():
                    if symbol.get_name() not in defined:
                        out.append((".".join(child_path), symbol.get_name()))
        walk(child, child_path, defined, out)


def main():
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    defined = module_names(tree)
    table = symtable.symtable(source, str(path), "exec")
    found = []
    walk(table, [], defined, found)
    gamma = [item for item in found if item[0].split(".")[0] in GAMMA_SCOPES]
    other = [item for item in found if item[0].split(".")[0] not in GAMMA_SCOPES]
    print(f"undefined names outside the Gamma block: {len(other)}")
    for scope, name in other:
        print(f"  {scope}: {name}")
    print(f"undefined names inside the Gamma block (expected after WP5): {len(gamma)}")
    for scope, name in gamma:
        print(f"  {scope}: {name}")


if __name__ == "__main__":
    main()
