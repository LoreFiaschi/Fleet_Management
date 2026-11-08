"""
Degradation models: the shared model ``base`` and one cell builder per model.

The package imports no module, so that it loads without Gurobi; ``base`` imports
the builder of a model by its name when a fleet needs it.
"""
