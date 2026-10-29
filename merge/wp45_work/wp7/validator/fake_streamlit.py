"""A recording stand-in for streamlit, to run the render_* functions without it.

``install(choices)`` puts a module ``streamlit`` in ``sys.modules``. Every call is
recorded in ``calls`` as (name, args summary). ``selectbox`` returns the first option;
``multiselect`` returns its ``default``; ``checkbox`` returns False.
"""
import sys
import types

import pandas as pd

calls = []


class _Box:
    """A context manager that also answers metric(), dataframe(), ... like a container."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        def method(*args, **kwargs):
            calls.append((name, args, kwargs))
        return method


def _tabs(labels):
    calls.append(("tabs", (list(labels),), {}))
    return [_Box() for _ in labels]


def _columns(spec):
    n = spec if isinstance(spec, int) else len(spec)
    return [_Box() for _ in range(n)]


def _selectbox(label, options, *args, **kwargs):
    calls.append(("selectbox", (label, list(options)), kwargs))
    return list(options)[0]


def _multiselect(label, options, default=None, **kwargs):
    calls.append(("multiselect", (label, list(options)), {"default": default}))
    return default


def _checkbox(label, *args, **kwargs):
    calls.append(("checkbox", (label,), kwargs))
    return False


def _generic(name):
    def method(*args, **kwargs):
        if name == "dataframe":
            assert isinstance(args[0], (pd.DataFrame, list)), type(args[0])
        calls.append((name, args, kwargs))
    return method


def install():
    calls.clear()
    st = types.ModuleType("streamlit")
    st.tabs, st.columns = _tabs, _columns
    st.selectbox, st.multiselect, st.checkbox = _selectbox, _multiselect, _checkbox
    for name in ("header", "subheader", "write", "caption", "metric", "dataframe", "pyplot",
                 "success", "error", "info", "warning", "json", "download_button", "text"):
        setattr(st, name, _generic(name))
    sys.modules["streamlit"] = st
    return st
