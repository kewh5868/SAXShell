from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dream import SAXSDreamResultsLoader, SAXSDreamWorkflow
    from .prefit import SAXSPrefitWorkflow

__all__ = [
    "SAXSDreamResultsLoader",
    "SAXSDreamWorkflow",
    "SAXSPrefitWorkflow",
]

_LAZY_EXPORTS = {
    "SAXSDreamResultsLoader": "saxshell.saxs.dream",
    "SAXSDreamWorkflow": "saxshell.saxs.dream",
    "SAXSPrefitWorkflow": "saxshell.saxs.prefit",
}


def __getattr__(name: str) -> object:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
