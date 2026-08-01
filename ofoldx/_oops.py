r"""
Guarded access to optional OOps operators.

A named operator that the installed OOps package does not provide, or any operator when OOps is not installed, is a
``MissingOopsKernel``: backend resolution never selects it for ``"auto"`` and refuses it for an explicit ``"oops"``,
so public PyTorch paths remain available.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Callable, NoReturn

from .errors import RequestRef, fail

_oops: ModuleType | None

try:
    _oops = import_module("oops")
except ModuleNotFoundError as error:
    if error.name != "oops":
        raise
    _oops = None


class MissingOopsKernel:
    r"""
    The stand-in for an OOps operator this installation does not provide; calling it raises
    ``oops_operator_unavailable``.

    Args:
        name: Attribute name of the operator on the ``oops`` package.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *_args: object, **_kwargs: object) -> NoReturn:
        fail("oops_operator_unavailable", RequestRef("backend", "oops"), operator=self.name)


def oops_kernel(name: str) -> Callable[..., object]:
    r"""
    Return a named OOps kernel, or its ``MissingOopsKernel`` when the installed package (if any) does not provide it.

    Args:
        name: Attribute name of the public callable on the ``oops`` package (e.g. ``"transition"``).

    Returns:
        The OOps-owned implementation callable, or the stand-in for the missing operator.
    """

    kernel = None if _oops is None else getattr(_oops, name, None)
    return kernel if callable(kernel) else MissingOopsKernel(name)
