"""Pytest plugin: inject a minimal fake numpy so coverage works in free-threaded
CPython (real numpy C-extension cannot load under coverage tracing)."""
import math
import sys
import types
import builtins

_bsum = builtins.sum
_bmax = builtins.max


class ndarray(list):
    def __init__(self, data):
        super().__init__(data)

    def _is2d(self):
        return bool(self) and isinstance(self[0], (list, ndarray))

    def __pow__(self, e):
        if self._is2d():
            return ndarray([row**e for row in self])
        return ndarray([x**e for x in self])

    def __add__(self, other):
        if self._is2d() and isinstance(other, ndarray) and len(other) != len(self):
            return ndarray([row + other for row in self])
        return ndarray([a + b for a, b in zip(self, other)])

    def __sub__(self, other):
        if self._is2d() and isinstance(other, ndarray) and len(other) != len(self):
            return ndarray([row - other for row in self])
        return ndarray([a - b for a, b in zip(self, other)])

    def __mul__(self, other):
        if self._is2d() and isinstance(other, ndarray) and len(other) != len(self):
            return ndarray([row * other for row in self])
        return ndarray([a * b for a, b in zip(self, other)])

    def sum(self, axis=None):
        if axis is None:
            if self._is2d():
                return _bsum(x for row in self for x in row)
            return _bsum(self)
        if axis == 1:
            return ndarray([_bsum(row) for row in self])
        if axis == 0:
            n = len(self)
            return ndarray([_bsum(self[i][j] for i in range(n)) for j in range(len(self[0]))])

    def mean(self, axis=None):
        if axis is None:
            if self._is2d():
                return _bsum(x for row in self for x in row) / _bsum(len(row) for row in self)
            return _bsum(self) / len(self)
        if axis == 1:
            return ndarray([_bsum(row) / len(row) for row in self])
        if axis == 0:
            n = len(self)
            return ndarray([_bsum(self[i][j] for i in range(n)) / n for j in range(len(self[0]))])

    def max(self):
        if self._is2d():
            return _bmax(x for row in self for x in row)
        return _bmax(self)

    @property
    def shape(self):
        if self._is2d():
            return (len(self), len(self[0]))
        return (len(self),)


def array(data):
    if isinstance(data, ndarray):
        return data
    return ndarray(list(data))


def sqrt(x):
    return math.sqrt(x)


def mean(x):
    return x.mean() if isinstance(x, ndarray) else (_bsum(x) / len(x) if x else None)


def max(x):
    return x.max() if isinstance(x, ndarray) else (_bmax(x) if x else None)


def sum(x, axis=None):
    if isinstance(x, ndarray):
        return x.sum(axis)
    return _bsum(x) if axis is None else None


def isscalar(x):
    return not isinstance(x, (list, ndarray, tuple))


def _build():
    mod = types.ModuleType("numpy")
    mod.ndarray = ndarray
    mod.array = array
    mod.sqrt = sqrt
    mod.mean = mean
    mod.max = max
    mod.sum = sum
    mod.isscalar = isscalar
    mod.bool_ = bool
    return mod


def pytest_configure(config):
    if "numpy" not in sys.modules:
        sys.modules["numpy"] = _build()