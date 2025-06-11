"""Getting and setting parameters in a nested model structure."""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol


class HasGetParams(Protocol):
    """Protocol for objects that can return their parameters as a dictionary."""

    @abstractmethod
    def get_params(self) -> dict[str, float]:
        """Get the parameters of the model as a dictionary."""
        raise NotImplementedError


class HasSetParams(Protocol):
    """Protocol for objects that can set their parameters from a dictionary."""

    @abstractmethod
    def set_params(self, *args: float, **kwargs: dict[str, float]) -> None:
        """Set the parameters of the model from a dictionary."""
        raise NotImplementedError


class HasGetAndSetParams(HasGetParams, HasSetParams):
    """Protocol for objects that can get and set their parameters."""


class ParamsManager:
    """Allow getting and setting params of nested models.

    In nested models, the parameter names are constructed by joining the names of the
    parent models with the parameter name, separated by an underscore (e.g.,
    ``parent_child_paramname``).

    Additionally, this class allows the distribution to multiple children, if their
    name is omitted in the parameter name. For example, if the parameter name is
    ``parent_paramname``, it should be distributed to all children of the parent model
    that have a parameter with the name ``paramname``.

    >>> class Leaf(HasGetAndSetParams):
    ...     param: float = 1.0
    ...     def get_params(self) -> dict[str, float]:
    ...         return {'foo': self.param}
    ...     def set_params(self, *args: float, **kwargs: dict[str, float]) -> None:
    ...         self.param = kwargs.get('foo', self.param)
    >>> middle = {
    ...     "middle1": ParamsManager({"bar": Leaf(), "qux": Leaf()}),
    ...     "middle2": ParamsManager({"bar": Leaf(), "qux": Leaf()}),
    ... }
    >>> root = ParamsManager(middle)
    >>> root.get_params()   # doctest: +NORMALIZE_WHITESPACE
    {'middle1_bar_foo': 1.0,
     'middle1_qux_foo': 1.0,
     'middle2_bar_foo': 1.0,
     'middle2_qux_foo': 1.0}
    >>> root.set_params(bar_foo=2.0)
    >>> root.get_params()   # doctest: +NORMALIZE_WHITESPACE
    {'middle1_bar_foo': 2.0,
     'middle1_qux_foo': 1.0,
     'middle2_bar_foo': 2.0,
     'middle2_qux_foo': 1.0}
    >>> root.set_params(middle2_foo=3.0)
    >>> root.get_params()   # doctest: +NORMALIZE_WHITESPACE
    {'middle1_bar_foo': 2.0,
     'middle1_qux_foo': 1.0,
     'middle2_bar_foo': 3.0,
     'middle2_qux_foo': 3.0}
    """

    def __init__(
        self,
        params_children: dict[str, HasGetAndSetParams | ParamsManager],
    ) -> None:
        """Define the children of the model."""
        if len(params_children) == 0:
            raise ValueError("ParamsManager must have at least one child.")

        self._params_children = params_children

    def get_params(self) -> dict[str, float]:
        """Get the parameters of the model as a dictionary."""
        params = {}

        for child_name, child in self._params_children.items():
            child_params = child.get_params()
            for param_name, param_value in child_params.items():
                full_param_name = f"{child_name}_{param_name}"
                params[full_param_name] = param_value

        return params

    def set_params(self, *args: float, **kwargs: dict[str, float]) -> None:
        """Set the parameters of the model from a dictionary."""
        new_params = dict(zip(self.get_params().keys(), args, strict=False))
        new_params.update(kwargs)

        for param_name, param_value in new_params.items():
            start, _, rest = param_name.partition("_")

            if start in self._params_children:
                child = self._params_children[start]
                child.set_params(**{rest: param_value})
            else:
                for child in self._params_children.values():
                    child.set_params(**{param_name: param_value})
