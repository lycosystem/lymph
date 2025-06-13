"""Getting and setting parameters in a nested model structure."""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol

from lymph.utils import zip_with_remainder


class SupportsParams(Protocol):
    """Protocol for objects that can get and set params via private methods."""

    @abstractmethod
    def _get_params(self) -> dict[str, float]:
        """Get the parameters of the model as a dictionary."""
        raise NotImplementedError

    @abstractmethod
    def _set_params(self, *args: float, **kwargs: dict[str, float]) -> None:
        """Set the parameters of the model from a dictionary."""
        raise NotImplementedError


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

    # Note: We use name mangling in this class, to avoid clashes with
    # attributes of the `DistributionManager` and `ModalityManager` mixins.

    __children: dict[str, SupportsParams]

    def __init__(self, child_attrs: list[str]) -> None:
        """Define the children of the model."""
        if len(child_attrs) == 0:
            raise ValueError("ParamsManager must have at least one child.")

        self.__children = {attr: getattr(self, attr) for attr in child_attrs}

    def _get_params(self) -> dict[str, float]:
        """Get the parameters of the model as a dictionary."""
        if len(self.__children) == 1:
            return next(iter(self.__children.values()))._get_params()

        params = {}
        for child_name, child in self.__children.items():
            child_params = child._get_params()
            for param_name, param_value in child_params.items():
                full_param_name = f"{child_name}_{param_name}"
                params[full_param_name] = param_value

        return params

    def _set_params(
        self,
        *args: float,
        **kwargs: dict[str, float],
    ) -> tuple[float, ...]:
        """Set the parameters of the model."""
        if len(self.__children) == 1:
            # If only one child, no need for using a prefix, just pass down directly.
            child = next(iter(self.__children.values()))
            return child._set_params(*args, **kwargs)

        zipped_names_and_vals, args = zip_with_remainder(self._get_params(), args)
        new_params = dict(zipped_names_and_vals)
        new_params.update(kwargs)

        for param_name, param_value in new_params.items():
            start, _, rest = param_name.partition("_")

            if start in self.__children:
                child = self.__children[start]
                child._set_params(**{rest: param_value})
            else:
                for child in self.__children.values():
                    child._set_params(**{param_name: param_value})

        return args
