"""Module for unilateral lymphatic spread, boosted for late T-stage."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from functools import wraps
from typing import Any, Literal

import numpy as np
import pandas as pd

from lymph import diagnosis_times, modalities, models, types, utils

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)


MAP_T_COL = ("_model", "core", "t_stage")
RAW_T_COL_OLD = ("tumor", "1", "t_stage")
RAW_T_COL_NEW = ("tumor", "core", "t_stage")


def select_tstage_model(method):
    """Decorate methods that simply delegate to the `early` or `late` model."""

    @wraps(method)
    def wrapper(self, *args, t_stage: str | None = None, **kwargs):
        if t_stage is None:
            raise ValueError("Must provide t_stage.")

        _model = self.early if t_stage == "early" else self.late
        return method(_model, *args, **kwargs)

    return wrapper


class BoostUnilateral(
    diagnosis_times.Composite,
    modalities.Composite,
    types.Model,
):
    """Class for modelling unilateral metastatic progression.

    Late T-stage is handled as a boost of early spread parameters. This is achieved
    by creating two instances of the :py:class:`~lymph.models.Unilateral` model, one
    for early T-stages and one for late T-stages.

    .. seealso::

        :py:class:`~lymph.models.Unilateral`
            Two instances of this class are created as attributes. One for the early
            and one for the late T-stage model.

    .. warning::

        This class is still a bit experimental and not super thoroughly tested.
    """

    def __init__(
        self,
        graph_dict: types.GraphDictType,
        named_params: Sequence[str] | None = None,
        uni_kwargs: dict[str, Any] | None = None,
        early_kwargs: dict[str, Any] | None = None,
        late_kwargs: dict[str, Any] | None = None,
        **_kwargs,
    ) -> None:
        """Initialize a ``unilateral`` model.

        The ``graph_dict`` is a dictionary of tuples as keys and lists of strings as
        values. It is passed to both :py:class:`.models.Unilateral` instances,
        which in turn pass it to the :py:class:`.graph.Representation` class that
        stores the graph.
        """
        self._init_models(
            graph_dict=graph_dict,
            uni_kwargs=uni_kwargs,
            early_kwargs=early_kwargs,
            late_kwargs=late_kwargs,
        )

        diagnosis_times.Composite.__init__(
            self,
            distribution_children={"early": self.early, "late": self.late},
            is_distribution_leaf=False,
        )
        modalities.Composite.__init__(
            self,
            modality_children={"early": self.early, "late": self.late},
            is_modality_leaf=False,
        )

        if named_params is not None:
            self.named_params = named_params

    def _init_models(
        self,
        graph_dict: dict[tuple[str], list[str]],
        uni_kwargs: dict[str, Any] | None = None,
        early_kwargs: dict[str, Any] | None = None,
        late_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize the two unilateral models."""
        uni_kwargs = uni_kwargs or {}

        _early_kwargs = uni_kwargs.copy()
        _early_kwargs["graph_dict"] = graph_dict
        _early_kwargs.update(early_kwargs or {})

        _late_kwargs = uni_kwargs.copy()
        _late_kwargs["graph_dict"] = graph_dict
        _late_kwargs.update(late_kwargs or {})

        self.early = models.Unilateral(**_early_kwargs)
        self.late = models.Unilateral(**_late_kwargs)

    @classmethod
    def binary(cls, *args, **kwargs) -> BoostUnilateral:
        """Initialize a binary bilateral model.

        This is a convenience method that sets the ``allowed_states`` of the
        ``uni_kwargs`` to ``[0, 1]``. All other ``args`` and ``kwargs`` are
        passed to the :py:meth:`.__init__` method.
        """
        uni_kwargs = kwargs.pop("uni_kwargs", {})
        uni_kwargs["allowed_states"] = [0, 1]
        return cls(*args, uni_kwargs=uni_kwargs, **kwargs)

    @classmethod
    def trinary(cls, *args, **kwargs) -> BoostUnilateral:
        """Initialize a trinary bilateral model.

        This is a convenience method that sets the ``allowed_states`` of the
        ``uni_kwargs`` to ``[0, 1, 2]``. All other ``args`` and ``kwargs`` are
        passed to the :py:meth:`.__init__` method.
        """
        uni_kwargs = kwargs.pop("uni_kwargs", {})
        uni_kwargs["allowed_states"] = [0, 1, 2]
        return cls(*args, uni_kwargs=uni_kwargs, **kwargs)

    @property
    def is_trinary(self) -> bool:
        """Return whether the model is trinary."""
        if self.early.is_trinary != self.late.is_trinary:
            raise ValueError("Both models must be of the same 'narity'.")

        return self.early.is_trinary

    @property
    def is_binary(self) -> bool:
        """Return whether the model is binary."""
        if self.early.is_binary != self.late.is_binary:
            raise ValueError("Both sides must be of the same 'narity'.")

        return self.early.is_binary

    def get_tumor_spread_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> types.ParamsType:
        """Return the parameters of the model's spread from tumor to LNLs."""
        params = {
            "early": self.early.get_tumor_spread_params(as_flat=as_flat),
            "late": self.late.get_tumor_spread_params(as_flat=as_flat),
        }
        if as_flat or not as_dict:
            params = utils.flatten(params)

        return params if as_dict else params.values()

    def get_lnl_spread_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> types.ParamsType:
        """Return the parameters of the model's spread from LNLs to tumor.

        Similarly to the :py:meth:`.get_tumor_spread_params` method.
        """
        params = {
            "early": self.early.get_lnl_spread_params(as_flat=as_flat),
            "late": self.late.get_lnl_spread_params(as_flat=as_flat),
        }
        if as_flat or not as_dict:
            params = utils.flatten(params)

        return params if as_dict else params.values()

    def get_spread_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> types.ParamsType:
        """Return the parameters of the model's spread edges.

        This is consistent with how the :py:meth:`.set_params`
        """
        params = self.get_tumor_spread_params(as_flat=False)
        params.update(self.get_lnl_spread_params(as_flat=as_flat))

        if as_flat or not as_dict:
            params = utils.flatten(params)

        return params if as_dict else params.values()

    def get_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> types.ParamsType:
        """Return the parameters of the model.

        It returns the combination of the call to the :py:meth:`.Unilateral.get_params`
        of the early and late T-stage model. For the use of the ``as_dict`` and
        ``as_flat`` arguments, see the documentation of the
        :py:meth:`.types.Model.get_params` method.

        Also see the :py:meth:`.get_spread_params` method to understand how the
        symmetry settings affect the return value.
        """
        params = self.get_spread_params(as_flat=as_flat)
        params.update(self.get_distribution_params(as_flat=as_flat))

        if as_flat or not as_dict:
            params = utils.flatten(params)

        return params if as_dict else params.values()

    def set_tumor_spread_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Set the parameters of the model's spread from tumor to LNLs."""
        kwargs, global_kwargs = utils.unflatten_and_split(
            kwargs,
            expected_keys=["early", "late"],
        )

        early_kwargs = global_kwargs.copy()
        early_kwargs.update(kwargs.get("early", {}))
        late_kwargs = global_kwargs.copy()
        late_kwargs.update(kwargs.get("late", {}))

        args = self.early.set_tumor_spread_params(*args, **early_kwargs)
        args = self.late.set_tumor_spread_params(*args, **late_kwargs)
        utils.synchronize_params(  # might be redundant check later
            get_from=self.early.graph.lnl_edges,
            set_to=self.late.graph.lnl_edges,
        )

        return args

    def set_lnl_spread_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Set the parameters of the model's spread from LNLs to tumor."""
        kwargs, global_kwargs = utils.unflatten_and_split(
            kwargs,
            expected_keys=["early", "late"],
        )

        early_kwargs = global_kwargs.copy()
        early_kwargs.update(kwargs.get("early", {}))
        late_kwargs = global_kwargs.copy()
        late_kwargs.update(kwargs.get("late", {}))

        args = self.early.set_lnl_spread_params(*args, **early_kwargs)

        return self.late.set_lnl_spread_params(*args, **late_kwargs)

    def set_spread_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Set the parameters of the model's spread edges."""
        args = self.set_tumor_spread_params(*args, **kwargs)
        return self.set_lnl_spread_params(*args, **kwargs)

    def set_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Set new parameters to the model."""
        args = self.set_spread_params(*args, **kwargs)
        return self.set_distribution_params(*args, **kwargs)

    def load_patient_data(
        self,
        patient_data: pd.DataFrame,
        side: str = "ipsi",
        mapping: callable | dict[int, Any] = utils.early_late_mapping,
    ) -> None:
        """Load patient data into the model.

        Amounts to calling the :py:meth:`~lymph.models.Unilateral.load_patient_data`
        method of both the early and late T-stage model.
        """
        t_stage_data = utils.get_item(patient_data, [RAW_T_COL_NEW, RAW_T_COL_OLD])
        is_early = t_stage_data.isin([1, 2])  # noqa: E712
        is_late = t_stage_data.isin([3, 4])  # noqa: E712

        early_patient_data = patient_data.loc[is_early]
        late_patient_data = patient_data.loc[is_late]

        self.early.load_patient_data(
            patient_data=early_patient_data,
            side=side,
            mapping=mapping,
        )
        self.late.load_patient_data(
            patient_data=late_patient_data,
            side=side,
            mapping=mapping,
        )

    def _hmm_likelihood(self, log: bool = True, t_stage: str | None = None) -> float:
        """Compute the HMM likelihood of data, using the stored params."""
        llh = 0.0 if log else 1.0

        early_likelihood = self.early._hmm_likelihood(log=log, t_stage=t_stage)
        late_likelihood = self.late._hmm_likelihood(log=log, t_stage=t_stage)

        if log:
            llh += early_likelihood + late_likelihood
        else:
            llh *= early_likelihood * late_likelihood
        return llh

    def _bn_likelihood(self, log: bool = True, t_stage: str | None = None) -> float:
        """Compute the BN likelihood of data, using the stored params."""
        llh = 0.0 if log else 1.0

        early_likelihood = self.early._bn_likelihood(log=log, t_stage=t_stage)
        late_likelihood = self.late._bn_likelihood(log=log, t_stage=t_stage)

        if log:
            llh += early_likelihood + late_likelihood
        else:
            llh *= early_likelihood * late_likelihood
        return llh

    def likelihood(
        self,
        given_params: types.ParamsType | None = None,
        log: bool = True,
        t_stage: str | None = None,
        mode: Literal["HMM", "BN"] = "HMM",
    ):
        """Compute the (log-)likelihood of the stored data given the model (and params).

        See the documentation of :py:meth:`.types.Model.likelihood` for more
        information on how to use the ``given_params`` parameter.

        Returns the log-likelihood if ``log`` is set to ``True``. The ``mode`` parameter
        determines whether the likelihood is computed for the hidden Markov model
        (``"HMM"``) or the Bayesian network (``"BN"``).

        .. note::

            The computation is much faster if no parameters are given, since then the
            transition matrix does not need to be recomputed.

        .. seealso::

            :py:meth:`.Unilateral.likelihood`
                The corresponding unilateral function.

        """
        try:
            # all functions and methods called here should raise a ValueError if the
            # given parameters are invalid...
            utils.safe_set_params(self, given_params)
        except ValueError:
            return -np.inf if log else 0.0

        if mode == "HMM":
            return self._hmm_likelihood(log, t_stage)
        if mode == "BN":
            return self._bn_likelihood(log, t_stage)

        raise ValueError("Invalid mode. Must be either 'HMM' or 'BN'.")

    @select_tstage_model
    def state_dist(self, model: models.Unilateral, *args, **kwargs) -> np.ndarray:
        """Compute the distribution over possible states.

        See :py:meth:`.models.Unilateral.state_dist` for more information.
        """
        return model.state_dist(*args, **kwargs)

    @select_tstage_model
    def posterior_state_dist(
        self,
        model: models.Unilateral,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """Compute the posterior distribution over hidden states given a diagnosis.

        See :py:meth:`.models.Unilateral.posterior_state_dist` for more information.
        """
        return model.posterior_state_dist(*args, **kwargs)

    @select_tstage_model
    def marginalize(self, model: models.Unilateral, *args, **kwargs) -> np.ndarray:
        """Marginalize ``given_state_dist`` over matching ``involvement`` patterns.

        See :py:meth:`.models.Unilateral.marginalize` for more information.
        """
        return model.marginalize(*args, **kwargs)

    def risk(
        self,
        involvement: types.PatternType,
        given_params: types.ParamsType | None = None,
        given_state_dist: np.ndarray | None = None,
        given_diagnosis: dict[str, types.PatternType] | None = None,
        t_stage: str = "early",
        mode: Literal["HMM", "BN"] = "HMM",
    ) -> float:
        """Compute risk of a certain ``involvement``, using the ``given_diagnosis``.

        See :py:meth:`.models.Unilateral.risk` for more information.
        """
        if t_stage == "early":
            return self.early.risk(
                involvement=involvement,
                given_params=given_params,
                given_state_dist=given_state_dist,
                given_diagnosis=given_diagnosis,
                t_stage=t_stage,
                mode=mode,
            )
        if t_stage == "late":
            return self.late.risk(
                involvement=involvement,
                given_params=given_params,
                given_state_dist=given_state_dist,
                given_diagnosis=given_diagnosis,
                t_stage=t_stage,
                mode=mode,
            )
        raise ValueError("T-stage needs to be early or late.")
