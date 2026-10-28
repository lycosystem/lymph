"""Module for unilateral lymphatic spread, boosted for late T-stage."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from lymph import diagnosis_times, modalities, models, types, utils

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)


MAP_T_COL = ("_model", "core", "t_stage")
RAW_T_COL_OLD = ("tumor", "1", "t_stage")
RAW_T_COL_NEW = ("tumor", "core", "t_stage")


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
        lnl_method: Literal["logit_ratio", "logit_shift"],
        tumor_method: Literal["logit_ratio", "logit_shift"],
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

        # set the methods of boosting, this is now mainly for testing different methods
        self.lnl_method = lnl_method
        self.tumor_method = tumor_method

        # set the parameters needed for boosting and transition
        # I think at some point I'll also need to add some setter
        # functions etc. to make things correct
        self.lnl_boost = 0.5
        self.tumor_boost = 0.5

        self.advanced_prob = 0.5

        diagnosis_times.Composite.__init__(
            self,
            distribution_children={"base": self.early, "boosted": self.late},
            is_distribution_leaf=False,
        )
        modalities.Composite.__init__(
            self,
            modality_children={"base": self.early, "boosted": self.late},
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

    @property
    def advanced_prob(self) -> float:
        """Return the probability of advanced T-stage."""
        if hasattr(self, "_advanced_prob"):
            return self._advanced_prob
        return 0.0

    @advanced_prob.setter
    def advanced_prob(self, value: float) -> None:
        """Set the probability of advanced T-stage."""
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("The advanced T-stage prob must be in the range [0, 1].")
        self._advanced_prob = value

    @property
    def lnl_boost(self) -> float:
        """Return the LNL boost."""
        if hasattr(self, "_lnl_boost"):
            return self._lnl_boost
        return 0.0

    @lnl_boost.setter
    def lnl_boost(self, value: float) -> None:
        """Set the LNL boost."""
        if self.lnl_method == "logit_shift":
            if value is not None and not -5.0 <= value <= 5.0:
                raise ValueError("The lnl boost must be in the range [-5, 5].")
        if self.lnl_method == "logit_ratio":
            if value is not None and not 0 <= value <= 10.0:
                raise ValueError("The lnl boost must be in the range [0, 10].")
        self._lnl_boost = value

    @property
    def tumor_boost(self) -> float:
        """Return the tumor boost."""
        if hasattr(self, "_tumor_boost"):
            return self._tumor_boost
        return 0.0

    @tumor_boost.setter
    def tumor_boost(self, value: float) -> None:
        """Set the tumor boost."""
        if self.tumor_method == "logit_shift":
            if value is not None and not -5.0 <= value <= 5.0:
                raise ValueError("The tumor boost must be in the range [-5, 5].")
        if self.tumor_method == "logit_ratio":
            if value is not None and not 0 <= value <= 10.0:
                raise ValueError("The tumor boost must be in the range [0, 10].")
        self._tumor_boost = value

    @staticmethod
    def boost_late(
        model_param: float,
        boost_param: float,
        method: Literal["logit_ratio", "logit_shift"],
    ) -> float:
        """Compute boosted parameter according to selected method."""
        param_logit = np.log(model_param / (1 - model_param))

        if method == "logit_ratio":
            exp_term = np.exp(boost_param * param_logit)
            return exp_term / (1 + exp_term)
        if method == "logit_shift":
            exp_term = np.exp(boost_param + param_logit)
            return exp_term / (1 + exp_term)

        raise ValueError("Invalid mode. Must be either 'logit_ratio' or 'logit_shift'.")

    def get_tumor_spread_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> types.ParamsType:
        """Return the parameters of the model's spread from tumor to LNLs."""
        # do we even need to return the late parameters? probably not, but might be nice
        # to also see them directly?
        params = self.early.get_tumor_spread_params(as_flat=as_flat)
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
        params = self.early.get_lnl_spread_params(as_flat=as_flat)
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
        params = {}

        params["lnl_boost"] = self.lnl_boost
        params["tumor_boost"] = self.tumor_boost
        params["advanced_prob"] = self.advanced_prob

        params.update(self.get_spread_params(as_flat=as_flat))
        params.update(self.get_distribution_params(as_flat=as_flat))

        if as_flat or not as_dict:
            params = utils.flatten(params)

        return params if as_dict else params.values()

    def set_tumor_spread_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Set the parameters of the model's spread from tumor to LNLs."""
        args = self.early.set_tumor_spread_params(*args, **kwargs)

        late_kwargs = {}
        for key, early_param in self.early.get_tumor_spread_params().items():
            late_kwargs[key] = self.boost_late(
                early_param,
                self.tumor_boost,
                self.tumor_method,
            )
        self.late.set_tumor_spread_params(**late_kwargs)

        return args

    def set_lnl_spread_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Set the parameters of the model's spread from LNLs to tumor."""
        args = self.early.set_lnl_spread_params(*args, **kwargs)

        late_kwargs = {}
        for key, early_param in self.early.get_lnl_spread_params().items():
            late_kwargs[key] = self.boost_late(
                early_param,
                self.lnl_boost,
                self.lnl_method,
            )
        self.late.set_lnl_spread_params(**late_kwargs)
        return args

    def set_spread_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Set the parameters of the model's spread edges."""
        args = self.set_tumor_spread_params(*args, **kwargs)
        return self.set_lnl_spread_params(*args, **kwargs)

    def set_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Set new parameters to the model."""
        lnl_boost, args = utils.popfirst(args)
        lnl_boost = kwargs.get(
            "lnl_boost",
            self.lnl_boost if lnl_boost is None else lnl_boost,
        )
        self.lnl_boost = kwargs.get("lnl_boost", lnl_boost)

        tumor_boost, args = utils.popfirst(args)
        tumor_boost = kwargs.get(
            "tumor_boost",
            self.tumor_boost if tumor_boost is None else tumor_boost,
        )
        self.tumor_boost = kwargs.get("tumor_boost", tumor_boost)

        advanced_prob, args = utils.popfirst(args)
        advanced_prob = kwargs.get(
            "advanced_prob",
            self.advanced_prob if advanced_prob is None else advanced_prob,
        )
        self.advanced_prob = kwargs.get("advanced_prob", advanced_prob)

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

    def tstage_evo(self) -> np.ndarray:
        """Evolve only the state of T-stage."""
        time_steps = np.arange(self.max_time + 1)
        tstage_states = np.zeros(shape=(self.max_time + 1, 2), dtype=float)
        tstage_states[:, 0] = (1.0 - self.advanced_prob) ** time_steps
        tstage_states[:, 1] = 1.0 - tstage_states[:, 0]
        return tstage_states

    def state_dist_evo(self) -> tuple[np.ndarray, np.ndarray]:
        """Evolve states for early and late T-stage.

        T-stage is part of the state, this means we need to consider the probability
        of staying early T-stage or transitioning to late T-stage. This means we need to
        construct the distribution for late T-stage recursively.
        """
        early_dist_evo = self.early.state_dist_evo()
        tstage_evo = self.tstage_evo()

        early_dist_evo *= tstage_evo[:, 0].reshape(-1, 1)
        late_dist_evo = np.zeros_like(early_dist_evo)

        for t in range(self.max_time):
            late_dist_evo[t + 1] = (
                self.advanced_prob * early_dist_evo[t] + late_dist_evo[t]
            ) @ self.late.transition_matrix()

        return early_dist_evo, late_dist_evo

    def state_dist(
        self,
        t_stage: str = "early",
        mode: Literal["HMM", "BN"] = "HMM",
    ) -> np.ndarray:
        """Compute the distribution over possible states.

        Do this for a given ``t_stage``, which is essentially a marginalization of the
        evolution over the possible states as computed by :py:meth:`.state_dist_evo`
        with the distribution over diagnosis times for the given T-stage from the
        dictionary returned by :py:meth:`.get_all_distributions`.
        """
        early_dist_evo, late_dist_evo = self.state_dist_evo()
        if mode == "HMM":
            if t_stage == "early":
                diag_time_dist = self.early.get_distribution(t_stage).pmf
                return diag_time_dist @ early_dist_evo
            if t_stage == "late":
                diag_time_dist = self.late.get_distribution(t_stage).pmf
                return diag_time_dist @ late_dist_evo
            raise ValueError("T-stage needs to be early or late.")

        raise NotImplementedError("Only HMM mode is supported as of now.")

    def obs_dist(
        self,
        given_state_dist: np.ndarray | None = None,
        t_stage: str = "early",
        mode: Literal["HMM", "BN"] = "HMM",
    ) -> np.ndarray:
        """Compute the distribution over all possible observations for a given T-stage.

        Returns an array of probabilities for each possible complete observation. This
        entails multiplying the distribution over states as returned by the
        :py:meth:`.state_dist` method with the :py:meth:`.observation_matrix`.

        Note that since the :py:attr:`.observation_matrix` can become very large, this
        method is not very efficient for inference. Instead, we compute the
        :py:meth:`.diagnosis_matrix` from the :py:meth:`.observation_matrix` and
        the :py:meth:`.data_matrix` and use these to compute the likelihood.
        """
        if given_state_dist is None:
            given_state_dist = self.state_dist(t_stage=t_stage, mode=mode)

        if t_stage == "early":
            return self.early.obs_dist(given_state_dist=given_state_dist)
        if t_stage == "late":
            return self.late.obs_dist(given_state_dist=given_state_dist)
        raise ValueError("T-stage needs to be early or late")

    def _hmm_likelihood(self, log: bool = True, t_stage: str | None = None) -> float:
        """Compute the HMM likelihood of data, using the stored params."""
        llh = 0.0 if log else 1.0

        early_dist_evo, late_dist_evo = self.state_dist_evo()

        # do the evolution of states for early
        patient_llhs_early = (
            self.get_distribution("early").pmf
            @ early_dist_evo
            @ self.early.diagnosis_matrix("early").T
        )
        llh = utils.add_or_mult(llh, patient_llhs_early, log=log)

        # do the evolution of states for late
        patient_llhs_late = (
            self.get_distribution("late").pmf
            @ late_dist_evo
            @ self.late.diagnosis_matrix("late").T
        )
        llh = utils.add_or_mult(llh, patient_llhs_late, log=log)

        if t_stage == "early":
            return utils.add_or_mult(0.0 if log else 1.0, patient_llhs_early, log=log)
        if t_stage == "late":
            return utils.add_or_mult(0.0 if log else 1.0, patient_llhs_late, log=log)
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

        raise ValueError("Invalid mode. Only 'HMM' is supported.")

    def posterior_state_dist(
        self,
        given_params: types.ParamsType | None = None,
        given_state_dist: np.ndarray | None = None,
        given_diagnosis: types.DiagnosisType | None = None,
        t_stage: str | int = "early",
        mode: Literal["HMM", "BN"] = "HMM",
    ) -> np.ndarray:
        """Compute the posterior distribution over hidden states given a diagnosis.

        See :py:meth:`.models.Unilateral.posterior_state_dist` for more information.
        """
        if given_state_dist is None:
            utils.safe_set_params(self, given_params)
            given_state_dist = self.state_dist(t_stage, mode=mode)

        if given_diagnosis is None:
            return given_state_dist

        given_state_dist = given_state_dist / given_state_dist.sum()

        return self.early.posterior_state_dist(
            given_state_dist=given_state_dist,
            given_diagnosis=given_diagnosis,
        )

    def marginalize(
        self,
        involvement: dict[str, types.PatternType] | None = None,
        given_state_dist: np.ndarray | None = None,
        t_stage: str = "early",
        mode: Literal["HMM", "BN"] = "HMM",
    ) -> np.ndarray:
        """Marginalize ``given_state_dist`` over matching ``involvement`` patterns.

        See :py:meth:`.models.Unilateral.marginalize` for more information.
        """
        if involvement is None:
            involvement = {}

        if given_state_dist is None:
            given_state_dist = self.state_dist(
                t_stage=t_stage,
                mode=mode,
            )

        return self.early.marginalize(
            involvement=involvement,
            given_state_dist=given_state_dist,
        )

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
        posterior_state_dist = self.posterior_state_dist(
            given_params=given_params,
            given_state_dist=given_state_dist,
            given_diagnosis=given_diagnosis,
            t_stage=t_stage,
            mode=mode,
        )

        return self.marginalize(
            involvement=involvement,
            given_state_dist=posterior_state_dist,
        )
