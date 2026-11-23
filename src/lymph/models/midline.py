"""Module for modeling metastatic progression bilaterally with tumor lateralization."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from lymph import diagnosis_times, matrix, modalities, models, types, utils

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)


EXT_COL_OLD = ("tumor", "1", "extension")
CENT_COL_OLD = ("tumor", "1", "central")
EXT_COL_NEW = ("tumor", "core", "extension")
CENT_COL_NEW = ("tumor", "core", "central")
MAP_EXT_COL = ("_model", "core", "extension")
MAP_CENT_COL = ("_model", "core", "central")
MAP_T_COL = ("_model", "core", "t_stage")
MAP_COH_COL = ("tumor", "algo", "ext_coh")
DIST_MID_COL = ("tumor", "algo", "extension_dist")


def parse_cohorts(
    cohorts: Mapping[int, bool] | Sequence[bool] | None,
) -> tuple[bool, ...]:
    """Turn the ``cohorts`` argument into a tuple of midline extension statuses.

    The position in the returned tuple is the cohort id. Cohorts are ordered from
    most lateralised to most extended: cohort 0 is the lateralised reference, and
    all lateralised cohorts come before all extended ones.

    >>> parse_cohorts(None)
    (False, True)
    >>> parse_cohorts({1: True, 0: False, 2: True})
    (False, True, True)
    >>> parse_cohorts([False, False, True])
    (False, False, True)
    """
    if cohorts is None:
        return (False, True)  # backwards compatible: one noext, one ext cohort

    if isinstance(cohorts, Mapping):
        ids = list(cohorts.keys())
        if sorted(ids) != list(range(len(ids))):
            raise ValueError(
                f"Cohort ids must be 0, 1, ..., N-1 without gaps, got {ids}.",
            )
        cohorts = [cohorts[cid] for cid in range(len(ids))]

    statuses = tuple(cohorts)
    if not all(isinstance(s, bool | np.bool_) for s in statuses):
        raise TypeError(f"Cohort statuses must be bools, got {statuses}.")

    statuses = tuple(bool(s) for s in statuses)
    # sorted() puts all False before all True; the set check ensures both exist
    if list(statuses) != sorted(statuses) or set(statuses) != {False, True}:
        raise ValueError(
            "Cohorts must be ordered with all lateralised (False) cohorts before "
            f"all extended (True) ones, and contain at least one of each: {statuses}.",
        )

    return statuses


class Midline(
    diagnosis_times.Composite,
    modalities.Composite,
    types.Model,
):
    r"""Models metastatic progression bilaterally with tumor lateralization.

    Model a bilateral lymphatic system where an additional risk factor can
    be provided in the data: Whether or not the primary tumor extended over the
    mid-sagittal line, or is located on the mid-sagittal line.

    It is reasonable to assume (and supported by data) that an extension of the primary
    tumor significantly increases the risk for metastatic spread to the contralateral
    side of the neck. This class attempts to capture this using a simple
    assumption: We assume that the probability of spread to the contralateral
    side for patients *with* midline extension is larger than for patients
    *without* it, but smaller than the probability of spread to the ipsilateral
    side. Formally:

    .. math::
        b_c^{\in} = \alpha \cdot b_i + (1 - \alpha) \cdot b_c^{\not\in}

    where :math:`b_c^{\in}` is the probability of spread from the primary tumor
    to the contralateral side for patients with midline extension, and
    :math:`b_c^{\not\in}` for patients without. :math:`\alpha` is the linear
    mixing parameter.
    """

    def __init__(
        self,
        graph_dict: types.GraphDictType,
        is_symmetric: dict[str, bool] | None = None,
        use_mixing: bool = True,
        cohorts: types.CohortsType | None = None,
        use_central: bool = False,
        use_midext_evo: bool = True,
        named_params: Sequence[str] | None = None,
        marginalize_unknown: bool = True,
        uni_kwargs: dict[str, Any] | None = None,
        **_kwargs,
    ):
        """Initialize the model.

        The class is constructed in a similar fashion to the :py:class:`~.Bilateral`:
        That class contains one :py:class:`~.Unilateral` for each side of the neck,
        while this class will contain several instances of :py:class:`~.Bilateral`,
        one for the ipsilateral side and two to three for the the contralateral side
        covering the cases a) no midline extension, b) midline extension, and c)
        central tumor location.

        Added keyword arguments in this constructor are ``use_mixing``, which controls
        whether to use the above described mixture of spread parameters from tumor to
        the LNLs. And ``use_central``, which controls whether to use a third
        :py:class:`~.Bilateral` model for the case of a central tumor location.

        The parameter ``use_midext_evo`` decides whether the tumor's midline extensions
        should be considered a random variable, in which case it is evolved like the
        state of the LNLs, or not.

        With ``marginalize_unknown`` (default: ``True``), the model will also load
        patients with unknown midline extension status into the model and marginalize
        over their state of midline extension when computing the likelihood. This extra
        data is stored in a :py:class:`~.Bilateral` instance accessible via the
        attribute ``"unknown"``. Note that this bilateral instance does not get updated
        parameters or any other kind of attention. It is solely used to store the data
        and generate diagnosis matrices for those data.

        The ``uni_kwargs`` are passed to all bilateral models.

        .. seealso::

            :py:class:`Bilateral`: Two to four of these are held as attributes by this
            class. One for the case of a mid-sagittal extension of the primary
            tumor, one for the case of no such extension, (possibly) one for the case of
            a central/symmetric tumor, and (possibly) one for the case of unknown
            midline extension status.
        """
        is_symmetric = dict(is_symmetric or {})
        is_symmetric.setdefault("tumor_spread", False)
        is_symmetric.setdefault("lnl_spread", True)

        if is_symmetric["tumor_spread"]:
            raise ValueError(
                "If you want the tumor spread to be symmetric, consider using the "
                "Bilateral class.",
            )
        self.is_symmetric = is_symmetric

        self.use_midext_evo = use_midext_evo
        if self.use_midext_evo and use_central:
            raise ValueError(
                "Evolution to central tumor not yet implemented. Choose to use either "
                "the central model or the midline extension evolution.",
                # Actually, this shouldn't be too hard, but we still need to think
                # about it for a bit.
            )

        # Initialize unified cohort model storage
        self._cohort_midext = parse_cohorts(cohorts)
        self._cohort_models = [
            models.Bilateral(
                graph_dict=graph_dict,
                uni_kwargs=uni_kwargs,
                is_symmetric=self.is_symmetric,
            )
            for _ in self._cohort_midext
        ]

        # Build children dict for composite classes
        children = {
            f"cohort{cid}": model for cid, model in enumerate(self._cohort_models)
        }

        other_children = {}
        if use_central:
            self._central = models.Bilateral(
                graph_dict=graph_dict,
                uni_kwargs=uni_kwargs,
                is_symmetric={
                    "tumor_spread": True,
                    "lnl_spread": self.is_symmetric["lnl_spread"],
                },
            )
            other_children["central"] = self.central

        self._unknown_cohort: dict[bool, models.Bilateral] = {}
        if marginalize_unknown:
            self._unknown = models.Bilateral(
                graph_dict=graph_dict,
                uni_kwargs=uni_kwargs,
                is_symmetric=self.is_symmetric,
            )
            other_children["unknown"] = self.unknown

            for midext, cids in [
                (False, self.lateral_cohorts),
                (True, self.ext_cohorts),
            ]:
                if len(cids) > 1:
                    self._unknown_cohort[midext] = models.Bilateral(
                        graph_dict=graph_dict,
                        uni_kwargs=uni_kwargs,
                        is_symmetric=self.is_symmetric,
                    )
                    key = "unknown_cohort_ext" if midext else "unknown_cohort_noext"
                    other_children[key] = self._unknown_cohort[midext]

        diagnosis_times.Composite.__init__(
            self,
            distribution_children={
                **children,
                **other_children,
            },
            is_distribution_leaf=False,
        )
        modalities.Composite.__init__(
            self,
            modality_children={**children, **other_children},
            is_modality_leaf=False,
        )

        if use_mixing:
            self.mixing_params = 1 / self.num_cohorts

        self.midext_probs = 1 / self.num_cohorts

        if named_params is not None:
            self.named_params = named_params

    @classmethod
    def binary(cls, *args, **kwargs) -> Midline:
        """Create a binary model."""
        uni_kwargs = kwargs.pop("uni_kwargs", {})
        uni_kwargs["allowed_states"] = [0, 1]
        return cls(*args, uni_kwargs=uni_kwargs, **kwargs)

    @classmethod
    def trinary(cls, *args, **kwargs) -> Midline:
        """Create a trinary model."""
        uni_kwargs = kwargs.pop("uni_kwargs", {})
        uni_kwargs["allowed_states"] = [0, 1, 2]
        return cls(*args, uni_kwargs=uni_kwargs, **kwargs)

    @property
    def is_trinary(self) -> bool:
        """Return whether the model is trinary."""
        if self.ext.is_trinary != self.noext.is_trinary:
            raise ValueError("The bilateral models must have the same trinary status.")

        if self.use_central and self.central.is_trinary != self.ext.is_trinary:
            raise ValueError("The bilateral models must have the same trinary status.")

        return self.ext.is_trinary

    @property
    def noext(self) -> models.Bilateral:
        """The reference lateralised cohort model."""
        return self._cohort_models[0]

    @property
    def ext(self) -> models.Bilateral:
        """The extended cohort model. Only defined if there is exactly one."""
        if len(self.ext_cohorts) != 1:
            raise AttributeError(
                "`ext` is ambiguous with more than one extended cohort.",
            )
        return self._cohort_models[self.ext_cohorts[0]]

    @property
    def midext_prob(self) -> float:
        """Return the probability of midline extension (for one extended cohort)."""
        if self.num_cohorts != 2:
            raise AttributeError(
                "`midext_prob` is ambiguous with more than two cohorts; "
                "use `midext_probs` instead.",
            )
        return self._midext_probs[1]

    @midext_prob.setter
    def midext_prob(self, value: float) -> None:
        """Set the probability of midline extension (only for one extended cohort)."""
        if self.num_cohorts != 2:
            raise AttributeError(
                "`midext_prob` is ambiguous with more than two cohorts; "
                "use `midext_probs` instead.",
            )
        self.midext_probs = {1: value}

    @property
    def midext_probs(self) -> dict[int, float]:
        """Return the probabilities of midline extension.

        Without midline evolution this denotes the probability of belonging to each
        cohort. With midline evolution, this denotes the probability of the tumor
        growing from one cohort to the next one within a timestep.
        """
        return self._midext_probs.copy()

    def _validate_midext_probs(self, value: dict[int, float]) -> None:
        """Raise a ``ValueError`` if ``value`` is not valid for midext probs."""
        if set(value) != set(self.mixed_cohorts):
            raise ValueError(
                f"Midext probs needed exactly for cohorts {list(self.mixed_cohorts)}, "
                f"got {sorted(value)}.",
            )
        if not all(0.0 <= v <= 1.0 for v in value.values()):
            raise ValueError(
                "All midline extension probs must be in the range [0, 1].",
            )
        if not self.use_midext_evo and sum(value.values()) > 1.0:
            raise ValueError(
                f"The sum of midline extension probs cannot exceed 1.0, (got {value}).",
            )

    @midext_probs.setter
    def midext_probs(self, value: float | Mapping[int, float]) -> None:
        """Set the probability of midline extension.

        A single number sets all cohorts to that value. A mapping only updates the given
        cohorts and keeps the others, so individual values can be set.
        """
        if isinstance(value, Mapping):
            new_probs = {**self._midext_probs, **value}
        else:
            new_probs = dict.fromkeys(self.mixed_cohorts, float(value))

        self._validate_midext_probs(new_probs)
        self._midext_probs = new_probs

    @property
    def mixing_param(self) -> float | None:
        """Return the mixing parameter (only for one extended cohort)."""
        if not self.use_mixing:
            return None
        if self.num_cohorts != 2:
            raise AttributeError(
                "`mixing_param` is ambiguous with more than two cohorts; "
                "use `mixing_params` instead.",
            )
        return self._mixing_params[1]

    @mixing_param.setter
    def mixing_param(self, value: float) -> None:
        """Set the mixing parameter (only for one extended cohort)."""
        if self.num_cohorts != 2:
            raise AttributeError(
                "`mixing_param` is ambiguous with more than two cohorts, "
                "use `mixing_params` instead.",
            )
        self.mixing_params = {1: value}

    @property
    def mixing_params(self) -> dict[int, float]:
        """Return the mixing parameter of each cohort."""
        if hasattr(self, "_mixing_params"):
            return self._mixing_params.copy()
        return {}

    def _validate_mixing_params(self, value: dict[int, float]) -> None:
        """Raise a ``ValueError`` if ``value`` is not valid for mixing params."""
        if set(value) != set(self.mixed_cohorts):
            raise ValueError(
                f"Mixing params needed exactly for cohorts {list(self.mixed_cohorts)}, "
                f"got {sorted(value)}.",
            )
        if not all(0.0 <= v <= 1.0 for v in value.values()):
            raise ValueError("The mixing parameters must be in the range [0, 1].")

    @mixing_params.setter
    def mixing_params(self, value: float | Mapping[int, float]) -> None:
        """Set the mixing parameters.

        A single number sets all cohorts to that value. A mapping only updates the given
        cohorts and keeps the others, so individual values can be set.
        """
        if isinstance(value, Mapping):
            new_params = {**self._mixing_params, **value}
        else:
            new_params = dict.fromkeys(self.mixed_cohorts, float(value))
        self._validate_mixing_params(new_params)
        self._mixing_params = new_params

    @property
    def use_mixing(self) -> bool:
        """Return whether the model uses a mixing parameter."""
        return hasattr(self, "_mixing_params")

    @property
    def use_central(self) -> bool:
        """Return whether the model uses a central model."""
        return hasattr(self, "_central")

    @property
    def central(self) -> models.Bilateral:
        """Return the central model."""
        if self.use_central:
            return self._central
        raise AttributeError("This instance does not account for central tumors.")

    @property
    def marginalize_unknown(self) -> bool:
        """Return whether the model marginalizes over unknown midline extension."""
        return hasattr(self, "_unknown")

    @property
    def unknown(self) -> models.Bilateral:
        """Return the model storing the patients with unknown midline extension."""
        if self.marginalize_unknown:
            return self._unknown
        raise AttributeError(
            "This instance does not marginalize over unknown midline extension.",
        )

    @property
    def num_cohorts(self) -> int:
        """Number of cohorts."""
        return len(self._cohort_midext)

    @property
    def cohorts(self) -> range:
        """All cohort ids, from most lateralised to most extended."""
        return range(self.num_cohorts)

    @property
    def mixed_cohorts(self) -> range:
        """Cohorts whose params are derived from the reference cohort 0."""
        return range(1, self.num_cohorts)

    @property
    def lateral_cohorts(self) -> list[int]:
        """Lateralised cohorts."""
        return [cid for cid, ext in enumerate(self._cohort_midext) if not ext]

    @property
    def ext_cohorts(self) -> list[int]:
        """Extended cohorts."""
        return [cid for cid, ext in enumerate(self._cohort_midext) if ext]

    def _mixing_key(self, cid: int) -> str:
        """Name of the mixing parameter of cohort ``cid``."""
        return "mixing" if self.num_cohorts == 2 else f"mixing{cid}"

    def _midext_key(self, cid: int) -> str:
        """Name of the midline extension probability of cohort ``cid``."""
        return "midext_prob" if self.num_cohorts == 2 else f"midext_prob{cid}"

    def _cohort_prefix(self, cid: int) -> str:
        """Prefix of the contra tumor spread of cohort ``cid`` without mixing."""
        if self.num_cohorts == 2:
            return "ext" if cid == 1 else "noext"
        return f"cohort{cid}"

    def get_tumor_spread_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> dict[str, float] | Iterable[float]:
        """Return the tumor spread parameters of the model.

        If the model uses the mixing parameter, the returned params will contain the
        ipsilateral spread from tumor to LNLs, the contralateral ones for the case of
        no midline extension, and the mixing parameter. Otherwise, it will contain the
        contralateral params for the cases of present and absent midline extension.
        """
        params = {}
        params["ipsi"] = self._cohort_models[0].ipsi.get_tumor_spread_params(
            as_flat=as_flat,
        )

        if self.use_mixing:
            params["contra"] = self._cohort_models[0].contra.get_tumor_spread_params(
                as_flat=as_flat,
            )
            mixing_params = self.mixing_params
            for cid in self.mixed_cohorts:
                params[self._mixing_key(cid)] = mixing_params[cid]

        else:
            for cid, model in enumerate(self._cohort_models):
                params[self._cohort_prefix(cid)] = {
                    "contra": model.contra.get_tumor_spread_params(as_flat=as_flat),
                }

        if as_flat or not as_dict:
            params = utils.flatten(params)

        return params if as_dict else params.values()

    def get_lnl_spread_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> dict[str, float] | Iterable[float]:
        """Return the LNL spread parameters of the model.

        Depending on the value of ``is_symmetric["lnl_spread"]``, the returned params
        may contain only one set of spread parameters (if ``True``) or one for the ipsi-
        and one for the contralateral side (if ``False``).
        """
        lnl_params = self._cohort_models[0].get_lnl_spread_params(as_flat=False)

        if not all(
            self._cohort_models[cid].get_lnl_spread_params(as_flat=False) == lnl_params
            for cid in self.mixed_cohorts
        ):
            warnings.warn(
                "LNL spread params not synched between ext and noext or cohort models. "
                "Returning the noext / reference model params.",
            )

        if self.use_central:
            central_lnl_params = self.central.get_lnl_spread_params(as_flat=False)
            if central_lnl_params != lnl_params:
                warnings.warn(
                    "LNL spread params not synched between central and noext models. "
                    "Returning the noext / reference params.",
                )

        if as_flat or not as_dict:
            lnl_params = utils.flatten(lnl_params)

        return lnl_params if as_dict else lnl_params.values()

    def get_spread_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> dict[str, float] | Iterable[float]:
        """Return the spread parameters of the model.

        This combines the returned values from the calls to
        :py:meth:`get_tumor_spread_params` and :py:meth:`get_lnl_spread_params`.
        """
        params = self.get_tumor_spread_params(as_flat=False)
        lnl_spread_params = self.get_lnl_spread_params(as_flat=False)

        if self.is_symmetric["lnl_spread"]:
            params.update(lnl_spread_params)
        else:
            if "contra" not in params:
                params["contra"] = {}
            params["ipsi"].update(lnl_spread_params["ipsi"])
            params["contra"].update(lnl_spread_params["contra"])

        if as_flat or not as_dict:
            params = utils.flatten(params)

        return params if as_dict else params.values()

    def get_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> types.ParamsType:
        """Return all the parameters of the model.

        Includes the spread parameters from the call to :py:meth:`get_spread_params`
        and the distribution parameters from the call to
        :py:meth:`~.diagnosis_times.Composite.get_distribution_params`. It also appends
        the probability of midline extension to the end of the returned params.
        """
        params = {}
        params.update(self.get_spread_params(as_flat=as_flat))
        params.update(self.get_distribution_params(as_flat=as_flat))

        midext_probs = self.midext_probs
        for cid in self.mixed_cohorts:
            params[self._midext_key(cid)] = midext_probs[cid]

        if as_flat or not as_dict:
            params = utils.flatten(params)

        return params if as_dict else params.values()

    def set_tumor_spread_params(
        self,
        *args: float,
        **kwargs: float,
    ) -> types.ParamsType:
        """Set the spread parameters of the midline model.

        In analogy to the :py:meth:`get_tumor_spread_params` method, this method sets
        the parameters describing how the tumor spreads to the LNLs. How many params
        to provide to this model depends on the value of the ``use_mixing`` and the
        ``use_central`` attributes. Have a look at what the
        :py:meth:`get_tumor_spread_params` method returns for an insight in what you
        can provide.
        """
        cohort_keys = [self._cohort_prefix(cid) for cid in self.cohorts]
        expected_keys = ["ipsi", "contra"] + cohort_keys

        kwargs, global_kwargs = utils.unflatten_and_split(
            kwargs,
            expected_keys=expected_keys,
        )

        # Set ipsilateral tumor spread for all models (same for all)
        ipsi_kwargs = global_kwargs.copy()
        ipsi_kwargs.update(kwargs.get("ipsi", {}))
        if self.use_central:
            self.central.set_tumor_spread_params(*args, **ipsi_kwargs)

        for cid in self.mixed_cohorts:
            self._cohort_models[cid].ipsi.set_tumor_spread_params(*args, **ipsi_kwargs)
        args = self._cohort_models[0].ipsi.set_tumor_spread_params(*args, **ipsi_kwargs)

        # Set contralateral tumor spread
        if self.use_mixing:
            contra_kwargs = global_kwargs.copy()
            contra_kwargs.update(kwargs.get("contra", {}))
            args = self._cohort_models[0].contra.set_tumor_spread_params(
                *args,
                **contra_kwargs,
            )

            new_mixing = {}
            for cid in self.mixed_cohorts:
                value, args = utils.popfirst(args)
                value = global_kwargs.get(self._mixing_key(cid), value)
                if value is not None:
                    new_mixing[cid] = value
            if new_mixing:
                self.mixing_params = new_mixing

            ipsi = self._cohort_models[0].ipsi.get_tumor_spread_params()
            contra = self._cohort_models[0].contra.get_tumor_spread_params()
            for cid, alpha in self.mixing_params.items():
                mixed = {
                    key: alpha * ipsi[key] + (1 - alpha) * contra[key] for key in ipsi
                }
                self._cohort_models[cid].contra.set_tumor_spread_params(
                    **mixed,
                )

        else:
            for cid, model in enumerate(self._cohort_models):
                cohort_kwargs, _ = utils.unflatten_and_split(
                    kwargs.get(self._cohort_prefix(cid), {}),
                    expected_keys=["contra"],
                )
                contra_kwargs = global_kwargs.copy()
                contra_kwargs.update(cohort_kwargs.get("contra", {}))
                args = model.contra.set_tumor_spread_params(
                    *args,
                    **contra_kwargs,
                )

        return args

    def set_lnl_spread_params(self, *args: float, **kwargs: float) -> Iterable[float]:
        """Set the LNL spread parameters of the midline model.

        This works exactly like the :py:meth:`.Bilateral.set_lnl_spread_params` for the
        user, but under the hood, the parameters also need to be distributed to all
        instances of :py:class:`~.Bilateral` depending on the value of the
        ``use_central`` attribute.
        """
        cohort_keys = [self._cohort_prefix(cid) for cid in self.cohorts]
        expected_keys = ["ipsi", "contra"] + cohort_keys

        kwargs, global_kwargs = utils.unflatten_and_split(
            kwargs,
            expected_keys=expected_keys,
        )

        ipsi_kwargs = global_kwargs.copy()
        ipsi_kwargs.update(kwargs.get("ipsi", {}))

        if self.is_symmetric["lnl_spread"]:
            # For symmetric lnl spread, use the same params for all sides of all models
            if self.use_central:
                self.central.ipsi.set_lnl_spread_params(*args, **global_kwargs)
                self.central.contra.set_lnl_spread_params(*args, **global_kwargs)

            # Apply to all models
            for cid in self.mixed_cohorts:
                self._cohort_models[cid].ipsi.set_lnl_spread_params(
                    *args,
                    **global_kwargs,
                )
                self._cohort_models[cid].contra.set_lnl_spread_params(
                    *args,
                    **global_kwargs,
                )
            self._cohort_models[0].ipsi.set_lnl_spread_params(*args, **global_kwargs)
            args = self._cohort_models[0].contra.set_lnl_spread_params(
                *args,
                **global_kwargs,
            )

        else:
            # For asymmetric lnl spread, use different params for ipsi and contra
            if self.use_central:
                self.central.ipsi.set_lnl_spread_params(*args, **ipsi_kwargs)

            # Apply ipsilateral params to all models
            for cid in self.mixed_cohorts:
                self._cohort_models[cid].ipsi.set_lnl_spread_params(
                    *args,
                    **ipsi_kwargs,
                )
            args = self._cohort_models[0].ipsi.set_lnl_spread_params(
                *args,
                **ipsi_kwargs,
            )

            # Apply contralateral params to all models based on midext status
            contra_kwargs = global_kwargs.copy()
            contra_kwargs.update(kwargs.get("contra", {}))
            if self.use_central:
                self.central.contra.set_lnl_spread_params(*args, **contra_kwargs)

            for cid in self.mixed_cohorts:
                self._cohort_models[cid].contra.set_lnl_spread_params(
                    *args,
                    **contra_kwargs,
                )
            args = self._cohort_models[0].contra.set_lnl_spread_params(
                *args,
                **contra_kwargs,
            )

        return args

    def set_spread_params(self, *args: float, **kwargs: float) -> Iterable[float]:
        """Set the spread parameters of the midline model."""
        args = self.set_tumor_spread_params(*args, **kwargs)
        return self.set_lnl_spread_params(*args, **kwargs)

    def set_params(
        self,
        *args: float,
        **kwargs: float,
    ) -> Iterable[float]:
        """Set all parameters of the model.

        Combines the calls to :py:meth:`.set_spread_params` and
        :py:meth:`.set_distribution_params`. Additionally, it sets the probabilities
        for midline extension. Note that these parameters are always the last ones that
        are set after the spread and distribution parameters.
        """
        # take the midext probs out of kwargs, so they aren't passed to the children
        midext_kwargs = {
            cid: kwargs.pop(self._midext_key(cid))
            for cid in self.mixed_cohorts
            if self._midext_key(cid) in kwargs
        }

        args = self.set_spread_params(*args, **kwargs)
        args = self.set_distribution_params(*args, **kwargs)

        new_midext_probs = {}
        for cid in self.mixed_cohorts:
            value, args = utils.popfirst(args)
            if value is not None:
                new_midext_probs[cid] = value
        new_midext_probs.update(midext_kwargs)  # names override positions

        if new_midext_probs:
            self.midext_probs = new_midext_probs

        return args

    def load_patient_data(
        self,
        patient_data: pd.DataFrame,
        mapping: callable = utils.early_late_mapping,
        cohort_bins: callable | None = None,
    ) -> None:
        """Load patient data into the model.

        This amounts to sorting the patients into three bins:

        1. Patients whose tumor is clearly lateralized, meaning the column
           ``("tumor", "core", "extension")`` reports ``False``. These get assigned to
           the :py:attr:`.noext` attribute.
        2. Those with a central tumor, indicated by ``True`` in the column
           ``("tumor", "core", "central")``. If the :py:attr:`.use_central` attribute is
           set to ``True``, these patients are assigned to the :py:attr:`.central`
           model. Otherwise, they are assigned to the :py:attr:`.ext` model.
        3. The rest, which amounts to patients whose tumor extends over the mid-sagittal
           line but is not central, i.e., symmetric w.r.t to the mid-sagittal line.
           These are assigned to the :py:attr:`.ext` model.

        The split data is sent to the :py:meth:`.Bilateral.load_patient_data` method of
        the respective models.
        """
        # pylint: disable=singleton-comparison
        midext_data = utils.get_item(patient_data, [EXT_COL_NEW, EXT_COL_OLD])
        # first load complete data into noext to assign the loaded dataset to self
        # self._cohort_models[0].load_patient_data(patient_data, mapping)
        # main_data = self._cohort_models[0].patient_data.copy()
        # main_data[MAP_EXT_COL] = midext_data
        # self.patient_data = main_data

        if cohort_bins is None:
            if len(self.cohorts) > 1:
                raise ValueError(
                    "If using mutliple cohorts, a cohort binning must be provided.",
                )
            is_lateralized = midext_data == False  # noqa: E712
            has_extension = midext_data == True  # noqa: E712
            self._cohort_models[0].load_patient_data(
                patient_data[is_lateralized],
                mapping,
            )

            if self.use_central:
                central_data = utils.get_item(
                    patient_data,
                    [CENT_COL_NEW, CENT_COL_OLD],
                )
                is_central = central_data == True  # noqa: E712
                has_extension = has_extension & ~is_central
                self.central.load_patient_data(patient_data[is_central], mapping)

            self._cohort_models[1].load_patient_data(
                patient_data[has_extension],
                mapping,
            )

        else:
            if DIST_MID_COL not in patient_data.columns:
                raise ValueError(
                    ""
                    "if using cohort mapping, patient dataframe"
                    " needs to contain distance column",
                )
            patient_data[MAP_COH_COL] = (
                pd.cut(
                    patient_data[DIST_MID_COL],
                    bins=cohort_bins,
                    labels=np.arange(len(cohort_bins) - 1),
                )
                .astype(float)
                .fillna(-1)
                .astype(int)
            )

            expected = set(np.arange(1, int(patient_data[MAP_COH_COL].max()) + 1))
            if set(self.cohorts) != expected:
                raise ValueError(
                    f"{self.cohorts} does not agree with"
                    " cohorts specified through binning!",
                )
            for coh in self._cohort_models.keys():
                mask = patient_data[MAP_COH_COL] == coh
                self._cohort_models[coh].load_patient_data(patient_data[mask], mapping)

            unknown_cohort = patient_data[MAP_COH_COL] == -1
            unknown_noext = unknown_cohort & (midext_data == False)  # noqa: E712
            unknown_ext = unknown_cohort & (midext_data == True)  # noqa: E712
            if unknown_noext.sum() > 0:
                self.noext.load_patient_data(patient_data[unknown_noext], mapping)
            if unknown_ext.sum() > 0:
                self.ext.load_patient_data(patient_data[unknown_ext], mapping)

            # TODO: also implement central for the cohorts?

        is_unknown = midext_data.isna()
        if self.marginalize_unknown and is_unknown.sum() > 0:
            self.unknown.load_patient_data(
                patient_data[is_unknown],
                mapping,
            )
        elif is_unknown.sum() > 0:
            warnings.warn(
                f"Discarding {is_unknown.sum()} patients where midline extension "
                "is unknown.",
            )

    def midext_evo(self) -> np.ndarray:
        """Evolve only the state of the midline extension."""
        time_steps = np.arange(self.max_time + 1)
        midext_states = np.zeros(shape=(self.max_time + 1, 2), dtype=float)
        midext_states[:, 0] = (1.0 - self.midext_prob) ** time_steps
        midext_states[:, 1] = 1.0 - midext_states[:, 0]
        return midext_states

    def midext_evo_cohorts(self) -> np.ndarray:
        """Evolve only the state of the cohorts."""
        # really not sure conceputally what this means in terms of the
        # cohorts!
        # will just consider with or without midext extension for now
        # but not sure if this is the right way
        time_steps = np.arange(self.max_time + 1)
        midext_states = np.zeros(
            shape=(self.max_time + 1, len(self._cohort_models.keys())),
            dtype=float,
        )
        max_coh = max(self.cohorts)
        midext_states[:, 0] = (1.0 - self.midext_probs[1]) ** time_steps
        for t in range(self.max_time):
            for i in self.cohorts[:-1]:
                midext_states[t, i] = (
                    self.midext_probs[i] * midext_states[t - 1, i - 1]
                    + (1 - self.midext_probs[i + 1]) * midext_states[t - 1, i]
                )
            midext_states[t, max_coh] = (
                self.midext_probs[max_coh] * midext_states[t - 1, max_coh - 1]
                + midext_states[t - 1, max_coh]
            )
        return midext_states

    # not implemented for the cohorts so far! so midext evo will not work!
    def contra_state_dist_evo_old(
        self,
        unmodified_midext_prob: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evolve contra side as mixture of with & without midline extension.

        This computes the evolution of the contralateral state distribution for both
        absent and present midline extension and returns them as a tuple.

        The first element of the tuple is the evolution of the contralateral state
        distribution while having no midline extension. This means that e.g. the value
        at index ``[t,i]`` is the probability of being in state ``i`` at time ``t``,
        **AND** not having midline extension after these ``t`` time steps.

        The second element of the tuple is the evolution of the contralateral state
        distribution where midline extension occurs at some time point. For example,
        the value at index ``[t,i]`` is the probability of being in state ``i`` at time
        ``t``, **AND** having developed midline extension at some time point before.

        To compute this second evolution, we need to mix the model without and with
        midline extension at each time step, following a recusion formula.
        """
        noext_contra_dist_evo = self.noext.contra.state_dist_evo()
        if unmodified_midext_prob:
            ext_contra_dist_evo = self.ext.contra.state_dist_evo()
        elif not self.use_midext_evo:
            ext_contra_dist_evo = self.ext.contra.state_dist_evo()
            noext_contra_dist_evo *= 1.0 - self.midext_prob
            ext_contra_dist_evo *= self.midext_prob

        else:
            ext_contra_dist_evo = np.zeros_like(noext_contra_dist_evo)
            midext_evo = self.midext_evo()

            # Evolution of contralateral state dists, given no midline extension,
            # multiplied with the probabilities of having no midline extension at all
            # time steps, resulting in a vector of length `max_time + 1`:
            #     P(X_c[t] | noext) * P(noext | t)
            noext_contra_dist_evo *= midext_evo[:, 0].reshape(-1, 1)

            for t in range(self.max_time):
                # For the case of midline extension, we need to consider all possible
                # paths that lead to a midline extension at time t+1. We can define
                # this recursively:
                ext_contra_dist_evo[t + 1] = (
                    # it's the probability of developing it just now, in which case we
                    # use the noext state and multiply it with the probability to
                    # develop midline extension at this time step, ...
                    self.midext_prob * noext_contra_dist_evo[t]
                    # ... plus the probability of having it already, in which case we
                    # use the ext state. The probability of "keeping" the midline
                    # extension is 1, so we don't need to multiply it with anything.
                    + ext_contra_dist_evo[t]
                ) @ self.ext.contra.transition_matrix()  # then evolve using ext model

        return noext_contra_dist_evo, ext_contra_dist_evo

    def contra_state_dist_evo(
        self,
        unmodified_midext_prob: bool = False,
    ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        # TODO: deal with unmodified midext_prob
        coh0_contra_dist_evo = self._cohort_models[0].contra.state_dist_evo()
        if not self.use_midext_evo:
            contra_dist_evo = {}
            contra_dist_evo[0] = coh0_contra_dist_evo * (
                1.0
                - sum(
                    [self.midext_probs[coh] for coh in self.cohorts],
                )
            )
            for coh in self.cohorts:
                contra_dist_evo[coh] = self._cohort_models[coh].contra.state_dist_evo()
                contra_dist_evo[coh] *= self.midext_probs[coh]

        else:
            contra_dist_evo = {}
            midext_evo_coh = self.midext_evo_cohorts()
            # TODO: need to figure out what midext_evo_cohort does exactly!!
            # I think it stores the probability that it never grows into the next one,
            # or at least that is what we would need, so not that it stays lateralised,
            # but depending on how it is used later on -> need to figure this out!
            contra_dist_evo[0] = coh0_contra_dist_evo * midext_evo_coh[:, 0].reshape(
                -1,
                1,
            )

            max_coh = max(self.cohorts)
            # calculate evolution over different cohorts (except for the last one)
            for i in range(1, max_coh):
                contra_dist_evo_temp = np.zeros_like(coh0_contra_dist_evo)
                for t in range(self.max_time):
                    contra_dist_evo_temp[t + 1] = (
                        self.midext_probs[i] * contra_dist_evo[i - 1][t]
                        + (1 - self.midext_probs[i + 1]) * contra_dist_evo_temp[t]
                    ) @ self._cohort_models[i].contra.transition_matrix()
                contra_dist_evo[i] = contra_dist_evo_temp

            # calculate evolution into the last cohort
            # (will stay there and not continue to grow)
            contra_dist_evo_temp = np.zeros_like(coh0_contra_dist_evo)
            for t in range(self.max_time):
                contra_dist_evo_temp[t + 1] = (
                    self.midext_probs[max_coh] * contra_dist_evo[max_coh - 1][t]
                    + contra_dist_evo_temp[t]
                ) @ self._cohort_models[max_coh].contra.transition_matrix()
            contra_dist_evo[max_coh] = contra_dist_evo_temp

        return contra_dist_evo

    def state_dist(
        self,
        t_stage: str = "early",
        mode: Literal["HMM", "BN"] = "HMM",
        central: bool = False,
    ) -> np.ndarray:
        """Compute the joint over ipsi- & contralateral hidden states and midline ext.

        If ``central=False``, the result has shape (2, num_states, num_states), where
        the first axis is for the midline extension status, the second for the
        ipsilateral state, and the third for the contralateral state.

        If ``central=True``, the result will be the state distribution of the central
        model's :py:meth:`.Bilateral.state_dist` method.
        """
        # TODO: make sure everything works out for central!
        if central:
            return self.central.state_dist(t_stage, mode)
        ipsi_dist_evo = self._cohort_models[0].ipsi.state_dist_evo()

        contra_dist_evo = self.contra_state_dist_evo()

        if mode == "HMM":
            result = np.empty(
                shape=(
                    len(self._cohort_models.keys()),
                    ipsi_dist_evo.shape[1],
                    ipsi_dist_evo.shape[1],
                ),
            )
            time_marg_matrix = np.diag(self.get_distribution(t_stage).pmf)
            for coh in self._cohort_models.keys():
                result[coh] = ipsi_dist_evo.T @ time_marg_matrix @ contra_dist_evo[coh]
            return result

        raise NotImplementedError("Only HMM mode is supported as of now.")

    def obs_dist(
        self,
        given_state_dist: np.ndarray | None = None,
        t_stage: str = "early",
        mode: Literal["HMM", "BN"] = "HMM",
        central: bool = False,
    ) -> np.ndarray:
        """Compute the joint distribution over the ipsi- & contralateral observations.

        If ``given_state_dist`` is provided, ``t_stage``, ``mode``, and ``central`` are
        ignored. The provided state distribution may be 2D or 3D. The returned
        distribution will have the same dimensionality.

        .. seealso::

            :py:meth:`.Unilateral.obs_dist`
                The corresponding unilateral function. Note that this method returns
                a 2D array, because it computes the probability of any possible
                combination of ipsi- and contralateral observations.

        """
        if given_state_dist is None:
            given_state_dist = self.state_dist(
                t_stage=t_stage,
                mode=mode,
                central=central,
            )
        if given_state_dist.ndim == 2:
            return self._cohort_models[0].obs_dist(given_state_dist=given_state_dist)

        obs_dist = [
            model.obs_dist(given_state_dist=given_state_dist[coh])
            for coh, model in self._cohort_models.items()
        ]
        return np.stack(obs_dist)

    # TODO: have not touched this at all for the cohorts implementation,
    # will need to see what we need to do about this!
    def patient_likelihoods(
        self,
        t_stage: str = None,
        mode: Literal["HMM", "BN"] = "HMM",
        ext_noext_arrays: bool = False,
    ) -> np.ndarray:
        """Compute the likelihood of each patient individually."""
        if mode != "HMM":
            raise NotImplementedError("Only HMM mode is supported as of now.")
        ipsi_dist_evo = self.ext.ipsi.state_dist_evo()
        contra_dist_evo = {}
        contra_dist_evo["noext"], contra_dist_evo["ext"] = self.contra_state_dist_evo(
            unmodified_midext_prob=ext_noext_arrays,
        )
        t_stages = self.t_stages if t_stage is None else [t_stage]
        patient_data = self.patient_data.loc[
            self.patient_data[MAP_T_COL].isin(t_stages)
        ]
        patient_llhs = np.zeros((len(patient_data), 2))
        for stage in t_stages:
            t_idx = patient_data[MAP_T_COL] == stage
            diag_time_matrix = np.diag(self.get_distribution(stage).pmf)
            # see the `Bilateral` model for why this is done in this way.
            for case in ["ext", "noext"]:
                loc = 0 if case == "ext" else 1
                ext_idx = patient_data[MAP_EXT_COL] == (case == "ext")
                joint_state_dist = (
                    ipsi_dist_evo.T @ diag_time_matrix @ contra_dist_evo[case]
                )
                _model = getattr(self, case)
                llhs = matrix.fast_trace(
                    _model.ipsi.diagnosis_matrix(stage),
                    joint_state_dist @ _model.contra.diagnosis_matrix(stage).T,
                )
                patient_llhs[t_idx & ext_idx, loc] = llhs
                try:
                    marg_patient_llhs = matrix.fast_trace(
                        self.unknown.ipsi.diagnosis_matrix(stage),
                        joint_state_dist
                        @ self.unknown.contra.diagnosis_matrix(stage).T,
                    )
                    patient_llhs[t_idx & patient_data[MAP_EXT_COL].isna(), loc] = (
                        marg_patient_llhs
                    )
                except AttributeError:
                    # an AttributeError is raised both when the model has no `unknown`
                    # attribute and when no data is loaded in the `unknown` model.
                    pass
        if (
            not ext_noext_arrays
        ):  # here we colapse the two columns into one likelihood per patient
            patient_llhs = patient_llhs.sum(axis=1)
        return patient_llhs

    def _hmm_likelihood(
        self,
        log: bool = True,
        for_t_stage: str | None = None,
    ) -> float:
        """Compute the likelihood of the stored data under the hidden Markov model."""
        llh = 0.0 if log else 1.0

        ipsi_dist_evo = self._cohort_models[0].ipsi.state_dist_evo()
        contra_dist_evo = self.contra_state_dist_evo()

        t_stages = self.t_stages if for_t_stage is None else [for_t_stage]
        for stage in t_stages:
            diag_time_matrix = np.diag(self.get_distribution(stage).pmf)
            num_states = ipsi_dist_evo.shape[1]
            marg_joint_state_dist = np.zeros(shape=(num_states, num_states))
            noext_marg_joint_state_dist = np.zeros(shape=(num_states, num_states))
            ext_marg_joint_state_dist = np.zeros(shape=(num_states, num_states))
            # see the `Bilateral` model for why this is done in this way.
            for case in self._cohort_models.keys():
                joint_state_dist = (
                    ipsi_dist_evo.T @ diag_time_matrix @ contra_dist_evo[case]
                )
                marg_joint_state_dist += joint_state_dist
                if self._cohort_midext_status[case]:
                    ext_marg_joint_state_dist += joint_state_dist
                else:
                    noext_marg_joint_state_dist += joint_state_dist
                _model = self._cohort_models[case]
                patient_llhs = matrix.fast_trace(
                    _model.ipsi.diagnosis_matrix(stage),
                    joint_state_dist @ _model.contra.diagnosis_matrix(stage).T,
                )
                llh = utils.add_or_mult(llh, patient_llhs, log=log)
            try:
                # marginalise over the extended cohorts
                ext_marg_patient_llh = matrix.fast_trace(
                    self.ext.ipsi.diagnosis_matrix(stage),
                    ext_marg_joint_state_dist
                    @ self.ext.contra.diagnosis_matrix(stage).T,
                )
                llh = utils.add_or_mult(llh, ext_marg_patient_llh, log=log)

                # marginalise over the no-extension cohorts
                noext_marg_patient_llh = matrix.fast_trace(
                    self.noext.ipsi.diagnosis_matrix(stage),
                    noext_marg_joint_state_dist
                    @ self.noext.contra.diagnosis_matrix(stage).T,
                )
                llh = utils.add_or_mult(llh, noext_marg_patient_llh, log=log)

                # marginalise over unknown midline extension status
                marg_patient_llhs = matrix.fast_trace(
                    self.unknown.ipsi.diagnosis_matrix(stage),
                    marg_joint_state_dist
                    @ self.unknown.contra.diagnosis_matrix(stage).T,
                )
                llh = utils.add_or_mult(llh, marg_patient_llhs, log=log)
            except AttributeError:
                # an AttributeError is raised both when the model has no `unknown`
                # attribute and when no data is loaded in the `unknown` model.
                pass

        if self.use_central:
            if log:
                llh += self.central.likelihood(log=log, t_stage=for_t_stage)
            else:
                llh *= self.central.likelihood(log=log, t_stage=for_t_stage)

        return llh

    def likelihood(
        self,
        given_params: types.ParamsType | None = None,
        log: bool = True,
        t_stage: str | None = None,
        mode: Literal["HMM", "BN"] = "HMM",
    ) -> float:
        """Compute the (log-)likelihood of the stored data given the model (and params).

        See the documentation of :py:meth:`.types.Model.likelihood` for more
        information on how to use the ``given_params`` parameter.

        Returns the log-likelihood if ``log`` is set to ``True``. Note that in contrast
        to the :py:class:`.Bilateral` model, the midline model does not support the
        Bayesian network mode.

        .. note::

            The computation is faster if no parameters are given, since then the
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

        raise NotImplementedError("Only HMM mode is supported as of now.")

    def posterior_state_dist(
        self,
        given_params: types.ParamsType | None = None,
        given_state_dist: np.ndarray | None = None,
        given_diagnosis: dict[str, types.DiagnosisType] | None = None,
        t_stage: str = "early",
        mode: Literal["HMM", "BN"] = "HMM",
        midext: bool | None = None,
        central: bool = False,
        cohort: int | None = None,
    ) -> float:
        """Compute the posterior state distribution.

        Using either the ``given_params`` or the ``given_state_dist`` argument, this
        method computes the posterior state distribution of the model for the
        ``given_diagnosis``, a specific ``t_stage``, whether the tumor extends over the
        mid-sagittal line (``midext``), and whether it is central (``central``, only
        used if :py:attr:`use_central` is ``True``).

        .. seealso::

            :py:meth:`.types.Model.posterior_state_dist`
                The corresponding method in the base class.
            :py:meth:`.Bilateral.posterior_state_dist`
                The bilateral method that is ultimately called by this one.
        """
        # NOTE: When given a 2D state distribution, it does not matter which of the
        #       Bilateral models is used to compute the risk, since the state dist is
        #       is the only thing that could differ between models.
        if given_state_dist is None:
            utils.safe_set_params(self, given_params)
            given_state_dist = self.state_dist(
                t_stage=t_stage,
                mode=mode,
                central=central,
            )

        if given_state_dist.ndim == 2:
            return self._cohort_models[0].posterior_state_dist(
                given_state_dist=given_state_dist,
                given_diagnosis=given_diagnosis,
            )

        if central:
            raise ValueError(
                "The `given_state_dist` must be 2D for the central model.",
            )

        if cohort is not None:
            if midext is not None and midext != self._cohort_midext_status[cohort]:
                raise ValueError(
                    f"Inconsistent cohort {cohort} and midext {midext} parameters",
                )
            given_state_dist = given_state_dist[cohort]
            given_state_dist = given_state_dist / given_state_dist.sum()
        elif midext is not None:
            sub_cohorts = [
                k for k, v in self._cohort_midext_status.items() if v == midext
            ]
            given_state_dist = np.sum(given_state_dist[sub_cohorts], axis=0)
            given_state_dist = given_state_dist / given_state_dist.sum()
        else:
            given_state_dist = np.sum(given_state_dist, axis=0)

        return self._cohort_models[0].posterior_state_dist(
            given_state_dist=given_state_dist,
            given_diagnosis=given_diagnosis,
        )

    def marginalize(
        self,
        involvement: dict[str, types.PatternType] | None = None,
        given_state_dist: np.ndarray | None = None,
        t_stage: str = "early",
        mode: Literal["HMM", "BN"] = "HMM",
        midext: bool | None = None,
        central: bool = False,
        cohort: int | None = None,
    ) -> float:
        """Marginalize ``given_state_dist`` over matching ``involvement`` patterns.

        Any state that matches the provided ``involvement`` pattern is marginalized
        over. For this, the :py:func:`.matrix.compute_encoding` function is used.

        The arguments ``t_stage``, ``mode``, and ``central`` are only used if
        ``given_state_dist`` is ``None``. In this case they are passed to the
        :py:meth:`.state_dist` method.
        """
        if involvement is None:
            involvement = {}

        if given_state_dist is None:
            given_state_dist = self.state_dist(
                t_stage=t_stage,
                mode=mode,
                central=central,
            )

        if given_state_dist.ndim == 2:
            return self._cohort_models[0].marginalize(
                involvement=involvement,
                given_state_dist=given_state_dist,
            )

        if cohort is not None:
            if midext is not None and midext != self._cohort_midext_status[cohort]:
                raise ValueError(
                    f"Inconsistent cohort {cohort} and midext {midext} parameters",
                )
            given_state_dist = given_state_dist[cohort]
        elif midext is not None:
            sub_cohorts = [
                k for k, v in self._cohort_midext_status.items() if v == midext
            ]
            given_state_dist = np.sum(given_state_dist[sub_cohorts], axis=0)
        else:
            given_state_dist = np.sum(given_state_dist, axis=0)

        return self._cohort_models[0].marginalize(
            involvement=involvement,
            given_state_dist=given_state_dist,
        )

    def risk(
        self,
        involvement: types.PatternType | None = None,
        given_params: types.ParamsType | None = None,
        given_state_dist: np.ndarray | None = None,
        given_diagnosis: dict[str, types.DiagnosisType] | None = None,
        t_stage: str = "early",
        midext: bool | None = None,
        central: bool = False,
        mode: Literal["HMM", "BN"] = "HMM",
        cohort: int | None = None,
    ) -> float:
        """Compute the risk of nodal involvement ``given_diagnosis``.

        In addition to the arguments of the :py:meth:`.Bilateral.risk` method, this
        also allows specifying if the patient's tumor extended over the mid-sagittal
        line (``midext=True``) or if it was even located right on that line
        (``central=True``).

        For logical reasons, ``midext=False`` makes no sense if ``central=True`` and
        is thus ignored.

        .. warning::

            As in the :py:meth:`.Bilateral.posterior_state_dist` method, you may
            provide a precomputed (joint) state distribution in the ``given_state_dist``
            argument. Here, this ``given_state_dist`` may be a 2D array, in which case
            it is assumed you know how it was computed and the arguments ``t_stage``,
            ``midext``, ``central``, and ``mode`` are ignored. If it is 3D, it should
            have the shape ``(2, num_states, num_states)`` and be the output of the
            :py:meth:`.Midline.state_dist` method. In this case, the ``midext``
            argument is *not* ignored: It may be used to select the correct state
            distribution (when ``True`` or ``False``), or marginalize over the midline
            extension status (when ``midext=None``).
        """
        # Validate cohort parameter for cohort models
        if cohort is not None:
            if not isinstance(cohort, int | np.integer):
                raise ValueError(f"cohort must be an integer, got {type(cohort)}")
            if cohort < 0:
                raise ValueError(f"cohort must be non-negative, got {cohort}")
            if cohort > 0 and hasattr(self, "cohorts") and cohort not in self.cohorts:
                available_cohorts = [0] + list(self.cohorts)
                raise ValueError(
                    f"cohort must be in {available_cohorts}, got {cohort}",
                )
        # I think we don't need to infer midext / cohort from each other,
        # only want to use 'one at a time'!

        posterior_state_dist = self.posterior_state_dist(
            given_params=given_params,
            given_state_dist=given_state_dist,
            given_diagnosis=given_diagnosis,
            t_stage=t_stage,
            midext=midext,
            central=central,
            mode=mode,
            cohort=cohort,
        )

        return self.marginalize(
            involvement=involvement,
            given_state_dist=posterior_state_dist,
            midext=midext,
            cohort=cohort,
        )

    # not checked so far for the cohorts implementation!
    def draw_patients(
        self,
        num: int,
        stage_dist: Iterable[float],
        rng: np.random.Generator | None = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Draw ``num`` patients from the parameterized model."""
        if rng is None:
            rng = np.random.default_rng(seed)

        if sum(stage_dist) != 1.0:
            warnings.warn("Sum of stage distribution is not 1. Renormalizing.")
            stage_dist = np.array(stage_dist) / sum(stage_dist)

        if self.use_central:
            raise NotImplementedError(
                "Drawing patients from the central model not yet supported.",
            )

        drawn_t_stages = rng.choice(
            a=self.t_stages,
            p=stage_dist,
            size=num,
        )
        distributions = self.get_all_distributions()
        drawn_diag_times = np.array(
            [
                distributions[t_stage].draw_diag_times(rng=rng)
                for t_stage in drawn_t_stages
            ],
        )

        if self.use_midext_evo:
            midext_evo = self.midext_evo()
            drawn_midexts = np.array(
                [
                    rng.choice(a=[False, True], p=midext_evo[t])
                    for t in drawn_diag_times
                ],
            )
        else:
            drawn_midexts = rng.choice(
                a=[False, True],
                p=[1.0 - self.midext_prob, self.midext_prob],
                size=num,
            )

        ipsi_evo = self.ext.ipsi.state_dist_evo()
        drawn_diags = np.empty(
            shape=(num, self.ext.ipsi.obs_list.shape[1] * 2),
            dtype=bool,
        )
        for case in ["ext", "noext"]:
            case_model = getattr(self, case)
            drawn_ipsi_diags = utils.draw_diagnosis(
                diagnosis_times=drawn_diag_times[drawn_midexts == (case == "ext")],
                state_evolution=ipsi_evo,
                observation_matrix=case_model.ipsi.observation_matrix(),
                possible_diagnosis=case_model.ipsi.obs_list,
                rng=rng,
                seed=seed,
            )
            drawn_contra_diags = utils.draw_diagnosis(
                diagnosis_times=drawn_diag_times[drawn_midexts == (case == "ext")],
                state_evolution=case_model.contra.state_dist_evo(),
                observation_matrix=case_model.contra.observation_matrix(),
                possible_diagnosis=case_model.contra.obs_list,
                rng=rng,
                seed=seed,
            )
            drawn_case_diags = np.concatenate(
                [drawn_ipsi_diags, drawn_contra_diags],
                axis=1,
            )
            drawn_diags[drawn_midexts == (case == "ext")] = drawn_case_diags

        # construct MultiIndex with "ipsi" and "contra" at top level to allow
        # concatenation of the two separate drawn diagnosis
        sides = ["ipsi", "contra"]
        modality_names = list(self.get_all_modalities().keys())
        lnl_names = list(self.ext.ipsi.graph.lnls.keys())
        multi_cols = pd.MultiIndex.from_product([sides, modality_names, lnl_names])

        # reorder the column levels and thus also the individual columns to match the
        # LyProX format without mixing up the data
        dataset = pd.DataFrame(drawn_diags, columns=multi_cols)
        dataset = dataset.reorder_levels(order=[1, 0, 2], axis="columns")
        dataset = dataset.sort_index(axis="columns", level=0)
        dataset["tumor", "core", "t_stage"] = drawn_t_stages
        dataset["tumor", "core", "extension"] = drawn_midexts
        dataset["patient", "core", "diagnosis_time"] = drawn_diag_times

        return dataset
