"""Test the midline model with multiple (distance) cohorts."""

import numpy as np
import pandas as pd
import pytest

from lymph import models
from lymph.models.midline import COHORT_COL, MAP_COHORT_COL, MAP_T_COL, parse_cohorts

from . import fixtures

EXT_COL = ("tumor", "1", "extension")
CENT_COL = ("tumor", "1", "central")
DIST_COL = ("tumor", "algo", "extension_dist")

COHORT_CONFIGS = [(False, True), (False, True, True), (False, False, True, True)]
COHORT_IDS = ["2coh", "3coh", "4coh"]
COHORT_BINS = {
    2: [-100.0, 0.0, 100.0],
    3: [-100.0, 0.0, 15.0, 100.0],
    4: [-100.0, -10.0, 0.0, 15.0, 100.0],
}


def make_model(
    cohorts: tuple[bool, ...],
    use_midext_evo: bool,
    use_central: bool = False,
    use_mixing: bool = True,
    trinary: bool = False,
    seed: int = 42,
) -> models.Midline:
    """Create a midline model with random parameters and two T-stages."""
    rng = np.random.default_rng(seed)
    model = (models.Midline.trinary if trinary else models.Midline.binary)(
        graph_dict=fixtures.get_graph("small"),
        cohorts=cohorts,
        use_midext_evo=use_midext_evo,
        use_central=use_central,
        use_mixing=use_mixing,
    )
    model.set_modality("CT", spec=0.76, sens=0.81)
    model.set_modality("FNA", spec=0.98, sens=0.80, kind="pathological")
    for t_stage in ["early", "late"]:
        model.set_distribution(
            t_stage,
            fixtures.create_random_dist("frozen", model.max_time, rng),
        )
    params = {k: rng.uniform(0.05, 0.5) for k in model.get_params()}
    for name in params:
        if name.startswith("midext_prob"):
            params[name] = 1.0 / model.num_cohorts  # keep their sum below 1
    model.set_params(**params)
    return model


def make_data(seed: int = 42) -> pd.DataFrame:
    """Load the USZ data and add a synthetic extension distance.

    Lateralised tumors get a negative, extended ones a positive distance. A few
    patients have no distance (but a known extension), and a few have neither.
    """
    rng = np.random.default_rng(seed)
    data = pd.read_csv("./tests/data/2021-usz-oropharynx.csv", header=[0, 1, 2])
    is_ext = data[EXT_COL] == True  # noqa: E712
    data[DIST_COL] = np.where(
        is_ext,
        rng.uniform(0.1, 30.0, len(data)),
        -rng.uniform(0.1, 20.0, len(data)),
    )
    data[EXT_COL] = data[EXT_COL].astype(object)
    data.loc[data.index[:8], DIST_COL] = np.nan
    data.loc[data.index[8:12], [EXT_COL, DIST_COL]] = np.nan
    return data


def load(model: models.Midline, data: pd.DataFrame) -> None:
    """Load data, using the distance bins matching the number of cohorts."""
    model.load_patient_data(data, cohort_bins=COHORT_BINS[model.num_cohorts])


def cohort_weights(model: models.Midline) -> np.ndarray:
    """Return the cohort probabilities (without midline extension evolution)."""
    probs = [model.midext_probs[cid] for cid in model.mixed_cohorts]
    return np.array([1.0 - sum(probs), *probs])


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    """Return patient data with distances and some unknown cohorts/extensions."""
    return make_data()


@pytest.fixture(params=COHORT_CONFIGS, ids=COHORT_IDS)
def cohorts(request) -> tuple[bool, ...]:
    """Return the cohort configurations to test."""
    return request.param


@pytest.fixture(params=[True, False], ids=["evo", "noevo"])
def use_midext_evo(request) -> bool:
    """Return whether to use the midline extension evolution."""
    return request.param


@pytest.fixture
def loaded_model(cohorts, use_midext_evo, data) -> models.Midline:
    """Return a model with random parameters and loaded data."""
    model = make_model(cohorts, use_midext_evo)
    load(model, data)
    return model


@pytest.fixture
def central_model(cohorts, data) -> models.Midline:
    """Return a model with a central model (which requires no evolution)."""
    model = make_model(cohorts, use_midext_evo=False, use_central=True)
    load(model, data)
    return model


# ----------------------------------------------------------------------------------
# Parameter setting
# ----------------------------------------------------------------------------------


def test_parameter_setting_roundtrip(loaded_model: models.Midline) -> None:
    """Check keyword and positional round trips for all cohort configs."""
    rng = np.random.default_rng(42)

    params_to_set = {k: rng.uniform() for k in loaded_model.get_params()}
    for name in params_to_set:
        if name.startswith("midext_prob"):
            params_to_set[name] = 1.0 / loaded_model.num_cohorts  # keep their sum <= 1
    loaded_model.set_params(**params_to_set)
    assert loaded_model.get_params() == params_to_set

    params_to_set = {k: rng.uniform() for k in loaded_model.get_params()}
    for name in params_to_set:
        if name.startswith("midext_prob"):
            params_to_set[name] = 1.0 / (2 * loaded_model.num_cohorts)
    loaded_model.set_params(*params_to_set.values())
    assert loaded_model.get_params() == params_to_set


def test_contra_b_params(loaded_model: models.Midline) -> None:
    """For every mixed cohort, check contra tumor spread parameters."""
    rng = np.random.default_rng(42)
    params_to_set = {k: rng.uniform() for k in loaded_model.get_params()}
    for name in params_to_set:
        if name.startswith("midext_prob"):
            params_to_set[name] = 1.0 / loaded_model.num_cohorts  # keep their sum <= 1
    loaded_model.set_params(**params_to_set)

    ipsi = loaded_model._cohort_models[0].ipsi.get_tumor_spread_params()
    contra = loaded_model._cohort_models[0].contra.get_tumor_spread_params()
    for cid, alpha in loaded_model.mixing_params.items():
        model_params = loaded_model._cohort_models[cid].contra.get_tumor_spread_params()
        mixed_params = {
            key: alpha * ipsi[key] + (1 - alpha) * contra[key] for key in ipsi
        }
        assert model_params == pytest.approx(mixed_params)


def test_contra_spread_without_mixing(cohorts) -> None:
    """Without mixing, each cohort's contra spread is set by its own keywords.

    Regression test: on ``main``, these keyword params were silently ignored.
    """
    model = make_model(cohorts, use_midext_evo=False, use_mixing=False)
    rng = np.random.default_rng(42)
    params_to_set = {k: rng.uniform() for k in model.get_params() if "_contra_" in k}
    model.set_params(**params_to_set)

    for cid in model.cohorts:
        prefix = f"{model._cohort_prefix(cid)}_contra_"
        expected = {
            k.removeprefix(prefix): v
            for k, v in params_to_set.items()
            if k.startswith(prefix)
        }
        actual = model._cohort_models[cid].contra.get_tumor_spread_params()
        assert actual == expected


def test_ipsi_and_lnl_spread_shared(central_model: models.Midline) -> None:
    """Ipsi tumor spread and LNL spread are the same in all cohorts and central."""
    reference = central_model._cohort_models[0]
    ipsi_params = reference.ipsi.get_tumor_spread_params()
    lnl_params = reference.get_lnl_spread_params()

    for child in [*central_model._cohort_models[1:], central_model.central]:
        assert child.ipsi.get_tumor_spread_params() == ipsi_params
        assert child.get_lnl_spread_params() == lnl_params


@pytest.mark.parametrize(
    "how",
    [
        "keyword",
        "positional",
        pytest.param(
            "setter",
            marks=pytest.mark.xfail(
                reason="setter does not update the mixed contra spread (as on main)",
                strict=True,
            ),
        ),
    ],
)
@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_mixing_edge_values_are_kept(cohorts, alpha: float, how: str) -> None:
    """A mixing param of 0.0 must not be replaced by the old value.

    Regression test: the mixing was set with ``new_value or old_value``, which
    discarded a new value of 0.0. At the edges, the mixed contra spread equals
    the reference contra (alpha = 0) or the ipsi spread (alpha = 1).
    """
    model = make_model(cohorts, use_midext_evo=False)
    cid = model.num_cohorts - 1
    key = model._mixing_key(cid)
    assert model.mixing_params[cid] != alpha

    if how == "keyword":
        model.set_params(**{key: alpha})
    elif how == "positional":
        params = model.get_params()
        params[key] = alpha
        model.set_params(*params.values())
    else:
        model.mixing_params = {cid: alpha}

    assert model.mixing_params[cid] == alpha
    reference = model._cohort_models[0]
    expected = reference.ipsi if alpha == 1.0 else reference.contra
    actual = model._cohort_models[cid].contra.get_tumor_spread_params()
    assert actual == pytest.approx(expected.get_tumor_spread_params())


@pytest.mark.parametrize("cohorts", COHORT_CONFIGS[1:], ids=COHORT_IDS[1:])
def test_midext_probs_sum_above_one_raises_without_evo(cohorts) -> None:
    """Without evolution, the cohort probabilities must not sum to more than 1."""
    model = make_model(cohorts, use_midext_evo=False)
    before = model.midext_probs
    too_large = {model._midext_key(cid): 0.6 for cid in model.mixed_cohorts}

    with pytest.raises(ValueError, match="sum"):
        model.midext_probs = 0.6
    with pytest.raises(ValueError, match="sum"):
        model.set_params(**too_large)
    assert model.midext_probs == before


@pytest.mark.parametrize("cohorts", COHORT_CONFIGS[1:], ids=COHORT_IDS[1:])
def test_midext_probs_sum_above_one_allowed_with_evo(cohorts) -> None:
    """With evolution, the probs are per-step transitions and may sum above 1."""
    model = make_model(cohorts, use_midext_evo=True)
    model.midext_probs = 0.6

    assert model.midext_probs == dict.fromkeys(model.mixed_cohorts, 0.6)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_midext_probs_out_of_range_raises(cohorts, use_midext_evo, value) -> None:
    """Each midline extension prob must lie in [0, 1], with or without evolution."""
    model = make_model(cohorts, use_midext_evo)

    with pytest.raises(ValueError, match="range"):
        model.midext_probs = value


# ----------------------------------------------------------------------------------
# Likelihood and patient likelihoods
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("t_stage", [None, "early", "late"])
def test_likelihood_equals_sum_of_patient_likelihoods(
    loaded_model: models.Midline,
    t_stage: str | None,
) -> None:
    """The total log-likelihood must equal the sum of the per-patient ones."""
    patient_llhs = loaded_model.patient_likelihoods(t_stage=t_stage)

    assert np.all(patient_llhs > 0.0)
    assert np.isclose(
        np.log(patient_llhs).sum(),
        loaded_model.likelihood(t_stage=t_stage),
    )


@pytest.mark.parametrize("t_stage", [None, "early"])
def test_patient_likelihoods_with_central(
    central_model: models.Midline,
    t_stage: str | None,
) -> None:
    """Central patients are included in the per-patient likelihoods."""
    patient_llhs = central_model.patient_likelihoods(t_stage=t_stage)

    assert np.all(patient_llhs > 0.0)
    assert np.isclose(
        np.log(patient_llhs).sum(),
        central_model.likelihood(t_stage=t_stage),
    )


def test_patient_likelihoods_only_selected_t_stage(
    loaded_model: models.Midline,
) -> None:
    """Only the patients of the given T-stage are returned."""
    t_stages = loaded_model.patient_data[MAP_T_COL]
    num_early = (t_stages == "early").sum()

    assert len(loaded_model.patient_likelihoods(t_stage="early")) == num_early
    assert len(loaded_model.patient_likelihoods()) == len(t_stages)


def test_per_cohort_weighted_sum_equals_summed(cohorts, data) -> None:
    """Weighting the per-cohort columns with the cohort probs gives the total."""
    model = make_model(cohorts, use_midext_evo=False)
    load(model, data)

    per_cohort = model.patient_likelihoods(per_cohort=True)
    summed = model.patient_likelihoods()

    assert per_cohort.shape == (len(summed), model.num_cohorts)
    assert np.allclose(per_cohort @ cohort_weights(model), summed)


def test_per_cohort_central_rows_are_nan(central_model: models.Midline) -> None:
    """Central patients do not belong to a cohort and get NaN per cohort."""
    per_cohort = central_model.patient_likelihoods(per_cohort=True)
    routes = central_model.patient_data[MAP_COHORT_COL]
    is_central = (routes == "central").to_numpy()

    assert is_central.any()
    assert np.isnan(per_cohort[is_central]).all()
    assert not np.isnan(per_cohort[~is_central]).any()


def test_per_cohort_raises_with_evo(cohorts, data) -> None:
    """Per-cohort (conditional) likelihoods are not defined with evolution."""
    model = make_model(cohorts, use_midext_evo=True)
    load(model, data)

    with pytest.raises(NotImplementedError):
        model.patient_likelihoods(per_cohort=True)


def test_central_patients_not_double_counted(cohorts, data) -> None:
    """Central patients only enter the central model, whatever their extension.

    Regression test: on ``main``, a central patient without extension was counted
    in both the lateralised and the central model.
    """
    data = data.copy()
    is_central = data[CENT_COL] == True  # noqa: E712
    data.loc[is_central[is_central].index[:3], [EXT_COL, DIST_COL]] = [False, -5.0]

    with_central = make_model(cohorts, use_midext_evo=False, use_central=True)
    load(with_central, data)
    without_central = make_model(cohorts, use_midext_evo=False)
    load(without_central, data[~is_central])

    assert np.isclose(
        with_central.likelihood(),
        without_central.likelihood() + with_central.central.likelihood(),
    )


def test_invalid_params_give_zero_likelihood(loaded_model: models.Midline) -> None:
    """Invalid parameters must result in a likelihood of zero."""
    name = next(iter(loaded_model.get_params()))

    assert loaded_model.likelihood(given_params={name: 1.5}) == -np.inf
    assert loaded_model.likelihood(given_params={name: 1.5}, log=False) == 0.0


# ----------------------------------------------------------------------------------
# Backwards compatibility (two cohorts)
# ----------------------------------------------------------------------------------


def test_default_cohorts_are_noext_and_ext() -> None:
    """Without ``cohorts``, the model has one lateralised and one extended cohort."""
    model = models.Midline(graph_dict=fixtures.get_graph("small"))
    assert model.num_cohorts == 2
    assert model.lateral_cohorts == [0]
    assert model.ext_cohorts == [1]


@pytest.mark.parametrize("use_mixing", [True, False])
def test_two_cohort_param_names_as_on_main(use_mixing: bool) -> None:
    """With two cohorts, the parameter names are the same as on ``main``."""
    model = make_model((False, True), use_midext_evo=False, use_mixing=use_mixing)
    names = set(model.get_params())

    assert "midext_prob" in names
    if use_mixing:
        assert {"contra_TtoII_spread", "mixing"} <= names
    else:
        assert {"noext_contra_TtoII_spread", "ext_contra_TtoII_spread"} <= names


def test_two_cohort_aliases() -> None:
    """The ``main`` attributes still work and point to the right cohort."""
    model = make_model((False, True), use_midext_evo=False)

    assert model.noext is model._cohort_models[0]
    assert model.ext is model._cohort_models[1]
    assert model.midext_prob == model.midext_probs[1]
    assert model.mixing_param == model.mixing_params[1]


@pytest.mark.parametrize("cohorts", COHORT_CONFIGS[1:], ids=COHORT_IDS[1:])
def test_two_cohort_aliases_raise_with_more_cohorts(cohorts) -> None:
    """The single-value aliases are ambiguous with more than two cohorts."""
    model = make_model(cohorts, use_midext_evo=False)

    with pytest.raises(AttributeError):
        _ = model.midext_prob
    with pytest.raises(AttributeError):
        _ = model.mixing_param


def test_two_cohorts_extension_only_equals_bins(use_midext_evo, data) -> None:
    """Loading by extension (as on ``main``) equals loading by matching bins."""
    by_ext = make_model((False, True), use_midext_evo)
    by_ext.load_patient_data(data)
    by_bins = make_model((False, True), use_midext_evo)
    load(by_bins, data)

    assert np.isclose(by_ext.likelihood(), by_bins.likelihood())


# ----------------------------------------------------------------------------------
# Parsing the cohorts argument
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cohorts", "error"),
    [
        ({0: False, 2: True}, ValueError),  # gap in ids
        ([True, False], ValueError),  # extended before lateralised
        ([False, False], ValueError),  # no extended cohort
        ([True], ValueError),  # no lateralised cohort
        ([0, 1], TypeError),  # not bools
    ],
)
def test_parse_cohorts_invalid(cohorts, error) -> None:
    """Invalid cohort definitions raise."""
    with pytest.raises(error):
        parse_cohorts(cohorts)


def test_parse_cohorts_accepts_numpy_bools() -> None:
    """Numpy bools (e.g. from a DataFrame) are accepted and converted."""
    assert parse_cohorts(np.array([False, True, True])) == (False, True, True)


# ----------------------------------------------------------------------------------
# Midline extension evolution
# ----------------------------------------------------------------------------------


def test_midext_evo_cohorts_is_distribution(cohorts) -> None:
    """Each time step is a distribution over cohorts, starting in cohort 0."""
    evo = make_model(cohorts, use_midext_evo=True).midext_evo_cohorts()

    assert evo.shape[1] == len(cohorts)
    assert np.all(evo >= 0.0)
    assert np.allclose(evo.sum(axis=1), 1.0)
    assert evo[0, 0] == 1.0
    assert np.all(np.diff(evo[:, 0]) <= 0.0)  # cohort 0 can only be left


def test_midext_evo_sums_cohorts_by_status(cohorts) -> None:
    """``midext_evo`` sums the cohort evolution into noext / ext."""
    model = make_model(cohorts, use_midext_evo=True)
    evo = model.midext_evo_cohorts()

    assert np.allclose(
        model.midext_evo(),
        np.stack(
            [evo[:, model.lateral_cohorts].sum(1), evo[:, model.ext_cohorts].sum(1)],
            axis=1,
        ),
    )


def test_midext_evo_raises_without_evo(cohorts) -> None:
    """The cohort evolution is only defined with ``use_midext_evo=True``."""
    with pytest.raises(ValueError, match="use_midext_evo"):
        make_model(cohorts, use_midext_evo=False).midext_evo_cohorts()


def test_contra_state_dist_evo_marginals(cohorts, use_midext_evo) -> None:
    """Summing the joint over states gives the cohort probability at each time."""
    model = make_model(cohorts, use_midext_evo)
    joint = model.contra_state_dist_evo()
    per_cohort = np.stack([joint[cid].sum(axis=1) for cid in model.cohorts], axis=1)

    if use_midext_evo:
        expected = model.midext_evo_cohorts()
    else:
        expected = np.broadcast_to(cohort_weights(model), per_cohort.shape)
    assert np.allclose(per_cohort, expected)


def test_state_dist_is_distribution(cohorts, use_midext_evo) -> None:
    """The joint over cohorts, ipsi and contra states sums to 1."""
    model = make_model(cohorts, use_midext_evo)
    state_dist = model.state_dist(t_stage="early")

    assert state_dist.shape[0] == len(cohorts)
    assert np.isclose(state_dist.sum(), 1.0)


# ----------------------------------------------------------------------------------
# Routing patients to the child models
# ----------------------------------------------------------------------------------


@pytest.fixture
def routing_data(data) -> pd.DataFrame:
    """Return six patients, each hitting one routing rule (for three cohorts)."""
    rows = data.iloc[20:26].copy()
    rows[EXT_COL] = [True, False, False, False, True, np.nan]
    rows[CENT_COL] = [True, False, False, False, False, False]
    rows[DIST_COL] = [5.0, -5.0, 20.0, np.nan, np.nan, np.nan]
    rows[COHORT_COL] = [1, 2, np.nan, np.nan, np.nan, np.nan]
    return rows


def test_route_patients_priority(routing_data) -> None:
    """Central > cohort column > distance bin > extension > unknown."""
    model = make_model((False, True, True), use_midext_evo=False, use_central=True)
    route = model._route_patients(routing_data, cohort_bins=COHORT_BINS[3])

    assert route.tolist() == [
        "central",  # central, although cohort column says 1
        "cohort2",  # cohort column, although distance says 0
        "cohort2",  # distance 20, although extension is False
        "cohort0",  # only extension known, one lateralised cohort
        "unknown_cohort_ext",  # only extension known, two extended cohorts
        "unknown",  # nothing known
    ]


def test_loaded_routes_in_patient_data(routing_data) -> None:
    """The ``patient_data`` property keeps the order and shows each route."""
    model = make_model((False, True, True), use_midext_evo=False, use_central=True)
    load(model, routing_data)
    route = model._route_patients(routing_data, cohort_bins=COHORT_BINS[3])

    assert model.patient_data[MAP_COHORT_COL].tolist() == route.tolist()


def test_route_patients_errors(routing_data) -> None:
    """Invalid cohort entries, bins or missing columns raise."""
    model = make_model((False, True, True), use_midext_evo=False)
    invalid_cohort = routing_data.copy()
    invalid_cohort[COHORT_COL] = 5

    with pytest.raises(ValueError, match="must contain one of"):
        model._route_patients(invalid_cohort, COHORT_BINS[3])
    with pytest.raises(ValueError, match="bins for"):
        model._route_patients(routing_data, COHORT_BINS[2])
    with pytest.raises(ValueError, match="is missing"):
        model._route_patients(routing_data.drop(columns=[DIST_COL]), COHORT_BINS[3])
    with pytest.raises(ValueError, match="is required"):
        model._route_patients(routing_data.drop(columns=[COHORT_COL]), None)


def test_unknown_patients_discarded_without_marginalization(routing_data) -> None:
    """Without ``marginalize_unknown``, unknown-cohort patients are discarded."""
    model = models.Midline(
        graph_dict=fixtures.get_graph("small"),
        cohorts=(False, True, True),
        marginalize_unknown=False,
    )
    with pytest.warns(UserWarning, match="Discarding 2 patients"):
        model.load_patient_data(routing_data, cohort_bins=COHORT_BINS[3])


# ----------------------------------------------------------------------------------
# Risk, posterior and marginalization
# ----------------------------------------------------------------------------------

INVOLVEMENT = {"ipsi": {"II": True}, "contra": {"II": True}}
DIAGNOSIS = {"ipsi": {"CT": {"II": True, "III": False}}}


def test_marginalize_midext_sums_cohorts(cohorts, use_midext_evo) -> None:
    """Marginalizing over a status equals the sum over its cohorts."""
    model = make_model(cohorts, use_midext_evo)
    state_dist = model.state_dist()

    for midext, cids in [(False, model.lateral_cohorts), (True, model.ext_cohorts)]:
        summed = sum(
            model.marginalize(INVOLVEMENT, state_dist, cohort=cid) for cid in cids
        )
        assert np.isclose(
            model.marginalize(INVOLVEMENT, state_dist, midext=midext),
            summed,
        )
    assert np.isclose(model.marginalize({}, state_dist), 1.0)


def test_risk_single_cohort_equals_midext(cohorts) -> None:
    """If a status has only one cohort, ``cohort=c`` and ``midext`` agree."""
    model = make_model(cohorts, use_midext_evo=False)

    for midext, cids in [(False, model.lateral_cohorts), (True, model.ext_cohorts)]:
        if len(cids) != 1:
            continue
        assert np.isclose(
            model.risk(INVOLVEMENT, given_diagnosis=DIAGNOSIS, midext=midext),
            model.risk(INVOLVEMENT, given_diagnosis=DIAGNOSIS, cohort=cids[0]),
        )


def test_risk_is_probability(cohorts, use_midext_evo) -> None:
    """The risk lies in [0, 1] for every cohort."""
    model = make_model(cohorts, use_midext_evo)

    for cid in model.cohorts:
        risk = model.risk(INVOLVEMENT, given_diagnosis=DIAGNOSIS, cohort=cid)
        assert 0.0 <= risk <= 1.0


@pytest.mark.parametrize(
    "kwargs",
    [{"cohort": 0, "midext": True}, {"cohort": 10}, {"cohort": True}],
    ids=["inconsistent", "out-of-range", "bool"],
)
def test_risk_invalid_cohort_raises(kwargs) -> None:
    """Invalid or inconsistent ``cohort`` / ``midext`` combinations raise."""
    model = make_model((False, True, True), use_midext_evo=False)

    with pytest.raises(ValueError):
        model.risk(INVOLVEMENT, **kwargs)


# ----------------------------------------------------------------------------------
# Drawing patients
# ----------------------------------------------------------------------------------


def test_draw_patients_columns(cohorts, use_midext_evo) -> None:
    """Drawn patients have a valid cohort and a matching extension status."""
    model = make_model(cohorts, use_midext_evo)
    drawn = model.draw_patients(num=200, stage_dist=[0.5, 0.5])
    drawn_cohorts = drawn[COHORT_COL]

    assert len(drawn) == 200
    assert drawn_cohorts.isin(model.cohorts).all()
    assert (
        drawn["tumor", "core", "extension"] == drawn_cohorts.isin(model.ext_cohorts)
    ).all()


def test_draw_patients_cohort_frequencies(cohorts) -> None:
    """Without evolution, cohorts are drawn with the cohort probabilities."""
    model = make_model(cohorts, use_midext_evo=False)
    drawn = model.draw_patients(num=5000, stage_dist=[0.5, 0.5])
    freqs = drawn[COHORT_COL].value_counts(normalize=True).sort_index()

    assert np.allclose(freqs.to_numpy(), cohort_weights(model), atol=0.03)


def test_draw_patients_can_be_reloaded(cohorts, use_midext_evo) -> None:
    """Drawn patients can be loaded again and end up in their drawn cohort."""
    model = make_model(cohorts, use_midext_evo)
    drawn = model.draw_patients(num=200, stage_dist=[0.5, 0.5])
    model.load_patient_data(drawn, mapping=lambda t: t)

    expected = ("cohort" + drawn[COHORT_COL].astype(str)).tolist()
    assert model.patient_data[MAP_COHORT_COL].tolist() == expected
    assert np.isfinite(model.likelihood())


def test_draw_patients_raises_with_central() -> None:
    """Drawing from the central model is not supported."""
    model = make_model((False, True), use_midext_evo=False, use_central=True)

    with pytest.raises(NotImplementedError):
        model.draw_patients(num=10, stage_dist=[0.5, 0.5])


# ----------------------------------------------------------------------------------
# Trinary models
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("trinary", [True, False])
def test_is_trinary(cohorts, trinary: bool) -> None:
    """All children, including the unknown and central ones, share the status."""
    model = make_model(cohorts, use_midext_evo=False, use_central=True, trinary=trinary)
    assert model.is_trinary is trinary


def test_trinary_likelihood(cohorts, data) -> None:
    """A trinary model can load data and compute a finite likelihood."""
    model = make_model(cohorts, use_midext_evo=False, trinary=True)
    load(model, data)

    assert np.isfinite(model.likelihood())
