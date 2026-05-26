import numpy as np
from scipy.stats import norm

# ==============================================
# model_lmfit: lmfit_model_profile
# model_pydream: log_likelihood_monosq_legacy
# inputs_lmfit: q, solvent_data, model_data, params
# inputs_pydream: q, solvent_data, model_data, params
# param_columns: Structure, Motif, Param, Value, Vary, Min, Max
#
# Legacy MDScatter pyDREAM template:
# - weighted simulated SAXS component profiles
# - monodisperse hard-sphere Percus-Yevick S(q)
# - optional solvent profile inside the same global scale as the solute
# - fixed Gaussian likelihood sigma of 1e-4
# - unnormalized summed log-likelihood
# ==============================================
#
# param: solv_w,0.0,True,0.0,1e10
# param: offset,0.0,True,-20.0,30.0
# param: eff_r,9.0,True,3.0,20.0
# param: vol_frac,0.0,True,0.0,0.1
# param: scale,1e-10,False,1e-12,1e-8


LEGACY_LIKELIHOOD_SIGMA = 1e-4


def calc_monodisperse_sq(r, vol_frac, q_values):
    """Return the legacy hard-sphere Percus-Yevick structure factor."""
    q_values = np.asarray(q_values, dtype=float)
    r = float(r)
    vol_frac = float(vol_frac)
    if r <= 0.0:
        raise ValueError("eff_r must be positive")
    if vol_frac < 0.0 or vol_frac >= 1.0:
        raise ValueError("vol_frac must satisfy 0 <= vol_frac < 1")

    a = 2.0 * q_values * r
    a_safe = np.where(np.abs(a) < 1e-12, 1e-12, a)

    alpha = (1.0 + 2.0 * vol_frac) ** 2 / (1.0 - vol_frac) ** 4
    beta = (
        -6.0 * vol_frac * (1.0 + vol_frac / 2.0) ** 2 / (1.0 - vol_frac) ** 4
    )
    gamma = (
        0.5 * vol_frac * (1.0 + 2.0 * vol_frac) ** 2 / (1.0 - vol_frac) ** 4
    )
    g1 = alpha / a_safe**2 * (np.sin(a_safe) - a_safe * np.cos(a_safe))
    g2 = (
        beta
        / a_safe**3
        * (
            2.0 * a_safe * np.sin(a_safe)
            + (2.0 - a_safe**2) * np.cos(a_safe)
            - 2.0
        )
    )
    g3 = (
        gamma
        / a_safe**5
        * (
            -(a_safe**4) * np.cos(a_safe)
            + 4.0
            * (
                (3.0 * a_safe**2 - 6.0) * np.cos(a_safe)
                + (a_safe**3 - 6.0 * a_safe) * np.sin(a_safe)
                + 6.0
            )
        )
    )
    g = g1 + g2 + g3
    sq = 1.0 / (1.0 + 24.0 * vol_frac * (g / a_safe))

    if np.any(np.abs(a) < 1e-12):
        sq = np.asarray(sq, dtype=float)
        sq[np.abs(a) < 1e-12] = (1.0 - vol_frac) ** 4 / (
            1.0 + 2.0 * vol_frac
        ) ** 2
    return sq


def _weight_keys_from_params(params):
    return sorted(
        (key for key in params if key.startswith("w") and key[1:].isdigit()),
        key=lambda key: int(key.lstrip("w")),
    )


def structure_factor_profile(q, solvent_data, model_data, **params):
    """Return the hard-sphere structure-factor trace S(q)."""
    del solvent_data, model_data
    return calc_monodisperse_sq(
        params["eff_r"],
        params["vol_frac"],
        np.asarray(q, dtype=float),
    )


def _legacy_forward_model(
    q_values,
    solvent_data,
    model_data,
    weights,
    solv_w,
    offset,
    eff_r,
    vol_frac,
    scale,
):
    q_values = np.asarray(q_values, dtype=float)
    solvent = (
        np.zeros_like(q_values)
        if solvent_data is None
        else np.asarray(solvent_data, dtype=float)
    )
    mixture = np.zeros_like(q_values, dtype=float)
    for weight, component in zip(weights, model_data):
        mixture += float(weight) * np.asarray(component, dtype=float)

    solute = mixture * calc_monodisperse_sq(eff_r, vol_frac, q_values)
    raw_model = solute + float(solv_w) * solvent
    return float(scale) * raw_model + float(offset)


def lmfit_model_profile(q, solvent_data, model_data, **params):
    """Evaluate the legacy monodisperse SAXS model for Prefit."""
    weight_keys = _weight_keys_from_params(params)
    weights = [params[key] for key in weight_keys]
    return _legacy_forward_model(
        q,
        solvent_data,
        model_data,
        weights,
        params["solv_w"],
        params["offset"],
        params["eff_r"],
        params["vol_frac"],
        params["scale"],
    )


def model_monosq_legacy(params):
    """Return the pyDREAM forward model intensity."""
    global q_values
    global theoretical_intensities
    global solvent_intensities

    n_profiles = len(theoretical_intensities)
    weights = params[:n_profiles]
    solv_w = params[n_profiles]
    offset = params[n_profiles + 1]
    eff_r = params[n_profiles + 2]
    vol_frac = params[n_profiles + 3]
    scale = params[n_profiles + 4]

    return _legacy_forward_model(
        q_values,
        solvent_intensities,
        theoretical_intensities,
        weights,
        solv_w,
        offset,
        eff_r,
        vol_frac,
        scale,
    )


def log_likelihood_monosq_legacy(params):
    """Return the legacy unnormalized Gaussian log-likelihood."""
    global experimental_intensities

    try:
        model_intensity = model_monosq_legacy(params)
    except (ValueError, FloatingPointError):
        return -np.inf
    if not np.all(np.isfinite(model_intensity)):
        return -np.inf

    experimental = np.asarray(experimental_intensities, dtype=float)
    return np.sum(
        norm.logpdf(
            experimental,
            loc=model_intensity,
            scale=LEGACY_LIKELIHOOD_SIGMA,
        )
    )
