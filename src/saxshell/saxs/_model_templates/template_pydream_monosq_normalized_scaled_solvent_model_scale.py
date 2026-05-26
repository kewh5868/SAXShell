import numpy as np
from scipy.stats import norm

# ==============================================
# model_lmfit: lmfit_model_profile
# model_pydream: log_likelihood_monosq_scaled_solvent_model_scale
# inputs_lmfit: q, solvent_data, model_data, params
# inputs_pydream: q, solvent_data, model_data, params
# param_columns: Structure, Motif, Param, Value, Vary, Min, Max
#
# param: solv_w,1.0,False,0.0,1.0
# param: offset,0,True,-20,30
# param: eff_r,3.0,True,3,20
# param: vol_frac,0.0,False,0.0,0.5
# param: scale,5e-4,True,1e-8,5e-3
#
# MonoSQ normalized, scaled-solvent, model-scaled variant:
#   I_raw(q) = I_solute(q) + solv_w * I_solvent(q)
#   I_model(q) = scale * I_raw(q) + offset
#
# The experimental trace is compared directly against I_model(q); scale and
# offset are never applied to experimental_intensities in this template.
# Existing templates are intentionally left unchanged for compatibility.
# ==============================================


def calc_monodisperse_sq(r, vol_frac, q_values):
    """Return the hard-sphere Percus-Yevick structure factor."""
    q_values = np.asarray(q_values, dtype=float)
    r = float(r)
    vol_frac = float(vol_frac)
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


def _bounded_solvent_weight(value):
    return float(value)


def _weight_keys_from_params(params):
    return sorted(
        (key for key in params if key.startswith("w") and key[1:].isdigit()),
        key=lambda key: int(key[1:]),
    )


def structure_factor_profile(q, solvent_data, model_data, **params):
    """Return the pure hard-sphere structure-factor trace S(q)."""
    del solvent_data, model_data
    return calc_monodisperse_sq(
        params["eff_r"],
        params["vol_frac"],
        np.asarray(q, dtype=float),
    )


def raw_monosq_scaled_solvent_profile(
    q_values,
    solvent_intensities,
    component_intensities,
    weights,
    solv_w,
    eff_r,
    vol_frac,
):
    """Return the unscaled solute-plus-weighted-solvent model branch."""
    q_values = np.asarray(q_values, dtype=float)
    mixture = np.zeros_like(q_values, dtype=float)
    for weight, component in zip(weights, component_intensities):
        mixture += float(weight) * np.asarray(component, dtype=float)

    solute_intensity = mixture * calc_monodisperse_sq(
        eff_r,
        vol_frac,
        q_values,
    )
    solvent_contribution = _bounded_solvent_weight(solv_w) * np.asarray(
        solvent_intensities,
        dtype=float,
    )
    return solute_intensity + solvent_contribution


def scaled_monosq_model_profile(
    q_values,
    solvent_intensities,
    component_intensities,
    weights,
    solv_w,
    eff_r,
    vol_frac,
    scale,
    offset,
):
    """Apply the fit transform to the model curve, not the data
    curve."""
    raw_model = raw_monosq_scaled_solvent_profile(
        q_values,
        solvent_intensities,
        component_intensities,
        weights,
        solv_w,
        eff_r,
        vol_frac,
    )
    return float(scale) * raw_model + float(offset)


def lmfit_model_profile(q, solvent_data, model_data, **params):
    """Evaluate the model-scaled MonoSQ SAXS model for lmfit."""
    weight_keys = _weight_keys_from_params(params)
    weights = [params[key] for key in weight_keys]

    return scaled_monosq_model_profile(
        q,
        solvent_data,
        model_data,
        weights,
        params["solv_w"],
        params["eff_r"],
        params["vol_frac"],
        params["scale"],
        params["offset"],
    )


def model_monosq_scaled_solvent_model_scale(params):
    """Return the forward model intensity for pyDREAM."""
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

    return scaled_monosq_model_profile(
        q_values,
        solvent_intensities,
        theoretical_intensities,
        weights,
        solv_w,
        eff_r,
        vol_frac,
        scale,
        offset,
    )


def log_likelihood_monosq_scaled_solvent_model_scale(params):
    """Return the normalized Gaussian log-likelihood for pyDREAM."""
    global experimental_intensities

    try:
        model_intensity = model_monosq_scaled_solvent_model_scale(params)
    except (ValueError, FloatingPointError):
        return -np.inf
    if not np.all(np.isfinite(model_intensity)):
        return -np.inf

    experimental = np.asarray(experimental_intensities, dtype=float)
    n_points = len(experimental)
    log_likelihood = np.sum(
        norm.logpdf(
            experimental,
            loc=model_intensity,
            scale=1e-4,
        )
    )

    if n_points == 0:
        return log_likelihood

    return log_likelihood / n_points
