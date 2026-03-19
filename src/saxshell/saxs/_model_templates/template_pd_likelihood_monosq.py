import numpy as np
from scipy.stats import norm

# ================================
# model_lmfit: lmfit_model_profile
# model_pydream: log_likelihood_monosq
# inputs_lmfit: q, solvent_data, model_data, params
# inputs_pydream: q, solvent_data, model_data, params
# param_columns: Structure, Motif, Param, Value, Vary, Min, Max
#
# param: solv_w,0.00,True,0.0,1.0
# param: offset,0,True,-20,30
# param: eff_r,9.0,True,3,20
# param: vol_frac,0.0,True,0.0,0.1
# param: scale,5e-4,False,1e-5,5e-3
# ================================


def calc_monodisperse_sq(r, vol_frac, q_values):
    """Calculate the monodisperse hard-sphere structure factor."""
    sqs = []
    alpha = (1 + 2 * vol_frac) ** 2 / (1 - vol_frac) ** 4
    beta = -6 * vol_frac * (1 + vol_frac / 2) ** 2 / (1 - vol_frac) ** 4
    gamma = 0.5 * vol_frac * (1 + 2 * vol_frac) ** 2 / (1 - vol_frac) ** 4
    for q in q_values:
        a = 2 * q * r
        g1 = alpha / a**2 * (np.sin(a) - a * np.cos(a))
        g2 = beta / a**3 * (2 * a * np.sin(a) + (2 - a**2) * np.cos(a) - 2)
        g3 = (
            gamma
            / a**5
            * (
                -(a**4) * np.cos(a)
                + 4
                * ((3 * a**2 - 6) * np.cos(a) + (a**3 - 6 * a) * np.sin(a) + 6)
            )
        )
        g = g1 + g2 + g3
        sqs.append(1 / (1 + 24 * vol_frac * (g / a)))
    return np.asarray(sqs)


def lmfit_model_profile(q, solvent_data, model_data, **params):
    """Evaluate the lmfit SAXS profile model."""
    weight_keys = sorted(
        (key for key in params if key.startswith("w")),
        key=lambda key: int(key.lstrip("w")),
    )
    weights = [params[key] for key in weight_keys]
    solv_w = params["solv_w"]
    offset = params["offset"]
    eff_r = params["eff_r"]
    vol_frac = params["vol_frac"]
    scale = params["scale"]

    mixture = np.zeros_like(q)
    for weight, component in zip(weights, model_data):
        mixture += weight * component

    iq = mixture * calc_monodisperse_sq(eff_r, vol_frac, q)
    iq += solv_w * solvent_data
    return iq * scale + offset


def log_likelihood_monosq(params):
    """Compute the normalized monodisperse SAXS log-likelihood."""
    global q_values, experimental_intensities
    global theoretical_intensities, solvent_intensities

    n_profiles = len(theoretical_intensities)
    weights = params[:n_profiles]
    solv_w = params[n_profiles]
    offset = params[n_profiles + 1]
    eff_r = params[n_profiles + 2]
    vol_frac = params[n_profiles + 3]
    scale = params[n_profiles + 4]

    mixture = np.zeros_like(q_values)
    for weight, component in zip(weights, theoretical_intensities):
        mixture += weight * component

    iq = mixture * calc_monodisperse_sq(eff_r, vol_frac, q_values)
    iq += solv_w * solvent_intensities
    model_intensity = iq * scale + offset

    n_points = len(experimental_intensities)
    log_likelihood = np.sum(
        norm.logpdf(
            experimental_intensities,
            loc=model_intensity,
            scale=1e-4,
        )
    )
    return log_likelihood / n_points if n_points > 0 else log_likelihood
