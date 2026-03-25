import numpy as np

# model_lmfit: lmfit_model_profile
# model_pydream: log_likelihood_candidate
# inputs_lmfit: q, solvent_data, model_data, params
# inputs_pydream: q_values, experimental_intensities, solvent_intensities, theoretical_intensities, params
# param_columns: Structure, Motif, Param, Value, Vary, Min, Max
# param: solv_w,0.05,True,0.0,1.0
# param: scale,1.0,True,0.0,10.0
# param: offset,0.0,True,-10.0,10.0

q_values = np.asarray([], dtype=float)
experimental_intensities = np.asarray([], dtype=float)
solvent_intensities = np.asarray([], dtype=float)
theoretical_intensities = [np.asarray([], dtype=float)]


def lmfit_model_profile(q, solvent_data, model_data, **params):
    del q
    component = np.asarray(model_data[0], dtype=float)
    solvent = np.asarray(solvent_data, dtype=float)
    blend = (1.0 - params["solv_w"]) * component + params["solv_w"] * solvent
    return params["scale"] * blend + params["offset"]


def log_likelihood_candidate(params):
    weight = float(params[0])
    solv_w = float(params[1])
    scale = float(params[2])
    offset = float(params[3])
    model = lmfit_model_profile(
        q_values,
        solvent_intensities,
        [weight * np.asarray(theoretical_intensities[0], dtype=float)],
        solv_w=solv_w,
        scale=scale,
        offset=offset,
    )
    residuals = np.asarray(experimental_intensities, dtype=float) - model
    return float(-0.5 * np.mean(residuals**2))
