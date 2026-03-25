import numpy as np

# model_lmfit: lmfit_model_profile
# model_pydream: log_likelihood_candidate
# inputs_lmfit: q, solvent_data, model_data, params
# param_columns: Structure, Motif, Param, Value, Vary, Min, Max
# param: scale,1.0,True,0.0,10.0


def lmfit_model_profile(q, solvent_data, model_data, **params):
    del q, solvent_data
    return params["scale"] * np.asarray(model_data[0], dtype=float)


def log_likelihood_candidate(params):
    del params
    return -1.0
