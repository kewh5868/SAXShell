from saxshell.saxs._model_templates import (
    template_pydream_poly_lma_hs_mix_approx as _mixed,
)

# model_lmfit: lmfit_model_profile
# model_pydream: log_likelihood_poly_lma_hs
# inputs_lmfit: q, solvent_data, model_data, effective_radii, params
# inputs_pydream: q_values, experimental_intensities, solvent_intensities, theoretical_intensities, effective_radii, params
# param_columns: Structure, Motif, Param, Value, Vary, Min, Max
# cluster_geometry_metadata: true
# param: phi_solute,0.02,False,0.0,0.5
# param: phi_int,0.02,True,0.0,0.4
# param: solvent_scale,1.0,False,0.0,1.0
# param: scale,1.0,True,1e-8,1e8
# param: offset,0.0,True,-1e6,1e6
# param: log_sigma,-9.21,True,-20.0,5.0

calc_hardsphere_sq = _mixed.calc_hardsphere_sq
normalize_profile_fractions = _mixed.normalize_profile_fractions
equivalent_volume_radius = _mixed.equivalent_volume_radius
polydisperse_lma_hs_model = _mixed.polydisperse_lma_hs_model
structure_factor_profile = _mixed.structure_factor_profile


def lmfit_model_profile(
    q, solvent_data, model_data, effective_radii, **params
):
    return _mixed.lmfit_model_profile(
        q,
        solvent_data,
        model_data,
        effective_radii,
        **params,
    )


def _sync_runtime_globals() -> None:
    for name in (
        "q_values",
        "experimental_intensities",
        "solvent_intensities",
        "theoretical_intensities",
        "effective_radii",
        "FULL_PARAMETER_NAMES",
    ):
        if name in globals():
            setattr(_mixed, name, globals()[name])


def model_poly_lma_hs(params):
    _sync_runtime_globals()
    return _mixed.model_poly_lma_hs(params)


def log_likelihood_poly_lma_hs(params):
    _sync_runtime_globals()
    return _mixed.log_likelihood_poly_lma_hs(params)
