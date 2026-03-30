import numpy as np
from scipy.stats import norm

# model_lmfit: lmfit_model_profile
# model_pydream: log_likelihood_poly_lma_hs
# inputs_lmfit: q, solvent_data, model_data, effective_radii, params
# inputs_pydream: q_values, experimental_intensities, solvent_intensities, theoretical_intensities, effective_radii, params
# param_columns: Structure, Motif, Param, Value, Vary, Min, Max
# cluster_geometry_metadata: true
#
# Approximate mixed-shape template:
# - hard-sphere Percus-Yevick is still evaluated on sphere radii
# - sphere rows use r_eff_<weight>
# - ellipsoid rows use a_eff/b_eff/c_eff and are collapsed to an
#   equivalent-volume sphere radius before S(Q) is evaluated
# - this is an approximate mixed hard-body route inspired by Hansen's
#   nonspherical-body-to-polydisperse-sphere literature, not an exact
#   hard-ellipsoid PY closure
#
# param: phi_solute,0.02,False,0.0,0.5
# param: phi_int,0.02,True,0.0,0.4
# param: solvent_scale,1.0,False,0.0,1.0
# param: scale,1.0,True,1e-8,1e8
# param: offset,0.0,True,-1e6,1e6
# param: log_sigma,-9.21,True,-20.0,5.0


def calc_hardsphere_sq(radius, volfraction, q_values):
    q_values = np.asarray(q_values, dtype=float)
    radius = float(radius)
    volfraction = float(volfraction)

    if radius <= 0:
        raise ValueError("radius must be positive")

    if volfraction < 0 or volfraction >= 1:
        raise ValueError("volfraction must satisfy 0 <= phi < 1")

    a = 2.0 * q_values * radius
    a_safe = np.where(np.abs(a) < 1e-12, 1e-12, a)

    alpha = (1.0 + 2.0 * volfraction) ** 2 / (1.0 - volfraction) ** 4
    beta = (
        -6.0
        * volfraction
        * (1.0 + volfraction / 2.0) ** 2
        / (1.0 - volfraction) ** 4
    )
    gamma = (
        0.5
        * volfraction
        * (1.0 + 2.0 * volfraction) ** 2
        / (1.0 - volfraction) ** 4
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

    sq = 1.0 / (1.0 + 24.0 * volfraction * (g / a_safe))

    if np.any(np.abs(a) < 1e-12):
        sq = np.asarray(sq, dtype=float)
        sq[np.abs(a) < 1e-12] = (1.0 - volfraction) ** 4 / (
            1.0 + 2.0 * volfraction
        ) ** 2

    return sq


def normalize_profile_fractions(raw_weights):
    raw_weights = np.asarray(raw_weights, dtype=float)

    if np.any(raw_weights < 0):
        raise ValueError("cluster weights must be non-negative")

    total = np.sum(raw_weights)
    if total <= 0:
        raise ValueError("at least one cluster weight must be positive")

    return raw_weights / total


def equivalent_volume_radius(semiaxes):
    semiaxes = np.asarray(semiaxes, dtype=float)
    if semiaxes.shape != (3,):
        raise ValueError("semiaxes must contain exactly three values")
    if np.any(semiaxes <= 0):
        raise ValueError("semiaxes must be positive")
    return float(np.cbrt(np.prod(semiaxes)))


def _weight_keys_from_params(params):
    return sorted(
        (key for key in params if key.startswith("w") and key[1:].isdigit()),
        key=lambda key: int(key.lstrip("w")),
    )


def _resolve_effective_radii(weight_keys, params, fallback_effective_radii):
    fallback_radii = np.asarray(fallback_effective_radii, dtype=float)
    resolved_radii: list[float] = []
    for index, weight_key in enumerate(weight_keys):
        sphere_name = f"r_eff_{weight_key}"
        ellipsoid_names = [
            f"a_eff_{weight_key}",
            f"b_eff_{weight_key}",
            f"c_eff_{weight_key}",
        ]
        if sphere_name in params:
            resolved_radii.append(float(params[sphere_name]))
            continue
        if all(name in params for name in ellipsoid_names):
            resolved_radii.append(
                equivalent_volume_radius(
                    [params[name] for name in ellipsoid_names]
                )
            )
            continue
        if index >= len(fallback_radii):
            raise ValueError(
                "Missing geometry parameters and fallback effective_radii "
                f"for component {weight_key}."
            )
        resolved_radii.append(float(fallback_radii[index]))
    return np.asarray(resolved_radii, dtype=float)


def _full_params_to_param_dict(full_params):
    try:
        names = FULL_PARAMETER_NAMES
    except NameError:
        return None
    if len(full_params) != len(names):
        return None
    return {
        str(name): float(full_params[index])
        for index, name in enumerate(names)
    }


def _bounded_solvent_weight(value):
    return float(np.clip(float(value), 0.0, 1.0))


def _effective_structure_factor_profile(
    q_values,
    cluster_intensities,
    effective_radii,
    raw_weights,
    phi_int,
):
    """Return the mixture-equivalent S(q) that modulates the form
    factor."""
    q_values = np.asarray(q_values, dtype=float)
    effective_radii = np.asarray(effective_radii, dtype=float)
    raw_weights = np.asarray(raw_weights, dtype=float)
    fractions = normalize_profile_fractions(raw_weights)
    numerator = np.zeros_like(q_values, dtype=float)
    denominator = np.zeros_like(q_values, dtype=float)
    fallback = np.zeros_like(q_values, dtype=float)

    for frac, iq_cluster, radius in zip(
        fractions, cluster_intensities, effective_radii
    ):
        iq_cluster = np.asarray(iq_cluster, dtype=float)
        sq = calc_hardsphere_sq(radius, phi_int, q_values)
        numerator += frac * iq_cluster * sq
        denominator += frac * iq_cluster
        fallback += frac * sq

    structure_factor = fallback.copy()
    valid_mask = np.abs(denominator) > 1e-12
    structure_factor[valid_mask] = (
        numerator[valid_mask] / denominator[valid_mask]
    )
    return structure_factor


def polydisperse_lma_hs_model(
    q_values,
    cluster_intensities,
    effective_radii,
    solvent_intensities,
    raw_weights,
    phi_solute,
    phi_int,
    solvent_scale,
    scale,
    offset,
):
    q_values = np.asarray(q_values, dtype=float)
    solvent_intensities = np.asarray(solvent_intensities, dtype=float)
    effective_radii = np.asarray(effective_radii, dtype=float)
    raw_weights = np.asarray(raw_weights, dtype=float)

    if len(cluster_intensities) != len(effective_radii):
        raise ValueError(
            "cluster_intensities and effective_radii must have the same "
            "length"
        )

    if len(cluster_intensities) != len(raw_weights):
        raise ValueError(
            "cluster_intensities and raw_weights must have the same length"
        )

    if phi_solute < 0 or phi_solute > 1:
        raise ValueError("phi_solute must satisfy 0 <= phi_solute <= 1")

    fractions = normalize_profile_fractions(raw_weights)
    solute_sum = np.zeros_like(q_values, dtype=float)

    for frac, iq_cluster, radius in zip(
        fractions, cluster_intensities, effective_radii
    ):
        iq_cluster = np.asarray(iq_cluster, dtype=float)
        sq = calc_hardsphere_sq(radius, phi_int, q_values)
        solute_sum += frac * iq_cluster * sq

    solvent_weight = _bounded_solvent_weight(solvent_scale)
    solute_contribution = scale * phi_solute * solute_sum
    solvent_contribution = (
        solvent_weight
        * (1.0 - phi_solute)
        * np.asarray(solvent_intensities, dtype=float)
    )

    return solute_contribution + solvent_contribution + offset


def lmfit_model_profile(
    q, solvent_data, model_data, effective_radii, **params
):
    weight_keys = _weight_keys_from_params(params)
    raw_weights = np.asarray([params[key] for key in weight_keys], dtype=float)
    resolved_effective_radii = _resolve_effective_radii(
        weight_keys,
        params,
        effective_radii,
    )

    return polydisperse_lma_hs_model(
        q_values=q,
        cluster_intensities=model_data,
        effective_radii=resolved_effective_radii,
        solvent_intensities=solvent_data,
        raw_weights=raw_weights,
        phi_solute=params["phi_solute"],
        phi_int=params["phi_int"],
        solvent_scale=params["solvent_scale"],
        scale=params["scale"],
        offset=params["offset"],
    )


def structure_factor_profile(
    q, solvent_data, model_data, effective_radii, **params
):
    """Return the mixture-equivalent structure-factor trace S(q)."""
    del solvent_data
    weight_keys = _weight_keys_from_params(params)
    raw_weights = np.asarray([params[key] for key in weight_keys], dtype=float)
    resolved_effective_radii = _resolve_effective_radii(
        weight_keys,
        params,
        effective_radii,
    )
    return _effective_structure_factor_profile(
        q,
        model_data,
        resolved_effective_radii,
        raw_weights,
        params["phi_int"],
    )


def model_poly_lma_hs(params):
    global q_values
    global theoretical_intensities
    global solvent_intensities
    global effective_radii

    params = np.asarray(params, dtype=float)
    named_params = _full_params_to_param_dict(params)
    if named_params is not None:
        weight_keys = _weight_keys_from_params(named_params)
        raw_weights = np.asarray(
            [named_params[key] for key in weight_keys],
            dtype=float,
        )
        resolved_effective_radii = _resolve_effective_radii(
            weight_keys,
            named_params,
            effective_radii,
        )
        phi_solute = named_params["phi_solute"]
        phi_int = named_params["phi_int"]
        solvent_scale = _bounded_solvent_weight(named_params["solvent_scale"])
        scale = named_params["scale"]
        offset = named_params["offset"]
    else:
        n_profiles = len(theoretical_intensities)
        raw_weights = params[:n_profiles]
        resolved_effective_radii = np.asarray(effective_radii, dtype=float)
        phi_solute = params[n_profiles]
        phi_int = params[n_profiles + 1]
        solvent_scale = _bounded_solvent_weight(params[n_profiles + 2])
        scale = params[n_profiles + 3]
        offset = params[n_profiles + 4]

    return polydisperse_lma_hs_model(
        q_values=q_values,
        cluster_intensities=theoretical_intensities,
        effective_radii=resolved_effective_radii,
        solvent_intensities=solvent_intensities,
        raw_weights=raw_weights,
        phi_solute=phi_solute,
        phi_int=phi_int,
        solvent_scale=solvent_scale,
        scale=scale,
        offset=offset,
    )


def log_likelihood_poly_lma_hs(params):
    global experimental_intensities

    params = np.asarray(params, dtype=float)
    named_params = _full_params_to_param_dict(params)
    if named_params is not None:
        log_sigma = named_params["log_sigma"]
    else:
        n_profiles = len(theoretical_intensities)
        log_sigma = params[n_profiles + 5]
    sigma = np.exp(log_sigma)

    if sigma <= 0:
        return -np.inf

    try:
        model_intensity = model_poly_lma_hs(params)
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
            scale=sigma,
        )
    )

    if n_points == 0:
        return log_likelihood

    return log_likelihood / n_points
