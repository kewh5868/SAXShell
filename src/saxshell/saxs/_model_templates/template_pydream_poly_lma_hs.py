import numpy as np
from scipy.stats import norm

# =============================================================
# model_lmfit: lmfit_model_profile
# model_pydream: log_likelihood_poly_lma_hs
# inputs_lmfit: q, solvent_data, model_data, effective_radii, params
# inputs_pydream: q_values, experimental_intensities, solvent_intensities, theoretical_intensities, effective_radii, params
# param_columns: Structure, Motif, Param, Value, Vary, Min, Max
# cluster_geometry_metadata: true
#
# Recommended parameter order:
#   w0, w1, ..., wN-1,
#   phi_solute,
#   phi_int,
#   solvent_scale,
#   scale,
#   offset,
#   log_sigma
#
# param: phi_solute,0.02,True,0.0,0.5
# param: phi_int,0.02,True,0.0,0.4
# param: solvent_scale,1.0,True,0.0,5.0
# param: scale,1.0,True,1e-8,1e8
# param: offset,0.0,True,-1e6,1e6
# param: log_sigma,-9.21,True,-20.0,5.0
#
# Canonical pyDREAM template:
# - normalized Gaussian log-likelihood
# - explicit n_profiles parameter slicing
# - decoupled forward model helper
# - discrete local-monodisperse style cluster sum
# - per-cluster hard-sphere S(Q) using effective radii
# - explicit solvent template, global scale, and offset
#
# Model equation:
#   I_model(q) = scale * [
#       phi_solute * sum_i x_i I_i(q) S_HS(q; R_eff_i, phi_int)
#       + solvent_scale * (1 - phi_solute) * I_solv(q)
#   ] + offset
#
# Internal abundance normalization:
#   x_i = f_i / sum_j f_j
#   f_i >= 0 for all i
#   sum_i x_i = 1 after normalization
#
# Parameter definitions:
#
# w0 ... wN-1
#   Generated cluster-abundance coefficients for the cluster library.
#   SAXSShell creates one weight parameter per averaged cluster profile
#   in md_saxs_map.json and keeps those rows aligned with the component
#   order used in Prefit and DREAM. These are normalized internally
#   into x_i so the fitted values act as relative solute-cluster
#   abundances instead of redundant global scale factors.
#
# phi_solute
#   Physical solute volume fraction in the measured solution. This term
#   scales the cluster contribution relative to the solvent template.
#   Good prior information can come from solution density, composition,
#   and solute/solvent molar masses. Keep this fixed or tightly bounded
#   unless the data are on a credible absolute scale.
#
# phi_int
#   Effective structural volume fraction used only inside the hard-
#   sphere Percus-Yevick structure factor. This is the packing term that
#   controls the strength and position of intercluster interference.
#   It is intentionally distinct from phi_solute because the effective
#   interaction packing seen by S(Q) need not equal the literal solute
#   loading for irregular or anisotropic clusters.
#
# solvent_scale
#   Multiplicative coefficient applied to the experimental solvent SAXS
#   template before the global scale and offset. This absorbs mismatch
#   from transmission, thickness, normalization, or imperfect solvent
#   subtraction.
#
# scale
#   Global multiplicative intensity factor applied to the full model.
#   Keep this free when the measured SAXS data are not on absolute
#   intensity scale or when the cluster I(Q) library is only known up to
#   an arbitrary normalization.
#
# offset
#   Constant additive background term. Use this to absorb fluorescence
#   or other approximately q-independent residual background. Replace
#   with a sloped or polynomial baseline only if the residuals show a
#   clear q dependence that a constant cannot capture.
#
# log_sigma
#   Natural logarithm of the Gaussian noise standard deviation used in
#   the pyDREAM likelihood, with sigma = exp(log_sigma). Fitting the log
#   of sigma enforces positivity and is usually numerically cleaner than
#   fitting sigma directly.
#
# Likelihood convention:
#   The log-likelihood is divided by the number of q points. This keeps
#   the average log-likelihood magnitude more comparable when you change
#   the fitted q-range, interpolation density, or total number of data
#   points between runs.
#
# Required pyDREAM globals:
#   q_values
#       1D experimental q grid.
#   experimental_intensities
#       Experimental SAXS intensities sampled on q_values.
#   solvent_intensities
#       Experimental solvent SAXS template sampled on q_values.
#   theoretical_intensities
#       List or array of cluster I(Q) profiles sampled on q_values.
#   effective_radii
#       One effective interaction radius for each cluster profile.
#
# Practical notes:
# - effective_radii are metadata, not fitted here
# - equivalent-volume sphere radii are a reasonable first choice
# - hard-sphere S(Q) is the default excluded-volume interaction model
# - charged or attractive systems may need a different S(Q)
#
# Relevant resources:
# Pedersen review on SAS modeling and local monodisperse ideas:
# https://neutrons.ornl.gov/sites/default/files/Pedersen97.pdf
#
# SasView structure-factor overview and beta(Q) discussion:
# https://www.sasview.org/docs/user/qtgui/Perspectives/Fitting/
# fitting_sq.html
#
# SasView hard-sphere Percus-Yevick model documentation:
# https://www.sasview.org/docs/user/models/hardsphere.html
#
# SasView Hayter-Penfold RMSA model for charged systems:
# https://www.sasview.org/docs/user/models/hayter_msa.html
# =============================================================


def calc_hardsphere_sq(radius, volfraction, q_values):
    """Return the hard-sphere Percus-Yevick structure factor."""
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
    """Normalize non-negative raw cluster weights into fractions."""
    raw_weights = np.asarray(raw_weights, dtype=float)

    if np.any(raw_weights < 0):
        raise ValueError("cluster weights must be non-negative")

    total = np.sum(raw_weights)
    if total <= 0:
        raise ValueError("at least one cluster weight must be positive")

    return raw_weights / total


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
    """Return the discrete polydisperse LMA hard-sphere SAXS model."""
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
            "cluster_intensities and raw_weights must have the same " "length"
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

    model = phi_solute * solute_sum
    model += solvent_scale * (1.0 - phi_solute) * solvent_intensities

    return scale * model + offset


def lmfit_model_profile(
    q, solvent_data, model_data, effective_radii, **params
):
    """Evaluate the polydisperse LMA hard-sphere model for lmfit."""
    weight_keys = sorted(
        (key for key in params if key.startswith("w")),
        key=lambda key: int(key.lstrip("w")),
    )
    raw_weights = np.asarray([params[key] for key in weight_keys], dtype=float)

    phi_solute = params["phi_solute"]
    phi_int = params["phi_int"]
    solvent_scale = params["solvent_scale"]
    scale = params["scale"]
    offset = params["offset"]

    return polydisperse_lma_hs_model(
        q_values=q,
        cluster_intensities=model_data,
        effective_radii=effective_radii,
        solvent_intensities=solvent_data,
        raw_weights=raw_weights,
        phi_solute=phi_solute,
        phi_int=phi_int,
        solvent_scale=solvent_scale,
        scale=scale,
        offset=offset,
    )


def model_poly_lma_hs(params):
    """Return the forward model intensity for pyDREAM."""
    global q_values
    global theoretical_intensities
    global solvent_intensities
    global effective_radii

    n_profiles = len(theoretical_intensities)

    raw_weights = params[:n_profiles]
    phi_solute = params[n_profiles]
    phi_int = params[n_profiles + 1]
    solvent_scale = params[n_profiles + 2]
    scale = params[n_profiles + 3]
    offset = params[n_profiles + 4]

    return polydisperse_lma_hs_model(
        q_values=q_values,
        cluster_intensities=theoretical_intensities,
        effective_radii=effective_radii,
        solvent_intensities=solvent_intensities,
        raw_weights=raw_weights,
        phi_solute=phi_solute,
        phi_int=phi_int,
        solvent_scale=solvent_scale,
        scale=scale,
        offset=offset,
    )


def log_likelihood_poly_lma_hs(params):
    """Return the normalized Gaussian log-likelihood for pyDREAM."""
    global experimental_intensities

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
