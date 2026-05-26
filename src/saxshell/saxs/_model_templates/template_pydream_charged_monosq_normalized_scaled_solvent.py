import math

import numpy as np
from scipy.stats import norm

# =============================================================
# model_lmfit: lmfit_model_profile
# model_pydream: log_likelihood_charged_monosq_scaled_solvent
# inputs_lmfit: q, solvent_data, model_data, params
# inputs_pydream: q, solvent_data, model_data, params
# param_columns: Structure, Motif, Param, Value, Vary, Min, Max
#
# param: solv_w,1.0,False,0.0,1.0
# param: offset,0,True,-20,30
# param: eff_r,20.75,True,1.0,200.0
# param: vol_frac,0.0192,False,1e-6,0.5
# param: charge,19.0,True,1e-6,200.0
# param: temperature,298.0,False,1.0,450.0
# param: concentration_salt,0.0,False,0.0,5.0
# param: dielectconst,78.0,False,1.0,200.0
# param: scale,5e-4,True,1e-8,5e-3
#
# Charged MonoSQ normalized, scaled-solvent variant:
#   I_raw(q) = sum_i w_i I_i(q) S_RMSA(q) + solv_w * I_solvent(q)
#   I_model(q) = scale * I_raw(q) + offset
#
# The Hayter-Penfold RMSA S(q) implementation below follows the SasView
# sasmodels hayter_msa structure-factor kernel:
# https://www.sasview.org/docs/user/models/hayter_msa.html
# =============================================================

_ELEMENTARY_CHARGE_C = 1.602189e-19
_BOLTZMANN_J_PER_K = 1.380662e-23
_VACUUM_PERMITTIVITY = 8.85418782e-12
_AVOGADRO = 6.022e23


def _validate_hayter_inputs(
    radius_effective,
    volfraction,
    charge,
    temperature,
    concentration_salt,
    dielectconst,
):
    radius_effective = float(radius_effective)
    volfraction = float(volfraction)
    charge = float(charge)
    temperature = float(temperature)
    concentration_salt = float(concentration_salt)
    dielectconst = float(dielectconst)

    if radius_effective <= 0.0:
        raise ValueError("eff_r must be positive")
    if not (0.0 < volfraction < 0.74):
        raise ValueError("vol_frac must satisfy 0 < vol_frac < 0.74")
    if not (0.0 < charge <= 200.0):
        raise ValueError("charge must satisfy 0 < charge <= 200")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if concentration_salt < 0.0:
        raise ValueError("concentration_salt must be non-negative")
    if dielectconst <= 0.0:
        raise ValueError("dielectconst must be positive")

    return (
        radius_effective,
        volfraction,
        charge,
        temperature,
        concentration_salt,
        dielectconst,
    )


def calc_hayter_msa_sq(
    radius_effective,
    volfraction,
    charge,
    temperature,
    concentration_salt,
    dielectconst,
    q_values,
):
    """Return the Hayter-Penfold RMSA charged-sphere structure
    factor."""
    (
        radius_effective,
        volfraction,
        charge,
        temperature,
        concentration_salt,
        dielectconst,
    ) = _validate_hayter_inputs(
        radius_effective,
        volfraction,
        charge,
        temperature,
        concentration_salt,
        dielectconst,
    )
    q_values = np.asarray(q_values, dtype=float)
    return np.asarray(
        [
            _hayter_msa_iq(
                float(q_value),
                radius_effective,
                volfraction,
                charge,
                temperature,
                concentration_salt,
                dielectconst,
            )
            for q_value in q_values
        ],
        dtype=float,
    )


def _hayter_msa_iq(
    q_value,
    radius_effective,
    volfraction,
    charge_number,
    temperature,
    concentration_salt,
    dielectconst,
):
    g = [float(value) for value in range(1, 18)]

    diameter_angstrom = 2.0 * radius_effective
    beta = 1.0 / (_BOLTZMANN_J_PER_K * temperature)
    permittivity = dielectconst * _VACUUM_PERMITTIVITY
    charge_coulomb = charge_number * _ELEMENTARY_CHARGE_C
    diameter_m = diameter_angstrom * 1.0e-10
    particle_volume = (4.0 * math.pi / 3.0) * (diameter_m / 2.0) ** 3
    salt_number_density = concentration_salt * _AVOGADRO * 1.0e3

    ionic_strength = (
        0.5
        * _ELEMENTARY_CHARGE_C
        * _ELEMENTARY_CHARGE_C
        * (charge_number * volfraction / particle_volume)
    )
    ionic_strength += (
        0.5
        * _ELEMENTARY_CHARGE_C
        * _ELEMENTARY_CHARGE_C
        * (2.0 * salt_number_density)
    )
    kappa = math.sqrt(2.0 * beta * ionic_strength / permittivity)

    kappa_diameter = kappa * diameter_m
    g[5] = (
        beta
        * charge_coulomb
        * charge_coulomb
        / (math.pi * permittivity * diameter_m * (2.0 + kappa_diameter) ** 2)
    )
    g[6] = kappa_diameter
    g[4] = volfraction

    ss = g[4] ** (1.0 / 3.0)
    g[9] = 2.0 * ss * g[5] * math.exp(g[6] - g[6] / ss)

    ierr = _sqcoef(0, g)
    if ierr < 0:
        return math.nan
    return _sqhcal(q_value * diameter_angstrom, g)


def _sqcoef(ir, g):
    max_iterations = 40
    accuracy = 5.0e-6
    f1 = 0.0
    f2 = 0.0

    ig = 1
    if g[6] >= (1.0 + 8.0 * g[4]):
        ig = 0
        g[15] = g[14]
        g[16] = g[4]
        ir = _sqfun(1, ir, g)
        g[14] = g[15]
        g[4] = g[16]
        if ir < 0 or g[14] >= 0.0:
            return ir

    g[10] = min(g[4], 0.20)
    if ig != 1 or g[9] >= 0.15:
        ii = 0
        while True:
            ii += 1
            if ii > max_iterations:
                return -1
            if g[10] <= 0.0:
                g[10] = g[4] / ii
            if g[10] > 0.6:
                g[10] = 0.35 / ii

            e1 = g[10]
            g[15] = f1
            g[16] = e1
            ir = _sqfun(2, ir, g)
            if ir < 0:
                return ir
            f1 = g[15]
            e1 = g[16]

            e2 = g[10] * 1.01
            g[15] = f2
            g[16] = e2
            ir = _sqfun(2, ir, g)
            if ir < 0:
                return ir
            f2 = g[15]
            e2 = g[16]

            denominator = f2 - f1
            if denominator == 0.0:
                return -1
            e2 = e1 - (e2 - e1) * f1 / denominator
            g[10] = e2
            delta = abs((e2 - e1) / e1) if e1 != 0.0 else abs(e2 - e1)
            if delta <= accuracy:
                break

        g[15] = g[14]
        g[16] = e2
        ir = _sqfun(4, ir, g)
        if ir < 0:
            return ir
        g[14] = g[15]
        ir = ii
        if ig != 1 or g[10] >= g[4]:
            return ir

    g[15] = g[14]
    g[16] = g[4]
    ir = _sqfun(3, ir, g)
    g[14] = g[15]
    g[4] = g[16]
    if ir >= 0 and g[14] < 0.0:
        ir = -3
    return ir


def _sqfun(ix, ir, g):
    accuracy = 1.0e-6
    max_iterations = 40

    a2 = a3 = b2 = b3 = v2 = v3 = p2 = p3 = 0.0

    reta = g[16]
    eta2 = reta * reta
    eta3 = eta2 * reta
    e12 = 12.0 * reta
    e24 = e12 + e12
    g[13] = (g[4] / g[16]) ** (1.0 / 3.0)
    g[12] = g[6] / g[13]
    ibig = 1 if (g[12] > 15.0 and ix == 1) else 0

    g[11] = g[5] * g[13] * math.exp(g[6] - g[12])
    rgek = g[11]
    rak = g[12]
    ak2 = rak * rak
    ak1 = 1.0 + rak
    dak2 = 1.0 / ak2
    dak4 = dak2 * dak2
    d = 1.0 - reta
    d2 = d * d
    dak = d / rak
    dd2 = 1.0 / d2
    dd4 = dd2 * dd2
    dd45 = dd4 * 2.0e-1
    eta3d = 3.0 * reta
    eta6d = eta3d + eta3d
    eta32 = eta3 + eta3
    eta2d = reta + 2.0
    eta2d2 = eta2d * eta2d
    eta21 = 2.0 * reta + 1.0
    eta22 = eta21 * eta21

    al1 = -eta21 * dak
    al2 = (14.0 * eta2 - 4.0 * reta - 1.0) * dak2
    al3 = 36.0 * eta2 * dak4

    be1 = -(eta2 + 7.0 * reta + 1.0) * dak
    be2 = 9.0 * reta * (eta2 + 4.0 * reta - 2.0) * dak2
    be3 = 12.0 * reta * (2.0 * eta2 + 8.0 * reta - 1.0) * dak4

    vu1 = -(eta3 + 3.0 * eta2 + 45.0 * reta + 5.0) * dak
    vu2 = (eta32 + 3.0 * eta2 + 42.0 * reta - 20.0) * dak2
    vu3 = (eta32 + 30.0 * reta - 5.0) * dak4
    vu4 = vu1 + e24 * rak * vu3
    vu5 = eta6d * (vu2 + 4.0 * vu3)

    ph1 = eta6d / rak
    ph2 = d - e12 * dak2

    ta1 = (reta + 5.0) / (5.0 * rak)
    ta2 = eta2d * dak2
    ta3 = -e12 * rgek * (ta1 + ta2)
    ta4 = eta3d * ak2 * (ta1 * ta1 - ta2 * ta2)
    ta5 = eta3d * (reta + 8.0) * 1.0e-1 - 2.0 * eta22 * dak2

    ex1 = math.exp(rak)
    ex2 = math.exp(-rak) if g[12] < 20.0 else 0.0
    sk = 0.5 * (ex1 - ex2)
    ck = 0.5 * (ex1 + ex2)
    ckma = ck - 1.0 - rak * sk
    skma = sk - rak * ck

    a1 = (e24 * rgek * (al1 + al2 + ak1 * al3) - eta22) * dd4
    if ibig == 0:
        a2 = e24 * (al3 * skma + al2 * sk - al1 * ck) * dd4
        a3 = (
            e24
            * (eta22 * dak2 - 0.5 * d2 + al3 * ckma - al1 * sk + al2 * ck)
            * dd4
        )

    b1 = (1.5 * reta * eta2d2 - e12 * rgek * (be1 + be2 + ak1 * be3)) * dd4
    if ibig == 0:
        b2 = e12 * (-be3 * skma - be2 * sk + be1 * ck) * dd4
        b3 = (
            e12
            * (
                0.5 * d2 * eta2d
                - eta3d * eta2d2 * dak2
                - be3 * ckma
                + be1 * sk
                - be2 * ck
            )
            * dd4
        )

    v1 = (
        eta21 * (eta2 - 2.0 * reta + 10.0) * 2.5e-1 - rgek * (vu4 + vu5)
    ) * dd45
    if ibig == 0:
        v2 = (vu4 * ck - vu5 * sk) * dd45
        v3 = (
            (eta3 - 6.0 * eta2 + 5.0) * d
            - eta6d * (2.0 * eta3 - 3.0 * eta2 + 18.0 * reta + 10.0) * dak2
            + e24 * vu3
            + vu4 * sk
            - vu5 * ck
        ) * dd45

    pp1 = ph1 * ph1
    pp2 = ph2 * ph2
    pp = pp1 + pp2
    p1p2 = ph1 * ph2 * 2.0
    p1 = (rgek * (pp1 + pp2 - p1p2) - 0.5 * eta2d) * dd2
    if ibig == 0:
        p2 = (pp * sk + p1p2 * ck) * dd2
        p3 = (pp * ck + p1p2 * sk + pp1 - pp2) * dd2

    t1 = ta3 + ta4 * a1 + ta5 * b1
    if ibig != 0:
        v3 = (
            (eta3 - 6.0 * eta2 + 5.0) * d
            - eta6d * (2.0 * eta3 - 3.0 * eta2 + 18.0 * reta + 10.0) * dak2
            + e24 * vu3
        ) * dd45
        t3 = ta4 * a3 + ta5 * b3 + e12 * ta2
        t3 += -4.0e-1 * reta * (reta + 10.0) - 1.0
        p3 = (pp1 - pp2) * dd2
        b3 = e12 * (0.5 * d2 * eta2d - eta3d * eta2d2 * dak2 + be3) * dd4
        a3 = e24 * (eta22 * dak2 - 0.5 * d2 - al3) * dd4
        um6 = t3 * a3 - e12 * v3 * v3
        um5 = t1 * a3 + a1 * t3 - e24 * v1 * v3
        um4 = t1 * a1 - e12 * v1 * v1
        lam6 = e12 * p3 * p3
        lam5 = e24 * p1 * p3 - b3 - b3 - ak2
        lam4 = e12 * p1 * p1 - b1 - b1
        w56 = um5 * lam6 - lam5 * um6
        w46 = um4 * lam6 - lam4 * um6
        fa = -w46 / w56
        ca = -fa
        g[3] = fa
        g[2] = ca
        g[1] = b1 + b3 * fa
        g[0] = a1 + a3 * fa
        g[8] = v1 + v3 * fa
        g[14] = -(p1 + p3 * fa)
        g[15] = 0.0 if abs(g[14]) < 1.0e-3 else g[14]
        g[10] = g[16]
    else:
        t2 = ta4 * a2 + ta5 * b2 + e12 * (ta1 * ck - ta2 * sk)
        t3 = ta4 * a3 + ta5 * b3
        t3 += e12 * (ta1 * sk - ta2 * (ck - 1.0))
        t3 += -4.0e-1 * reta * (reta + 10.0) - 1.0

        um1 = t2 * a2 - e12 * v2 * v2
        um2 = t1 * a2 + t2 * a1 - e24 * v1 * v2
        um3 = t2 * a3 + t3 * a2 - e24 * v2 * v3
        um4 = t1 * a1 - e12 * v1 * v1
        um5 = t1 * a3 + t3 * a1 - e24 * v1 * v3
        um6 = t3 * a3 - e12 * v3 * v3

        if ix in {1, 3}:
            lam1 = e12 * p2 * p2
            lam2 = e24 * p1 * p2 - b2 - b2
            lam3 = e24 * p2 * p3
            lam4 = e12 * p1 * p1 - b1 - b1
            lam5 = e24 * p1 * p3 - b3 - b3 - ak2
            lam6 = e12 * p3 * p3

            w16 = um1 * lam6 - lam1 * um6
            w15 = um1 * lam5 - lam1 * um5
            w14 = um1 * lam4 - lam1 * um4
            w13 = um1 * lam3 - lam1 * um3
            w12 = um1 * lam2 - lam1 * um2
            w26 = um2 * lam6 - lam2 * um6
            w25 = um2 * lam5 - lam2 * um5
            w24 = um2 * lam4 - lam2 * um4
            w36 = um3 * lam6 - lam3 * um6
            w35 = um3 * lam5 - lam3 * um5
            w34 = um3 * lam4 - lam3 * um4
            w32 = um3 * lam2 - lam3 * um2
            w46 = um4 * lam6 - lam4 * um6
            w56 = um5 * lam6 - lam5 * um6
            w3526 = w35 + w26
            w3425 = w34 + w25

            w4 = w16 * w16 - w13 * w36
            w3 = 2.0 * w16 * w15 - w13 * w3526 - w12 * w36
            w2 = w15 * w15 + 2.0 * w16 * w14 - w13 * w3425 - w12 * w3526
            w1 = 2.0 * w15 * w14 - w13 * w24 - w12 * w3425
            w0 = w14 * w14 - w12 * w24

            if ix == 1:
                fap = (w14 - w34 - w46) / (w12 - w15 + w35 - w26 + w56 - w32)
            else:
                g[14] = 0.5 * eta2d * dd2 * math.exp(-rgek)
                if 0.0 <= g[11] <= 2.0 and g[12] <= 1.0:
                    e24g = e24 * rgek * math.exp(rak)
                    pwk = math.sqrt(e24g)
                    qpw = (
                        (1.0 - math.sqrt(1.0 + 2.0 * d2 * d * pwk / eta22))
                        * eta21
                        / d
                    )
                    g[14] = -qpw * qpw / e24 + 0.5 * eta2d * dd2
                pg = p1 + g[14]
                ca = ak2 * pg + 2.0 * (b3 * pg - b1 * p3)
                ca += e12 * g[14] * g[14] * p3
                ca = -ca / (ak2 * p2 + 2.0 * (b3 * p2 - b2 * p3))
                fap = -(pg + p2 * ca) / p3

            ii = 0
            while True:
                ii += 1
                if ii > max_iterations:
                    return -2
                fa = fap
                fun = w0 + (w1 + (w2 + (w3 + w4 * fa) * fa) * fa) * fa
                fund = w1 + (2.0 * w2 + (3.0 * w3 + 4.0 * w4 * fa) * fa) * fa
                fap = fa - fun / fund
                delta = abs((fap - fa) / fa) if fa != 0.0 else abs(fap - fa)
                if delta <= accuracy:
                    break

            ir += ii
            fa = fap
            ca = -(w16 * fa * fa + w15 * fa + w14) / (w13 * fa + w12)
            g[14] = -(p1 + p2 * ca + p3 * fa)
            g[15] = 0.0 if abs(g[14]) < 1.0e-3 else g[14]
            g[10] = g[16]
        else:
            ca = ak2 * p1 + 2.0 * (b3 * p1 - b1 * p3)
            ca = -ca / (ak2 * p2 + 2.0 * (b3 * p2 - b2 * p3))
            fa = -(p1 + p2 * ca) / p3
            if ix == 2:
                g[15] = (
                    um1 * ca * ca
                    + (um2 + um3 * fa) * ca
                    + um4
                    + um5 * fa
                    + um6 * fa * fa
                )
            if ix == 4:
                g[15] = -(p1 + p2 * ca + p3 * fa)

        g[3] = fa
        g[2] = ca
        g[1] = b1 + b2 * ca + b3 * fa
        g[0] = a1 + a2 * ca + a3 * fa
        g[8] = (v1 + v2 * ca + v3 * fa) / g[0]

    g24 = e24 * rgek * ex1
    g[7] = (rak * ak2 * g[2] - g24) / (ak2 * g24)
    return ir


def _sqhcal(qq, g):
    etaz = g[10]
    akz = g[12]
    gekz = g[11]
    e24 = 24.0 * etaz
    x1 = math.exp(akz)
    x2 = math.exp(-akz) if g[12] < 20.0 else 0.0
    ck = 0.5 * (x1 + x2)
    sk = 0.5 * (x1 - x2)
    ak2 = akz * akz

    qk = qq / g[13]
    q2k = qk * qk
    if qk <= 1.0e-8:
        return -1.0 / g[0]
    if qk <= 0.01:
        aqk = g[0] * (8.0 + 2.0 * etaz) + 6.0 * g[1] - 12.0 * g[3]
        aqk -= (
            24.0
            * (
                gekz * (1.0 + akz)
                - ck * akz * g[2]
                + g[3] * (ck - 1.0)
                + (g[2] - g[3] * akz) * sk
            )
            / ak2
        )
        aqk += q2k * (
            -((g[0] * (48.0 + 15.0 * etaz) + 40.0 * g[1]) / 60.0)
            + g[3]
            + (4.0 / ak2)
            * (
                gekz * (9.0 + 7.0 * akz)
                + ck * (9.0 * g[3] - 7.0 * g[2] * akz)
                + sk * (9.0 * g[2] - 7.0 * g[3] * akz)
            )
        )
        return 1.0 / (1.0 - g[10] * aqk)

    qk2 = 1.0 / q2k
    qk3 = qk2 / qk
    qqk = 1.0 / (qk * (q2k + ak2))
    sink = math.sin(qk)
    cosk = math.cos(qk)
    asink = akz * sink
    qcosk = qk * cosk

    aqk = g[0] * (sink - qcosk)
    aqk += g[1] * ((2.0 * qk2 - 1.0) * qcosk + 2.0 * sink - 2.0 / qk)
    inter = 24.0 * qk3 + 4.0 * (1.0 - 6.0 * qk2) * sink
    aqk += (
        0.5
        * etaz
        * g[0]
        * (inter - (1.0 - 12.0 * qk2 + 24.0 * qk2 * qk2) * qcosk)
    )
    aqk *= qk3
    aqk += g[2] * (ck * asink - sk * qcosk) * qqk
    aqk += g[3] * (sk * asink - qk * (ck * cosk - 1.0)) * qqk
    aqk += g[3] * (cosk - 1.0) * qk2
    aqk -= gekz * (asink + qcosk) * qqk
    return 1.0 / (1.0 - e24 * aqk)


def _bounded_solvent_weight(value):
    return float(value)


def _weight_keys_from_params(params):
    return sorted(
        (key for key in params if key.startswith("w") and key[1:].isdigit()),
        key=lambda key: int(key[1:]),
    )


def structure_factor_profile(q, solvent_data, model_data, **params):
    """Return the pure charged hard-sphere RMSA structure-factor
    trace."""
    del solvent_data, model_data
    return calc_hayter_msa_sq(
        params["eff_r"],
        params["vol_frac"],
        params["charge"],
        params["temperature"],
        params["concentration_salt"],
        params["dielectconst"],
        np.asarray(q, dtype=float),
    )


def raw_charged_monosq_scaled_solvent_profile(
    q_values,
    solvent_intensities,
    component_intensities,
    weights,
    solv_w,
    eff_r,
    vol_frac,
    charge,
    temperature,
    concentration_salt,
    dielectconst,
):
    """Return the unscaled charged-S(Q) solute plus weighted solvent."""
    q_values = np.asarray(q_values, dtype=float)
    mixture = np.zeros_like(q_values, dtype=float)
    for weight, component in zip(weights, component_intensities):
        mixture += float(weight) * np.asarray(component, dtype=float)

    structure_factor = calc_hayter_msa_sq(
        eff_r,
        vol_frac,
        charge,
        temperature,
        concentration_salt,
        dielectconst,
        q_values,
    )
    solvent_contribution = _bounded_solvent_weight(solv_w) * np.asarray(
        solvent_intensities,
        dtype=float,
    )
    return mixture * structure_factor + solvent_contribution


def charged_monosq_scaled_solvent_profile(
    q_values,
    solvent_intensities,
    component_intensities,
    weights,
    solv_w,
    eff_r,
    vol_frac,
    charge,
    temperature,
    concentration_salt,
    dielectconst,
    scale,
    offset,
):
    """Apply the global scale and offset to the charged MonoSQ model."""
    raw_model = raw_charged_monosq_scaled_solvent_profile(
        q_values,
        solvent_intensities,
        component_intensities,
        weights,
        solv_w,
        eff_r,
        vol_frac,
        charge,
        temperature,
        concentration_salt,
        dielectconst,
    )
    return float(scale) * raw_model + float(offset)


def lmfit_model_profile(q, solvent_data, model_data, **params):
    """Evaluate the charged scaled-solvent MonoSQ SAXS model for
    lmfit."""
    weight_keys = _weight_keys_from_params(params)
    weights = [params[key] for key in weight_keys]

    return charged_monosq_scaled_solvent_profile(
        q,
        solvent_data,
        model_data,
        weights,
        params["solv_w"],
        params["eff_r"],
        params["vol_frac"],
        params["charge"],
        params["temperature"],
        params["concentration_salt"],
        params["dielectconst"],
        params["scale"],
        params["offset"],
    )


def model_charged_monosq_scaled_solvent(params):
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
    charge = params[n_profiles + 4]
    temperature = params[n_profiles + 5]
    concentration_salt = params[n_profiles + 6]
    dielectconst = params[n_profiles + 7]
    scale = params[n_profiles + 8]

    return charged_monosq_scaled_solvent_profile(
        q_values,
        solvent_intensities,
        theoretical_intensities,
        weights,
        solv_w,
        eff_r,
        vol_frac,
        charge,
        temperature,
        concentration_salt,
        dielectconst,
        scale,
        offset,
    )


def log_likelihood_charged_monosq_scaled_solvent(params):
    """Return the normalized Gaussian log-likelihood for pyDREAM."""
    global experimental_intensities

    try:
        model_intensity = model_charged_monosq_scaled_solvent(params)
    except (OverflowError, ValueError, FloatingPointError):
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
