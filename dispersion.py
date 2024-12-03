import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import scipy.optimize as opt
import plasma_dispersion as pd


def dispersion_function(k, z, drift_one, vt_one, two_scale, drift_two, vt_two):
    """
    Computes two-species plasma dispersion function epsilon(zeta, k) = 0
    """
    sq2 = 2 ** 0.5
    k2 = k / sq2
    k_sq = k ** 2.0
    z_e_one = (z - drift_one) / vt_one / sq2
    z_e_two = (z - drift_two) / vt_two / sq2
    return 1 - 0.5 * (pd.Zprime(z_e_one) +
                      two_scale * pd.Zprime(z_e_two) * (vt_one / vt_two) ** 2) / k_sq / (1 + two_scale)  #


def analytic_jacobian(k, z, drift_one, vt_one, two_scale, drift_two, vt_two):
    sq2 = 2 ** 0.5
    k2 = k / sq2
    k_sq = k ** 2.0
    z_e_one = (z - drift_one) / vt_one / sq2
    z_e_two = (z - drift_two) / vt_two / sq2
    return -0.5 * (pd.Zdoubleprime(z_e_one) / vt_one +
                   two_scale * pd.Zdoubleprime(z_e_two) / vt_two) / k_sq / (1 + two_scale)


def dispersion_fsolve(z, k, drift_one, vt_one, two_scale, drift_two, vt_two):
    freq = z[0] + 1j * z[1]
    d = dispersion_function(k, freq, drift_one, vt_one, two_scale, drift_two, vt_two)
    return [np.real(d), np.imag(d)]


def jacobian_fsolve(z, k, drift_one, vt_one, two_scale, drift_two, vt_two):
    freq = z[0] + 1j * z[1]
    jac = analytic_jacobian(k, freq, drift_one, vt_one, two_scale, drift_two, vt_two)
    jr, ji = np.real(jac), np.imag(jac)
    return [[jr, -ji], [ji, jr]]


def dispersion_function_two_species(k, z, mass_ratio, temperature_ratio, electron_drift):
    """
    Computes two-species plasma dispersion function epsilon(zeta, k) = 0
    """
    thermal_velocity_ratio = np.sqrt(temperature_ratio / mass_ratio)
    k_sq = k ** 2.0
    z_e = z - electron_drift / np.sqrt(2)
    z_p = thermal_velocity_ratio * z
    return 1.0 - (pd.Zprime(z_e) + temperature_ratio * pd.Zprime(z_p)) / k_sq


def analytic_jacobian_two_species(k, z, mass_ratio, temperature_ratio, electron_drift):
    thermal_velocity_ratio = np.sqrt(temperature_ratio / mass_ratio)
    k_sq = k ** 2.0
    z_e = z - electron_drift / np.sqrt(2)
    z_p = thermal_velocity_ratio * z
    fe = 1
    fp = thermal_velocity_ratio
    return -0.5 * (pd.Zdoubleprime(z_e) / fe + temperature_ratio * pd.Zdoubleprime(z_p) / fp) / k_sq


def dispersion_fsolve_two_species(z, k, mass_ratio, temperature_ratio, electron_drift):
    freq = z[0] + 1j * z[1]
    d = dispersion_function_two_species(k, freq, mass_ratio, temperature_ratio, electron_drift)
    return [np.real(d), np.imag(d)]


def jacobian_fsolve_two_species(z, k, mass_ratio, temperature_ratio, electron_drift):
    freq = z[0] + 1j * z[1]
    jac = analytic_jacobian(k, freq, mass_ratio, temperature_ratio, electron_drift)
    jr, ji = np.real(jac), np.imag(jac)
    return [[jr, -ji], [ji, jr]]


if __name__ == '__main__':
    # parameters
    #mr = 1 / 1836  # / 10  # 1836  # me / mp
    #tr = 1.0  # Te / Tp
    k = 0.1

    e_d = 0  # 3.0

    # grid
    om_r = np.linspace(-0.1, 0.1, num=500)
    om_i = np.linspace(-0.1, 0.1, num=500)

    k_scale = k

    zr = om_r / k_scale
    zi = om_i / k_scale

    z = np.tensordot(zr, np.ones_like(zi), axes=0) + 1.0j * np.tensordot(np.ones_like(zr), zi, axes=0)

    X, Y = np.meshgrid(om_r, om_i, indexing='ij')

    # eps = dispersion_function(k, z, mr, tr, e_d)
    chi = 1
    vb = 4
    vtb = chi ** (1 / 3) * vb
    eps = dispersion_function(k, z, drift_one=0, vt_one=1, two_scale=chi, drift_two=vb, vt_two=vtb)
    eps = dispersion_function(k, z, drift_one=5, vt_one=1, two_scale=1, drift_two=-5, vt_two=1)
    cb = np.linspace(-1, 1, num=100)

    guess_r, guess_i = 0.1 / k, 0.1 / k
    solution = opt.root(dispersion_fsolve, x0=np.array([guess_r, guess_i]),
                        args=(k, 0, 1, chi, vb, vtb), jac=jacobian_fsolve)
    print(solution.x * k_scale)


    # Visualize the modes
    x = np.linspace(-500, 500, num=2000)
    v = np.linspace(-3, 7, num=1000)
    X, V = np.meshgrid(x, v, indexing='ij')
    df = -V * np.exp(-0.5 * V ** 2.0) - chi * (V - vb) * np.exp(-0.5 * (V - vb) ** 2.0 / vtb ** 2.0) / (vtb ** 3.0)
    plt.imshow(df)
    plt.show()