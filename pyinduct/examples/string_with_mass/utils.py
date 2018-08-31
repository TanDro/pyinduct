import pyinduct as pi
import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt

# matplotlib configuration
plt.rcParams.update({'text.usetex': True})


def pprint(expression="\n\n\n"):
    if isinstance(expression, np.ndarray):
        expression = sp.Matrix(expression)
    sp.pprint(expression, num_columns=180)

def get_primal_eigenvector(according_paper=False):
    if according_paper:
        # some condensed parameters
        alpha = beta = sym.c / 2
        tau0 = 1 / sp.sqrt(sym.a * sym.b)
        w = tau0 * sp.sqrt((sym.lam + alpha) ** 2 - beta ** 2)
        # matrix exponential
        expm_A = sp.Matrix([
            [sp.cosh(w * sym.z),
             (sym.lam + sym.c) / sym.b / w * sp.sinh(w * sym.z)],
            [sym.lam / sym.a / w * sp.sinh(w * sym.z),
             sp.cosh(w * sym.z)]
        ])

    else:
        # matrix
        A = sp.Matrix([[sp.Float(0), (sym.lam + sym.c) / sym.b],
                       [sym.lam/sym.a, sp.Float(0)]])
        # matrix exponential
        expm_A = sp.exp(A * sym.z)

    # inital values at z=0 (scaled by xi(s))
    phi0 = sp.Matrix([[sp.Float(1)], [sym.lam / sym.d]])
    # solution
    phi = expm_A * phi0

    return phi

def plot_eigenvalues(eigenvalues):
    plt.figure(facecolor="white")
    plt.scatter(np.real(eigenvalues), np.imag(eigenvalues))
    ax = plt.gca()
    ax.set_xlabel(r"$Re(\lambda)$")
    ax.set_ylabel(r"$Im(\lambda)$")
    plt.show()

def find_eigenvalues(characteristic_equation, n, grid):
    num_char_eq = lambdify(sym.lam, characteristic_equation, modules="numpy")
    eig_vals =  pi.find_roots(
        num_char_eq, grid, n_roots=n, cmplx=True if len(grid) == 2 else False,
    sort_mode="component")
    # two eigenvalues by 0
    eig_vals = np.array(list(eig_vals) + [0])

    return sort_eigenvalues(eig_vals)

def sort_eigenvalues(eigenvalues):
    imag_ev = list()
    real_ev = list()
    for ev in eigenvalues:
        if np.isclose(np.imag(ev), 0):
            real_ev.append(0 if np.isclose(ev, 0) else np.real(ev))
        elif np.imag(ev) > 0:
            imag_ev.append(ev)
            # make sure that the conjugated complex eigenvalue to `ev` exists
            if not sum([np.isclose(_ev, np.conj(ev)) for _ev in eigenvalues]) == 1:
                raise ValueError("Complex conjugate to {} are not found.".format(ev))

    eig_vals = list(np.flipud(sorted(real_ev)))
    for ev in np.array(imag_ev)[np.argsort(np.imag(imag_ev))]:
        eig_vals.append(np.real(ev) + 1j * np.imag(ev))
        eig_vals.append(np.real(ev) - 1j * np.imag(ev))

    if len(eigenvalues) != len(eig_vals):
        raise ValueError(
            "Something went wrong! (only odd number of eigenvalues considered)"
        )

    return np.array(eig_vals)


class Parameters:
    def __init__(self):
        pass


# parameters
param = Parameters()
param.D = 8e-3 # m
param.L = 19.83 # m
param.rho_0 = 1.21 # kg/m^3
param.p_0 = 1.01 # bar
param.T_0 = 293.15 # K
param.R_s = 287.05 # J/kg K
param.epsilon = 1.5e-6 # m
param.gamma = 1.4
param.V_vol = 6.46e-4 # m^3
param.R_vol = 4e-3 # K/W
param.eta_0 = 1.82e-5 # ???

# condensed parameters
param.A = np.pi * param.D ** 2 / 4
param.a_iso = np.sqrt(param.gamma * param.R_s * param.T_0)
param.a = param.L ** -1
param.b = param.a_iso ** 2 / param.L
param.c = 32 * param.eta_0 / param.D ** 2 / param.rho_0
param.d = param.A / param.V_vol

# symbols
sym = Parameters()
sym.a, sym.b, sym.c, sym.d, sym.theta,  sym.z, sym.t = [
    sp.Symbol(sym, real=True) for sym in
    (r"a", r"b", r"c", r"d", r"theta", r"z", r"t")]
sym.lam = sp.symbols(r"lambda")
subs_list = [(sym.a, param.a), (sym.b, param.b),
             (sym.c, param.c), (sym.d, param.d)]

# print parameters
pprint("Sytem parameters:")
pprint(sp.Eq(sym.a, param.a))
pprint(sp.Eq(sym.b, param.b))
pprint(sp.Eq(sym.c, param.c))
pprint(sp.Eq(sym.d, param.d))
pprint()
