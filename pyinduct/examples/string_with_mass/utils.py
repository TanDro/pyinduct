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

def find_eigenvalues(n):
    def characteristic_equation(om):
        return om * (np.sin(om) + param.m * om * np.cos(om))

    eig_om = pi.find_roots(
        characteristic_equation, np.linspace(0, np.pi * n, 5 * n), n)

    eig_vals = list(sum([(1j * ev, -1j * ev) for ev in eig_om], ()))

    return eig_om, sort_eigenvalues(eig_vals)

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
param.m = 1

# symbols
sym = Parameters()
sym.m, sym.lam, sym.tau, sym.om, sym.theta, sym.z, sym.t, sym.u, sym.yt = [
    sp.Symbol(sym, real=True) for sym in (r"m", r"lambda", r"tau", r"omega", r"theta", r"z", r"t", r"u", r"\tilde{y}")]
subs_list = [(sym.m, param.m)]

# print parameters
pprint("Sytem parameters:")
pprint(sp.Eq(sym.m, param.m))
pprint()
