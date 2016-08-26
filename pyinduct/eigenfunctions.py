"""
This modules provides eigenfunctions for a certain set of parabolic problems. Therefore functions for the computation
of the corresponding eigenvalues are included.
The functions which compute the eigenvalues are deliberately separated from the predefined eigenfunctions in
order to handle transformations and reduce effort by the controller implementation.
"""

import numpy as np
import scipy.integrate as si
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from . import utils as ut
from . import placeholder as ph
from .core import Function, back_project_from_base
from .shapefunctions import LagrangeFirstOrder, LagrangeSecondOrder
from .placeholder import FieldVariable, TestFunction
from .visualization import EvalData
from numbers import Number
from functools import partial
import warnings
import copy as cp
import collections
import pyqtgraph as pg


class AddMulFunction(object):
    """
    (Temporary) Function class wich can multiplied with scalars and added with functions.
    Only needed to compute the matrix (of scalars) vector (of functions) product in
    :py:class:`FiniteTransformFunction`. Will be no longer needed when :py:class:`pyinduct.core.Function`
    is overloaded with :code:`__add__` and :code:`__mul__` operator.

    Args:
        function (callable):
    """

    def __init__(self, function):
        self.function = function

    def __call__(self, z):
        return self.function(z)

    def __mul__(self, other):
        return AddMulFunction(lambda z: self.function(z) * other)

    def __add__(self, other):
        return AddMulFunction(lambda z: self.function(z) + other(z))


class FiniteTransformFunction(Function):
    """
    Provide a transformed :py:class:`pyinduct.core.Function` :math:`\\bar x(z)` through the transformation
    :math:`\\bar{\\boldsymbol{\\xi}} = T * \\boldsymbol \\xi`,
    with the function vector :math:`\\boldsymbol \\xi\\in\\mathbb R^{2n}` and
    with a given matrix :math:`T\\in\\mathbb R^{2n\\times 2n}`.
    The operator :math:`*` denotes the matrix (of scalars) vector (of functions)
    product. The interim result :math:`\\bar{\\boldsymbol{\\xi}}` is a vector

    .. math:: \\bar{\\boldsymbol{\\xi}} = (\\bar\\xi_{1,0},...,\\bar\\xi_{1,n-1},\\bar\\xi_{2,0},...,\\bar\\xi_{2,n-1})^T.

    of functions

    .. math::
        &\\bar\\xi_{1,j} = \\bar x(jl_0 + z),\\qquad j=0,...,n-1, \\quad l_0=l/n, \\quad z\\in[0,l_0] \\\\
        &\\bar\\xi_{2,j} = \\bar x(l - jl_0 + z).

    Finally, the provided function :math:`\\bar x(z)` is given through :math:`\\bar\\xi_{1,0},...,\\bar\\xi_{1,n-1}`.

    Note:
        For a more extensive documentation see section 4.2 in:

        - Wang, S. und F. Woittennek: Backstepping-Methode für parabolische Systeme mit punktförmigem inneren
          Eingriff. Automatisierungstechnik, 2015.

          http://dx.doi.org/10.1515/auto-2015-0023

    Args:
        function (callable):
            Function :math:`x(z)` which will subdivided in :math:`2n` Functions

            .. math::
                &\\bar\\xi_{1,j} = x(jl_0 + z),\\qquad j=0,...,n-1, \\quad l_0=l/n, \\quad z\\in[0,l_0] \\\\
                &\\bar\\xi_{2,j} = x(l - jl_0 + z).

            The vector of functions :math:`\\boldsymbol\\xi` consist of these functions:

            .. math:: \\boldsymbol\\xi = (\\xi_{1,0},...,\\xi_{1,n-1},\\xi_{2,0},...,\\xi_{2,n-1})^T) .

        M (numpy.ndarray): Matrix :math:`T\\in\\mathbb R^{2n\\times 2n}` of scalars.
        l (numbers.Number): Length of the domain (:math:`z\in[0,l]`).
    """

    def __init__(self, function, M, l, scale_func=None, nested_lambda=False):

        if not isinstance(function, collections.Callable):
            raise TypeError
        if not isinstance(M, np.ndarray) or len(M.shape) != 2 or np.diff(M.shape) != 0 or M.shape[0] % 1 != 0:
            raise TypeError
        if not all([isinstance(num, (int, float)) for num in [l, ]]):
            raise TypeError

        self.function = function
        self.M = M
        self.l = l
        if scale_func == None:
            self.scale_func = lambda z: 1
        else:
            self.scale_func = scale_func

        self.n = int(M.shape[0] / 2)
        self.l0 = l / self.n
        self.z_disc = np.array([(i + 1) * self.l0 for i in range(self.n)])

        if not nested_lambda:
            # iteration mode
            Function.__init__(self,
                              self._call_transformed_func,
                              nonzero=(0, l),
                              derivative_handles=[])
        else:
            # nested lambda mode
            self.x_func_vec = list()

            for i in range(self.n):
                self.x_func_vec.append(AddMulFunction(
                    partial(lambda z, k: self.scale_func(k * self.l0 + z) * self.function(k * self.l0 + z), k=i)))
            for i in range(self.n):
                self.x_func_vec.append(AddMulFunction(
                    partial(lambda z, k: self.scale_func(self.l - k * self.l0 - z) * self.function(
                        self.l - k * self.l0 - z), k=i)))

            self.y_func_vec = np.dot(self.x_func_vec, np.transpose(M))

            Function.__init__(self, self._call_transformed_func_vec, nonzero=(0, l), derivative_handles=[])

    def _call_transformed_func_vec(self, z):
        i = int(z / self.l0)
        zz = z % self.l0
        if np.isclose(z, self.l0 * i) and not np.isclose(0, zz):
            zz = 0
        return self.y_func_vec[i](zz)

    def _call_transformed_func(self, z):
        i = int(z / self.l0)
        if i < 0 or i > self.n * 2 - 1:
            raise ValueError
        zz = z % self.l0
        if np.isclose(z, self.l0 * i) and not np.isclose(0, zz):
            zz = 0
        to_return = 0
        for j in range(self.n * 2):
            mat_el = self.M[i, j]
            if mat_el != 0:
                if j <= self.n - 1:
                    to_return += mat_el * self.function(j * self.l0 + zz) * self.scale_func(j * self.l0 + zz)
                elif j >= self.n:
                    jj = j - self.n
                    to_return += mat_el * self.function(self.l - jj * self.l0 - zz) * self.scale_func(
                        self.l - jj * self.l0 - zz)
                elif j < 0 or j > 2 * self.n - 1:
                    raise ValueError
        return to_return


class TransformedSecondOrderEigenfunction(Function):
    """
    Provide the eigenfunction :math:`\\varphi(z)` to an eigenvalue problem of the form

    .. math:: a_2(z)\\varphi''(z) + a_1(z)\\varphi'(z) + a_0(z)\\varphi(z) = \\lambda\\varphi(z)

    where :math:`\\lambda` is a predefined (potentially complex) eigenvalue and :math:`[z_0,z_1]\\ni z` is the domain.

    Args:
        target_eigenvalue (numbers.Number): :math:`\\lambda`
        init_state_vect (array_like):
            .. math:: \\Big(\\text{Re}\\{\\varphi(0)\\}, \\text{Re}\\{\\varphi'(0)\\}, \\text{Im}\\{\\varphi(0)\\}, \\text{Im}\\{\\varphi'(0)\\}\\Big)^T
        dgl_coefficients (array_like):
            :math:`\\Big( a2(z), a1(z), a0(z) \\Big)^T`
        domain (array_like):
            :math:`\\Big( z_0, ..... , z_1 \\Big)`
    """

    def __init__(self, target_eigenvalue, init_state_vect, dgl_coefficients, domain):

        if not all([isinstance(state, (int, float)) for state in init_state_vect]) \
            and len(init_state_vect) == 4 and isinstance(init_state_vect, (list, tuple)):
            raise TypeError
        if not len(dgl_coefficients) == 3 and isinstance(dgl_coefficients, (list, tuple)) \
            and all([isinstance(coef, collections.Callable) or isinstance(coef, (int, float)) for coef in
                     dgl_coefficients]):
            raise TypeError
        if not isinstance(domain, (np.ndarray, list)) \
            or not all([isinstance(num, (int, float)) for num in domain]):
            raise TypeError

        if isinstance(target_eigenvalue, complex):
            self._eig_val_real = target_eigenvalue.real
            self._eig_val_imag = target_eigenvalue.imag
        elif isinstance(target_eigenvalue, (int, float)):
            self._eig_val_real = target_eigenvalue
            self._eig_val_imag = 0.
        else:
            raise TypeError

        self._init_state_vect = init_state_vect
        self._a2, self._a1, self._a0 = [ut._convert_to_function(coef) for coef in dgl_coefficients]
        self._domain = domain

        state_vect = self._transform_eigenfunction()
        self._transf_eig_func_real, self._transf_d_eig_func_real = state_vect[0:2]
        self._transf_eig_func_imag, self._transf_d_eig_func_imag = state_vect[2:4]

        Function.__init__(self, self._phi, nonzero=(domain[0], domain[-1]), derivative_handles=[self._d_phi])

    def _ff(self, y, z):
        a2, a1, a0 = [self._a2, self._a1, self._a0]
        wr = self._eig_val_real
        wi = self._eig_val_imag
        d_y = np.array([y[1],
                        -(a0(z) - wr) / a2(z) * y[0] - a1(z) / a2(z) * y[1] - wi / a2(z) * y[2],
                        y[3],
                        wi / a2(z) * y[0] - (a0(z) - wr) / a2(z) * y[2] - a1(z) / a2(z) * y[3]
                        ])
        return d_y

    def _transform_eigenfunction(self):

        eigenfunction = si.odeint(self._ff, self._init_state_vect, self._domain)

        return [eigenfunction[:, 0], eigenfunction[:, 1], eigenfunction[:, 2], eigenfunction[:, 3]]

    def _phi(self, z):
        return np.interp(z, self._domain, self._transf_eig_func_real)

    def _d_phi(self, z):
        return np.interp(z, self._domain, self._transf_d_eig_func_real)


class SecondOrderRobinEigenfunction(Function):
    """
    Provide the eigenfunction :math:`\\varphi(z)` to an eigenvalue problem of the form

    .. math::
        a_2\\varphi''(z) + a_1&\\varphi'(z) + a_0\\varphi(z) = \\lambda\\varphi(z) \\\\
        \\varphi'(0) &= \\alpha \\varphi(0) \\\\
        \\varphi'(l) &= -\\beta \\varphi(l).

    The eigenfrequency

    .. math:: \\omega = \\sqrt{-\\frac{a_1^2}{4a_2^2}+\\frac{a_0-\\lambda}{a_2}}

    must be provided (with :py:class:`compute_rad_robin_eigenfrequencies`).

    Args:
        om (numbers.Number): eigenfrequency :math:`\\omega`
        param (array_like): :math:`\\Big( a_2, a_1, a_0, \\alpha, \\beta \\Big)^T`
        spatial_domain (tuple): Start point :math:`z_0` and end point :math:`z_1` of
            the spatial domain :math:`[z_0,z_1]\\ni z`.
        phi_0 (numbers.Number): Factor to scale the eigenfunctions (correspond :math:`\\varphi(0)=\\text{phi\\_0}`).
    """

    def __init__(self, om, param, spatial_domain, phi_0=1):
        self._om = om
        self._param = param
        self.phi_0 = phi_0
        Function.__init__(self, self._phi, nonzero=spatial_domain, derivative_handles=[self._d_phi, self._dd_phi])

    def _phi(self, z):
        a2, a1, a0, alpha, beta = self._param
        om = self._om
        eta = -a1 / 2. / a2

        cosX_term = np.cos(om * z)
        if not np.isclose(0, np.abs(om), atol=1e-100):
            sinX_term = (alpha - eta) / om * np.sin(om * z)
        else:
            sinX_term = (alpha - eta) * z

        phi_i = np.exp(eta * z) * (cosX_term + sinX_term)

        return return_real_part(phi_i * self.phi_0)

    def _d_phi(self, z):
        a2, a1, a0, alpha, beta = self._param
        om = self._om
        eta = -a1 / 2. / a2

        cosX_term = alpha * np.cos(om * z)
        if not np.isclose(0, np.abs(om), atol=1e-100):
            sinX_term = (eta * (alpha - eta) / om - om) * np.sin(om * z)
        else:
            sinX_term = eta * (alpha - eta) * z - om * np.sin(om * z)

        d_phi_i = np.exp(eta * z) * (cosX_term + sinX_term)

        return return_real_part(d_phi_i * self.phi_0)

    def _dd_phi(self, z):
        a2, a1, a0, alpha, beta = self._param
        om = self._om
        eta = -a1 / 2. / a2

        cosX_term = (eta * (2 * alpha - eta) - om ** 2) * np.cos(om * z)
        if not np.isclose(0, np.abs(om), atol=1e-100):
            sinX_term = ((eta ** 2 * (alpha - eta) / om - (eta + alpha) * om)) * np.sin(om * z)
        else:
            sinX_term = eta ** 2 * (alpha - eta) * z - (eta + alpha) * om * np.sin(om * z)

        d_phi_i = np.exp(eta * z) * (cosX_term + sinX_term)

        return return_real_part(d_phi_i * self.phi_0)


class SecondOrderDirichletEigenfunction(Function):
    """
    Provide the eigenfunction :math:`\\varphi(z)` to an eigenvalue problem of the form

    .. math::
        a_2\\varphi''(z) + a_1&\\varphi'(z) + a_0\\varphi(z) = \\lambda\\varphi(z) \\\\
        \\varphi(0) &= 0 \\\\
        \\varphi(l) &= 0.

    The eigenfrequency

    .. math:: \\omega = \\sqrt{-\\frac{a_1^2}{4a_2^2}+\\frac{a_0-\\lambda}{a_2}}

    must be provided.

    Args:
        om (numbers.Number): eigenfrequency :math:`\\omega`
        param (array_like): :math:`\\Big( a_2, a_1, a_0, None, None \\Big)^T`
        spatial_domain (tuple): Start point :math:`z_0` and end point :math:`z_1` of
            the spatial domain :math:`[z_0,z_1]\\ni z`.
        norm_fac (numbers.Number): Factor to scale the eigenfunctions.
    """

    def __init__(self, omega, param, spatial_domain, norm_fac=1.):
        self._omega = omega
        self._param = param
        self.norm_fac = norm_fac

        a2, a1, a0, _, _ = self._param
        self._eta = -a1 / 2. / a2
        Function.__init__(self, self._phi, nonzero=spatial_domain, derivative_handles=[self._d_phi, self._dd_phi])

    def _phi(self, z):
        eta = self._eta
        om = self._omega

        phi_i = np.exp(eta * z) * np.sin(om * z)

        return return_real_part(phi_i * self.norm_fac)

    def _d_phi(self, z):
        eta = self._eta
        om = self._omega

        d_phi_i = np.exp(eta * z) * (om * np.cos(om * z) + eta * np.sin(om * z))

        return return_real_part(d_phi_i * self.norm_fac)

    def _dd_phi(self, z):
        eta = self._eta
        om = self._omega

        d_phi_i = np.exp(eta * z) * (om * (eta + 1) * np.cos(om * z) + (eta - om ** 2) * np.sin(om * z))

        return return_real_part(d_phi_i * self.norm_fac)


def compute_rad_robin_eigenfrequencies(param, l, n_roots=10, show_plot=False):
    """
    Return the first :code:`n_roots` eigenfrequencies :math:`\\omega` (and eigenvalues :math:`\\lambda`)

    .. math:: \\omega = \\sqrt{-\\frac{a_1^2}{4a_2^2}+\\frac{a_0-\\lambda}{a_2}}

    to the eigenvalue problem

    .. math::
        a_2\\varphi''(z) + a_1&\\varphi'(z) + a_0\\varphi(z) = \\lambda\\varphi(z) \\\\
        \\varphi'(0) &= \\alpha\\varphi(0) \\\\
        \\varphi'(l) &= -\\beta\\varphi(l).

    Args:
        param (array_like): :math:`\\Big( a_2, a_1, a_0, \\alpha, \\beta \\Big)^T`
        l (numbers.Number): Right boundary value of the domain :math:`[0,l]\\ni z`.
        n_roots (int): Amount of eigenfrequencies to be compute.
        show_plot (bool): A plot window of the characteristic equation appears if it is :code:`True`.

    Return:
        tuple --> booth tuple elements are numpy.ndarrays of length :code:`nroots`:
            :math:`\\Big(\\big[\\omega_1,...,\\omega_\\text{n\\_roots}\Big], \\Big[\\lambda_1,...,\\lambda_\\text{n\\_roots}\\big]\\Big)`
    """

    a2, a1, a0, alpha, beta = param
    eta = -a1 / 2. / a2

    def characteristic_equation(om):
        if np.round(om, 200) != 0.:
            zero = (alpha + beta) * np.cos(om * l) + ((eta + beta) * (alpha - eta) / om - om) * np.sin(om * l)
        else:
            zero = (alpha + beta) * np.cos(om * l) + (eta + beta) * (alpha - eta) * l - om * np.sin(om * l)
        return zero

    def complex_characteristic_equation(om):
        if np.round(om, 200) != 0.:
            zero = (alpha + beta) * np.cosh(om * l) + ((eta + beta) * (alpha - eta) / om + om) * np.sinh(om * l)
        else:
            zero = (alpha + beta) * np.cosh(om * l) + (eta + beta) * (alpha - eta) * l + om * np.sinh(om * l)
        return zero

    # assume 1 root per pi/l (safety factor = 3)
    om_end = 3 * n_roots * np.pi / l
    start_values = np.arange(0, om_end, .1)
    om = ut.find_roots(characteristic_equation, 2 * n_roots, start_values, rtol=int(np.log10(l) - 6),
                       show_plot=show_plot).tolist()

    # delete all around om = 0
    om.reverse()
    for i in range(np.sum(np.array(om) < np.pi / l / 2e1)):
        om.pop()
    om.reverse()

    # if om = 0 is a root then add 0 to the list
    zero_limit = alpha + beta + (eta + beta) * (alpha - eta) * l
    if np.round(zero_limit, 6 + int(np.log10(l))) == 0.:
        om.insert(0, 0.)

    # regard complex roots
    om_squared = np.power(om, 2).tolist()
    complex_root = fsolve(complex_characteristic_equation, om_end)
    if np.round(complex_root, 6 + int(np.log10(l))) != 0.:
        om_squared.insert(0, -complex_root[0] ** 2)

    # basically complex eigenfrequencies
    om = np.sqrt(np.array(om_squared).astype(complex))

    if len(om) < n_roots:
        raise ValueError("RadRobinEigenvalues.compute_eigen_frequencies()"
                         "can not find enough roots")

    eig_frequencies = om[:n_roots]
    eig_values = a0 - a2 * eig_frequencies ** 2 - a1 ** 2 / 4. / a2
    return eig_frequencies, eig_values


def return_real_part(to_return):
    """
    Check if the imaginary part of :code:`to_return` vanishes
    and return the real part.

    Args:
        to_return (numbers.Number or array_like): Variable to check.

    Raises:
        ValueError: If (all) imaginary part(s) not vanishes.

    Return:
        numbers.Number or array_like: Real part of :code:`to_return`.
    """
    if not isinstance(to_return, (Number, list, np.ndarray)):
        raise TypeError
    if isinstance(to_return, (list, np.ndarray)):
        if not all([isinstance(num, Number) for num in to_return]):
            raise TypeError

    maybe_real = np.atleast_1d(np.real_if_close(to_return))

    if maybe_real.dtype == 'complex':
        raise ValueError("Something goes wrong, imaginary part does not vanish")
    else:
        if maybe_real.shape == (1,):
            maybe_real = maybe_real[0]
        return maybe_real


def get_adjoint_rad_evp_param(param):
    """
    Return to the eigen value problem of the reaction-advection-diffusion
    equation with robin and/or dirichlet boundary conditions

    .. math::
        a_2\\varphi''(z) + a_1&\\varphi'(z) + a_0\\varphi(z) = \\lambda\\varphi(z) \\\\
        \\varphi(0) = 0 \\quad &\\text{or} \\quad \\varphi'(0) = \\alpha\\varphi(0) \\\\
        \\varphi`(l) = 0 \\quad &\\text{or} \\quad \\varphi'(l) = -\\beta\\varphi(l)

    the parameters for the adjoint problem (with the same structure).

    Args:
        param (array_like): :math:`\\Big( a_2, a_1, a_0, \\alpha, \\beta \\Big)^T`

    Return:
        tuple:
            Parameters :math:`\\big(a_2, \\tilde a_1=-a_1, a_0, \\tilde \\alpha, \\tilde \\beta \\big)` for
            the adjoint problem

            .. math::
                a_2\\psi''(z) + a_1&\\psi'(z) + a_0\\psi(z) = \\lambda\\psi(z) \\\\
                \\psi(0) = 0 \\quad &\\text{or} \\quad \\psi'(0) = \\tilde\\alpha\\psi(0) \\\\
                \\psi`(l) = 0 \\quad &\\text{or} \\quad \\psi'(l) = -\\tilde\\beta\\psi(l).
    """
    a2, a1, a0, alpha, beta = param

    if alpha == None:
        alpha_n = None
    else:
        alpha_n = a1 / a2 + alpha

    if beta == None:
        beta_n = None
    else:
        beta_n = -a1 / a2 + beta
    a1_n = -a1

    return a2, a1_n, a0, alpha_n, beta_n


def transform2intermediate(param, d_end=None):
    """
    Transformation :math:`\\tilde x(z,t)=x(z,t)e^{\\int_0^z \\frac{a_1(\\bar z)}{2 a_2}\,d\\bar z}`
    which eliminate the advection term :math:`a_1 x(z,t)` from the
    reaction-advection-diffusion equation

    .. math:: \\dot x(z,t) = a_2 x''(z,t) + a_1(z) x'(z,t) + a_0(z) x(z,t)

    with robin

    .. math:: x'(0,t) = \\alpha x(0,t), \\quad x'(l,t) = -\\beta x(l,t)

    or dirichlet

    .. math:: x(0,t) = 0, \\quad x(l,t) = 0

    or mixed boundary condition.

    Args:
        param (array_like): :math:`\\Big( a_2, a_1, a_0, \\alpha, \\beta \\Big)^T`

    Raises:
        TypeError: If :math:`a_1(z)` is callable but no derivative handle is defined for it.

    Return:
        tuple:
            Parameters :math:`\\big(a_2, \\tilde a_1=0, \\tilde a_0(z), \\tilde \\alpha, \\tilde \\beta \\big)` for
            the transformed system

            .. math:: \\dot{\\tilde{x}}(z,t) = a_2 \\tilde x''(z,t) + \\tilde a_0(z) \\tilde x(z,t)

            and the corresponding boundary conditions (:math:`\\alpha` and/or :math:`\\beta` set to None by dirichlet
            boundary condition).

    """
    if not isinstance(param, (tuple, list)) or not len(param) == 5:
        raise TypeError("pyinduct.utils.transform_2_intermediate(): argument param must from type tuple or list")

    a2, a1, a0, alpha, beta = param
    if isinstance(a1, collections.Callable) or isinstance(a0, collections.Callable):
        if not len(a1._derivative_handles) >= 1:
            raise TypeError
        a0_z = ut._convert_to_function(a0)
        a0_n = lambda z: a0_z(z) - a1(z) ** 2 / 4 / a2 - a1.derive(1)(z) / 2
    else:
        a0_n = a0 - a1 ** 2 / 4 / a2

    if alpha is None:
        alpha_n = None
    elif isinstance(a1, collections.Callable):
        alpha_n = a1(0) / 2. / a2 + alpha
    else:
        alpha_n = a1 / 2. / a2 + alpha

    if beta is None:
        beta_n = None
    elif isinstance(a1, collections.Callable):
        beta_n = -a1(d_end) / 2. / a2 + beta
    else:
        beta_n = -a1 / 2. / a2 + beta

    a2_n = a2
    a1_n = 0

    return a2_n, a1_n, a0_n, alpha_n, beta_n
