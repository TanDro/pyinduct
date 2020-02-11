import warnings
import sympy as sp
import numpy as np
import sys
from tqdm import tqdm
import collections
import pyinduct as pi
from pyinduct.core import (domain_intersection, integrate_function, get_base,
                           get_transformation_info, get_weight_transformation)
from pyinduct.simulation import simulate_state_space, SimulationInput
from sympy.utilities.lambdify import implemented_function
from abc import ABC, abstractmethod


__all__ = ["VariablePool", "NumericalCauchyIntegration",
           "transform_to_diagonal_sys"]


class VariablePool:
    registry = dict()

    def __init__(self, description):
        if description in self.registry:
            raise ValueError("Variable pool '{}' already exists.".format(description))

        self.registry.update({description: self})
        self.description = description
        self.variables = dict()
        self.categories = dict()
        self.categories.update({None: list()})

    def __getitem__(self, item):
        if len(self.categories[item]) == 0:
            return None

        elif len(self.categories[item]) == 1:
            return self.categories[item][0]

        else:
            return self.categories[item]

    def _new_variable(self, name, dependency, implementation, category, **kwargs):
        assert isinstance(name, str)

        if name in self.variables:
            raise ValueError("Name '{}' already in variable pool.".format(name))

        if dependency is None and implementation is None:
            variable = sp.Symbol(name, **kwargs)

        elif implementation is None:
            assert isinstance(dependency, collections.Iterable)
            variable = sp.Function(name, **kwargs)(*dependency)

        elif callable(implementation):
            variable = implemented_function(name, implementation, **kwargs)(*dependency)

        else:
            raise NotImplementedError

        self.variables.update({name: variable})

        if category not in self.categories:
            self.categories.update({category: list()})

        self.categories[category].append(variable)

        return variable

    def _new_variables(self, names, dependencies, implementations, category, **kwargs):
        assert isinstance(names, collections.Iterable)

        if dependencies is None:
            dependencies = [None] * len(names)
        if implementations is None:
            implementations = [None] * len(names)

        assert len(names) == len(dependencies)
        assert len(names) == len(implementations)

        variables = list()
        for name, dependency, implementation in zip(names, dependencies, implementations):
            variables.append(self._new_variable(name, dependency, implementation, category, **kwargs))

        return variables

    def new_symbol(self, name, category, **kwargs):
        return self._new_variable(name, None, None, category, **kwargs)

    def new_symbols(self, names, category, **kwargs):
        return self._new_variables(names, None, None, category, **kwargs)

    def new_function(self, name, dependency, category, **kwargs):
        return self._new_variable(name, dependency, None, category, **kwargs)

    def new_functions(self, names, dependencies, category, **kwargs):
        return self._new_variables(names, dependencies, None, category, **kwargs)

    def new_implemented_function(self, name, dependency, implementation, category, **kwargs):
        return self._new_variable(name, dependency, implementation, category, **kwargs)

    def new_implemented_functions(self, names, dependencies, implementations, category, **kwargs):
        return self._new_variables(names, dependencies, implementations,
                                   category, **kwargs)


global_variable_pool = VariablePool("GLOBAL")

dummy_counter = 0


def new_dummy_variable(dependcy, implementation, **kwargs):
    global dummy_counter

    name = "_pyinduct_dummy{}".format(dummy_counter)
    dummy_counter += 1

    return global_variable_pool._new_variable(
        name, dependcy, implementation, "dummies", **kwargs)


def new_dummy_variables(dependcies, implementations, **kwargs):
    dummies = list()

    for dependcy, implementation in zip(dependcies, implementations):
        dummies.append(new_dummy_variable(dependcy, implementation, **kwargs))

    return dummies


def pprint(expr, description=None, n=None, limit=4, num_columns=180,
           discard_small_values=False, tolerance=1e-6):
    """
    Wraps sympy.pprint, adds description to the console output
    (if given) and the availability of hiding the output if
    the approximation order exceeds a given limit.

    Args:
        expr (sympy.Expr or array-like): Sympy expression or list of sympy
            expressions to pprint.
        description (str): Description of the sympy expression to pprint.
        n (int): Current approximation order, default None, means
            :code:`limit` will be ignored.
        limit (int): Limit approximation order, default 4.
        num_columns (int): Kwarg :code:`num_columns` of sympy.pprint,
            default 180.
        discard_small_values (bool): If true: round numbers < tolerance to 0.
            Default: false.
        tolerance (float): Applies when discard_small_values is true.
            Default: 1e-6.
    """
    if n is not None and n > limit:
        return

    else:
        if description is not None:
            print("\n>>> {}".format(description))

        if discard_small_values:
            # this is not clever or perfomant, but short
            expr = sp.nsimplify(expr, tolerance=tolerance, rational=True).n()

        sp.pprint(expr, num_columns=num_columns)


class SimulationInputWrapper:
    """
    Wraps a :py:class:`.SimulationInput` into a callable, for further use
    as sympy implemented function (input function) and call during
    the simulation, see :py:class:`.simulate_system`.
    """
    def __init__(self, sim_input):
        assert isinstance(sim_input, SimulationInput)

        self._sim_input = sim_input

    def __call__(self, kwargs):
        return self._sim_input(**kwargs)


class Feedback(SimulationInput):

    def __init__(self, expression, base_weights_info, name=str(), args=None):
        SimulationInput.__init__(self, name=name)

        self.feedback_gains = dict()
        for lbl, vec in base_weights_info.items():
            gain, _expression = sp.linear_eq_to_matrix(sp.Matrix([expression]), list(vec))
            expression = (-1) * _expression
            self.feedback_gains.update({lbl: np.array(gain).astype(float)})

        self.remaining_terms = None
        if not expression == expression * 0:
            if args is None:
                raise ValueError("The feedback law holds variables, which "
                                 "could not be sort into the linear feedback "
                                 "gains. Provide the weights variable 'weights' "
                                 "and the time variabel 't' as tuple over the "
                                 "'args' argument.")

            # TODO: check that 'expression' depends only on 'args'
            elif False:
                pass

            else:
                self.remaining_terms = sp.lambdify(args, expression, "numpy")

        self.feedback_gain_sum = dict()


    def _calc_output(self, **kwargs):
        """
        Calculates the controller output based on the current_weights and time.

        Keyword Args:
            weights: Current weights of the simulations system approximation.
            weights_lbl (str): Corresponding label of :code:`weights`.
            time (float): Current simulation time.

        Return:
            dict: Controller output :math:`u`.
        """


        # determine sum over feedback gains
        if kwargs["weight_lbl"] not in self.feedback_gain_sum:
            self.feedback_gain_sum[kwargs["weight_lbl"]] = \
                self.evaluate_feedback_gain_sum(self.feedback_gains,
                                kwargs["weight_lbl"],
                                (1, len(kwargs["weights"])))

        # linear feedback u = k^T * x
        res = self.feedback_gain_sum[kwargs["weight_lbl"]] @ kwargs["weights"]

        # add constant, nonlinear and other crazy terms
        if self.remaining_terms is not None:
            res += self.remaining_terms(kwargs["weights"], kwargs["time"])

        return dict(output=res)

    @staticmethod
    def evaluate_feedback_gain_sum(gains, weight_label, vect_shape):
        r"""
        Transform the different feedback gains in `ce` to the basis
        `weight_label` and accumulate them to one gain vector.
        For weight transformations the procedure is straight forward.
        If the feedback gain :math:`u(t) = k^Tc(t)` was approximated with respect
        to the weights from the state
        :math:`x(z,t) = \sum_{i=1}^{n}c_i(t)\varphi_i(z)`
        but during the simulation only the weights from base
        :math:`\bar{x}(z,t) = \sum_{i=1}^{m} \bar{c}_i(t)\varphi_i(z)`
        are available a weights transformation
        .. math::
            :nowrap:
            \begin{align*}
              c(t) = N^{-1}M\bar{c}(t), \qquad
              N_{(i,j)} = \langle \varphi_i(z), \varphi_j(z) \rangle, \qquad
              M_{(i,j)} = \langle \varphi_i(z), \bar{\varphi}_j(z) \rangle
            \end{align*}
        will be computed.

        Args:
            gains (dict): Dictionary of all feedback gains.
            weight_label (string): Label of functions the weights correspond to.
            vect_shape (tuple): Shape of the feedback vector.

        Return:
            :class:`numpy.array`: Accumulated feedback/observer gain.
        """
        gain_sum = np.zeros(vect_shape)
        identity = np.eye(max(vect_shape))

        for lbl, gain in gains.items():
            # collect information
            org_order = 0
            tar_order = 0
            info = get_transformation_info(
                weight_label,
                lbl,
                tar_order,
                org_order)

            # fetch handle
            transformation = get_weight_transformation(info)

            # evaluate
            for i, iv in enumerate(identity):
                gain_sum[0, i] += np.dot(gain, transformation(iv))

        return gain_sum


def simulate_system(rhs, funcs, init_conds, base_label, input_syms,
                    time_sym, temp_domain, settings=None):
    r"""
    Simulate finite dimensional ode according to the provided
    right hand side (:code:`rhs`)

    .. math:: \partial_t c(t) = f(c(t), u(t))

    Args:
        rhs (sympy.Matrix): Vector :math:`f(c(t), u(t))`
        funcs (sympy.Matrix): Vector: :math:`c(t)`
        init_conds (array-like): Vector:
            :math:`c(t_0), \quad t_0 = \text{temp_domain[0]}`
        base_label (str): Label of a finite dimension base
            :math:`\varphi_i, i=1,...,n` which is registered with the module
            :py:mod:`pyinduct.registry`.
        input_syms (array-like): List of system input symbols/
            implemented functions :math:`u(t)`, see
            :py:class:`.SimulationInputWrapper`.
        time_sym (sympy.Expr): Symbol the variable :math:`t`.
        temp_domain (.Domain): Temporal domain.
        **settings: Kwargs will be passed through to scipy.integrate.ode.

    Returns:
        See :py:func:`.simulate_state_space`.
    """
    # check if all simulation input symbols have only one
    # depended variable and uniqueness of it
    input_arg = input_syms[0].args[0]
    assert all([len(sym.args) == 1 for sym in input_syms])
    assert all([input_arg == sym.args[0] for sym in input_syms])

    # check length of args
    n = len(pi.get_base(base_label))
    assert all([n == len(it) for it in [init_conds, funcs]])

    # check if all inputs holds an SimulationInputWrapper as implementation
    assert all(isinstance(inp._imp_, SimulationInputWrapper) for inp in list(input_syms))

    # dictionary / kwargs for the pyinuct simulation input call
    _input_var = dict(time=0, weights=init_conds, weight_lbl=base_label)

    # derive callable from the symbolic expression of the right hand side
    print("\n>>> lambdify right hand side")
    rhs_lam = sp.lambdify((funcs, time_sym, input_arg), rhs, modules="numpy")
    assert len(rhs_lam(init_conds, 0, _input_var)) == n

    def _rhs(_t, _q):
        _input_var["time"] = _t
        _input_var["weights"] = _q

        return rhs_lam(_q, _t, _input_var)

    return simulate_state_space(_rhs, init_conds, temp_domain, settings)


def evaluate_implemented_functions(expression):

    der_replace_dict = dict()
    for der in expression.atoms(sp.Derivative):

        # only derivatives will be processed which holds
        # exact one sympy function
        if len(der.atoms(sp.Function)) != 1:
            continue

        # skip if the function is not the only argument of the derivative
        func = der.atoms(sp.Function).pop()
        if der.args[0] != func:
            continue

        # skip if the function has no implementation
        if not hasattr(func, "_imp_"):
            continue

        # skip if the function has more or less than one dependent variable
        if len(func.args) != 1:
            continue

        # determine derivative order
        der_order = get_derivative_order(der)

        imp = func._imp_
        if isinstance(imp, pi.Function):
            new_imp = imp.derive(der_order)
            dummy_der = new_dummy_variable(func.args, new_imp)
            der_replace_dict.update({der: dummy_der})

        elif callable(imp):
            raise NotImplementedError(
                "Only derivatives of a pyinduct.Function "
                "can be aquired.")

        else:
            raise NotImplementedError

    # replace all derived implemented pyinduct functions
    # with a dummy function which holds the derivative as implementation
    expr_without_derivatives = expression.xreplace(der_replace_dict)

    # evaluate if possible
    evaluated_expression = expr_without_derivatives.doit().n()

    # undo replace if the derivative could not be evaluated
    reverse_replace = dict([(v, k) for k, v in der_replace_dict.items()])

    return evaluated_expression.xreplace(reverse_replace)


def evaluate_integrals(expression):
    expr_expand = expression.expand()

    replace_dict = dict()
    for integral in tqdm(expr_expand.atoms(sp.Integral),
                         desc=">>> evaluate integrals", file=sys.stdout):
        if not len(integral.args[1]) == 3:
            raise ValueError(
                "Only the evaluation of definite integrals is implemented.")
        integrand = integral.args[0]
        dependent_var, limit_a, limit_b = integral.args[1]
        all_funcs = integrand.atoms(sp.Function)
        impl_funcs = {func for func in all_funcs if hasattr(func, "_imp_")}

        if len(impl_funcs) == 0:
            replace_dict.update({integral: integral.doit()})

        elif isinstance(integrand, (sp.Mul, sp.Function, sp.Derivative)):

            constants = list()
            dependents = list()
            if isinstance(integrand, sp.Mul):
                for arg in integrand.args:
                    if dependent_var in arg.free_symbols:
                        dependents.append(arg)

                    else:
                        constants.append(arg)

            elif isinstance(integrand, (sp.Function, sp.Derivative)):
                dependents.append(integrand)

            else:
                raise NotImplementedError

            assert len(dependents) != 0
            assert np.prod([sym for sym in constants + dependents]) == integrand

            # collect numeric implementation of all
            # python and pyinduct functions
            py_funcs = list()
            pi_funcs = list()
            prove_integrand = sp.Integer(1)
            domain = {(float(limit_a), float(limit_b))}
            prove_replace = dict()
            for func in dependents:

                # check: maximal one free symbol
                free_symbol = func.free_symbols
                assert len(free_symbol) <= 1

                # check: free symbol is the integration variable
                if len(free_symbol) == 1:
                    assert free_symbol.pop() == dependent_var

                # if this term is not a function try to lambdify the function
                # and implement the lambdified function
                if len(func.atoms(sp.Function)) == 0:
                    lam_func = sp.lambdify(dependent_var, func)
                    orig_func = func
                    func = new_dummy_variable((dependent_var,), lam_func)
                    prove_replace.update({func: orig_func})

                # check: only one sympy function in expression
                _funcs = func.atoms(sp.Function)
                assert len(_funcs) == 1

                # check: only one dependent variable
                _func = _funcs.pop()
                assert len(_func.args) == 1

                # check: correct dependent variable
                assert _func.args[0] == dependent_var

                # determine derivative order
                if isinstance(func, sp.Derivative):
                    der_order = get_derivative_order(func)

                else:
                    der_order = 0

                # for a semantic check
                prove_integrand *= sp.diff(_func, dependent_var, der_order)

                # categorize _imp_ in python and pyinduct functions
                implementation = _func._imp_
                if isinstance(implementation, pi.Function):
                    domain = domain_intersection(domain, implementation.nonzero)
                    pi_funcs.append((implementation, int(der_order)))

                elif callable(implementation):
                    if der_order != 0:
                        raise NotImplementedError(
                            "Only derivatives of a pyinduct.Function "
                            "can be aquired.")

                    py_funcs.append(implementation)

                else:
                    raise NotImplementedError

            # check if things will be processed correctly
            prove_integrand = np.prod(
                [sym for sym in constants + [prove_integrand]])
            assert sp.Integral(
                prove_integrand, (dependent_var, limit_a, limit_b)
            ).xreplace(prove_replace) == integral

            # function to integrate
            def _integrand(z, py_funcs=py_funcs, pi_funcs=pi_funcs):
                mul = ([f(z) for f in py_funcs] +
                       [f.derive(ord)(z) for f, ord in pi_funcs])

                return np.prod(mul)

            _integral = integrate_function(_integrand, domain)[0]
            result = np.prod([sym for sym in constants + [_integral]])

            replace_dict.update({integral: result})

        else:
            raise NotImplementedError

    return expr_expand.xreplace(replace_dict)


def derive_first_order_representation(expression, funcs, input_,
                                      mode="sympy.solve",
                                      interim_results=None):

    # make sure funcs depends on one varialble only
    assert len(funcs.free_symbols) == 1
    depvar = funcs.free_symbols.pop()

    if mode == "sympy.solve":
        # use sympy solve for rewriting
        print("\n>>> rewrite  as c' = f(c,u)")
        sol = sp.solve(expression, sp.diff(funcs, depvar))
        rhs = sp.Matrix([sol[it] for it in sp.diff(funcs, depvar)])

        return rhs

    elif mode == "sympy.linear_eq_to_matrix":
        # rewrite expression as E1 * c' + E0 * c + G * u = 0
        print("\n>>> rewrite as E1 c' + E0 c + G u = 0")
        E1, _expression = sp.linear_eq_to_matrix(expression,
                                                 list(sp.diff(funcs, depvar)))
        expression = (-1) * _expression
        E0, _expression = sp.linear_eq_to_matrix(expression, list(funcs))
        expression = (-1) * _expression
        G, _expression = sp.linear_eq_to_matrix(expression, list(input_))
        assert _expression == _expression * 0

        # rewrite expression as c' = A c + B * u
        print("\n>>> rewrite as c' = A c + B u")
        if len(E1.atoms(sp.Symbol, sp.Function)) == 0:
            E1_num = np.array(E1).astype(float)
            E1_inv = sp.Matrix(np.linalg.inv(E1_num))

        else:
            warnings.warn("Since the matrix E1 depends on symbol(s) and/or \n"
                          "function(s) the method sympy.Matrix.inv() was \n"
                          "used. Check result! (numpy.linalg.inv() is more \n"
                          "reliable)")
            E1_inv = E1.inv()

        A = -E1_inv * E0
        B = -E1_inv * G

        if interim_results is not None:
            interim_results.update({
                "E1": E1, "E0": E0, "G": G, "A": A, "B": B,
            })

        return A * funcs + B * input_


def implement_as_linear_ode(rhs, funcs, input_):

    A, _rhs = sp.linear_eq_to_matrix(rhs, list(funcs))
    _rhs *= -1
    B, _rhs = sp.linear_eq_to_matrix(_rhs, list(input_))
    assert _rhs == _rhs * 0
    assert len(A.atoms(sp.Symbol, sp.Function)) == 0
    assert len(B.atoms(sp.Symbol, sp.Function)) == 0

    A_num = np.array(A).astype(float)
    B_num = np.array(B).astype(float)

    def __rhs(c, u):
        assert u.shape[2] == 1
        return A_num @ c + B_num @ u[:, :, 0]

    return new_dummy_variable((funcs, input_), __rhs)


def get_derivative_order(derivative):
    # its really a derivative?
    assert isinstance(derivative, sp.Derivative)

    der_order = derivative.args[1][1]

    # its really a integer value?
    assert der_order == int(der_order)

    return int(der_order)


class NumericalCauchyIntegration(ABC):
    """
    Class for a numerical integration of the cauchy problem

    Usable for a system with two states.
    """
    def __init__(self, eta_traj, z0, zT, N):
        """
        Initialization of the class.

        :param eta_traj: pyinduct.trajectory.SmoothTransition to plan a
        transition between to static states
        :param z0: beginning point of the integration (mostly = 0)
        :param zT: end point of the integration (mostly = length)
        :param N: number of integration points
        """
        # initialize the ABC class
        super().__init__()
        # initialize variables
        self.N = N
        self.dz = np.float64(np.abs(zT - z0)/N)
        self.eta_traj = eta_traj
        self.z0 = z0
        self.zT = zT
        self.dt = 0
        self.k = 0
        self.T0 = 0
        self.T1 = 0
        self.ti = 0
        self.tf = 0
        self.dT = 0
        self.time = [0, 0]
        self.space = [0, 0]

    @abstractmethod
    def _lmda(self, i, z=None):
        """
        Eigenvalue function of the transformed system.

        :param i: use first or second eigenvalue
        :param z: if lambda is a function of z, it will evaluated
        :return: a numerical value at the position z or constant if z=None
        """
        pass

    @abstractmethod
    def _char(self, i, t0, z0, z):
        """
        Evaluate the characteristic of the system.

        :param i: use first or second characteristic
        :param t0: start point in time
        :param z0: start point of the char in z
        :param z: end point of the char in z
        :return: return a numerical value for the evaluated char at z
        """
        pass

    @abstractmethod
    def _mat_C(self, z=None):
        r"""
        Matrix of the remaining part of the system.

        .. math::
            \frac{\partial \pmb{x}(z, t)}{\partial z} + \Lambda(z)
            \frac{\partial \pmb{x}(z, t)}{\partial t} = C(z) \pmb{x}(z, t)

        :param z: point where the matrix should get evaluated
        :return: numpy array with 2x2 dimension
        """
        pass

    @abstractmethod
    def _mat_T(self, z=None):
        """
        Matrix of the transformation into the new states.

        :param z: point where the matrix should get evaluated
        :return: numpy array with 2x2 dimension
        """
        pass

    @abstractmethod
    def _mat_Tinv(self, z=None):
        """
        Matrix of the inverse transformation into the original states.

        :param z: point where the matrix should get evaluated
        :return: numpy array with 2x2 dimension
        """
        pass

    def _qvec(self, zeta1, zeta2, z=None):
        r"""
        Vector for the calculation of next zeta point (like lambda for tau).

        It's defined over the remaining part on the right side of the transformed system, with:

        .. math::
            \pmb{q}(z) = C(z)\pmb{x}(z, t)

        .. math::
            \frac{\partial \pmb{x}(z, t)}{\partial z} + \Lambda(z)
            \frac{\partial \pmb{x}(z, t)}{\partial t} = C(z) \pmb{x}(z, t)

        :param zeta1: first state of current point
        :param zeta2: second state of current point
        :param z: point on which the matrix multiplication should take place
        :return: the results for both states
        """
        if not isinstance(zeta1, np.ndarray):
            raise NotImplementedError(
                "zeta1 should be a state vector over a time transition")
        if not isinstance(zeta2, np.ndarray):
            raise NotImplementedError(
                "zeta2 should be a state vector over a time transition")
        if len(zeta1) != len(zeta2):
            raise NotImplementedError(
                "zeta1 and zeta2 need to have the same dimension")
        zeta = np.vstack((zeta1, zeta2))
        matC = self._mat_C(z)
        q1 = np.zeros(len(zeta1))
        q2 = np.zeros(len(zeta2))
        for i in range(len(zeta1)):
            qvec = np.inner(matC, zeta[:, i])
            q1[i] = qvec[0]
            q2[i] = qvec[1]
        return q1, q2

    def get_Dt(self, T):
        """
        Calculation of necessary time steps.

        The number of points get calculated by the number of dividing points in
        the z axis.

        :param T: length of whole interval
        :return: dt time interval between points,
                 k number of all points
        """
        self.k = 2*self.N
        self.dt = T/self.k
        return self.dt, self.k

    def get_times(self):
        """
        Calculate the beginning and end time of the trajectory of the flat output.

        :return: begin-, endtime and the difference of the transistion time
        """
        self.ti = self._char(1, self.eta_traj.t0, self.z0, self.zT)
        self.tf = self._char(2, self.eta_traj.t1, self.z0, self.zT)
        self.T0 = 2 * self.ti - self.eta_traj.t0
        self.T1 = 2 * self.tf - self.eta_traj.t1
        self.dT = self.T1 - self.T0
        return self.T0, self.T1, self.dT, self.ti, self.tf

    def get_timeandspace(self):
        """
        Calculate the points of the z- and t-axis of the system.

        :return: time and space point arrays
        """
        self.get_Dt(self.dT)
        # create time and space axis
        self.time = np.linspace(self.T0, self.T1, self.k + 1)
        self.space = np.linspace(self.z0, self.zT, self.N+1)
        return self.time, self.space

    def transform(self, x1, x2, z=None):
        """
        Transformation of the system into the hyperbolic normal form.

        :param x1: input vector for the state x1
        :param x2: input vector for the state x2
        :param z: point z for which the transformation matrices should be evaluated
        :return: return two transformed state vectors
        """
        tmp = np.inner(self._mat_T(z), np.array([[x1], [x2]]).T)
        v1 = np.reshape(tmp[0, :], (len(x1)))
        v2 = np.reshape(tmp[1, :], (len(x2)))
        return v1, v2

    def transform_inv(self, v1, v2, z=None):
        """
        Inverse transformation of the transformed states into the original states.

        :param v1: input vector for the transformed state v1
        :param v2: input vector for the transformed state v2
        :param z: point z for which the transformation matrices should be evaluated
        :return: return the two original state vectors
        """
        tmp = np.inner(self._mat_Tinv(z), np.array([[v1], [v2]]).T)
        x1 = np.reshape(tmp[0, :], (len(v1)))
        x2 = np.reshape(tmp[1, :], (len(v2)))
        return x1, x2

    def integrate_solution(self, x1_z0, x2_z0):
        """
        Integration algorithm to calculate the integrated solution of the cauchy
        problem.

        :param x1_z0: first state transition on ground of the flat output
        :param x2_z0: second state transition on ground of the flat output
        :return: the array of the integrated transition of both states
        """
        dim = [len(self.space), len(self.time)]
        N = len(self.space)
        zeta1 = np.zeros(dim)
        zeta1[0, :] = np.copy(x1_z0)
        zeta2 = np.zeros(dim)
        zeta2[0, :] = np.copy(x2_z0)
        tau1_bar = np.zeros(dim)
        tau1_bar[0, :] = np.copy(self.time)
        tau2_bar = np.zeros(dim)
        tau2_bar[0, :] = np.copy(self.time)
        zeta1_bar = np.zeros(dim)
        zeta2_bar = np.zeros(dim)
        for j, z in enumerate(tqdm(self.space, file=sys.stdout,
                              desc=">>> integrate cauchy problem")):
            if j+1 == N:
                break
            # calculate time shifts over the characteristics
            tau1_bar[j+1, :] = self.time + self.dz*self._lmda(1, z)
            tau2_bar[j+1, :] = self.time + self.dz*self._lmda(2, z)
            # calculate the new estimated states
            q1, q2 = self._qvec(zeta1[j, :], zeta2[j, :], z)
            zeta1_bar[j+1, :] = zeta1[j, :] + self.dz*q1[:]
            zeta2_bar[j+1, :] = zeta2[j, :] + self.dz*q2[:]
            # interpolate the next state within the estimated time and place
            zeta1[j+1, :] = np.interp(self.time, tau1_bar[j+1, :],
                                      zeta1_bar[j+1, :])
            zeta2[j+1, :] = np.interp(self.time, tau2_bar[j+1, :],
                                      zeta2_bar[j+1, :])
        return [zeta1, zeta2]


def transform_to_diagonal_sys(A, B, z0, z):
    r"""
    Transform system into normal coordinates and calculate the diagonal matrix
    with the eigenvalues.

    As input it is necessary to define two sympy matrix of the desired system
    from the form like:

    .. math::
        \frac{\partial \pmb{x}}{\partial z}(z, t) + A(z) \frac{\partial \pmb{x}}
        {\partial t}(z, t) = B(z)\pmb{x}(z, t)

    From this form the eigenvalues will be calculated and a transformation
    :math:`\bar{\pmb{x}} = T(z)\pmb{x}` will be made. The resulting system after
    the transformation in normal coordinates has following form:

    .. math::
        \frac{\partial \bar{\pmb{x}}}{\partial z}(z, t) + \Lambda(z)
        \frac{\partial \bar{\pmb{x}}}{\partial t}(z, t) =
        C(z)\bar{\pmb{x}}(z, t)

    :param A: sp.Matrix() like described above
    :param B: sp.Matrix() like described above
    :param z: symbol for which will get checked if matrix depends from it
    :return: return the matrix :math:`\Lambda(z)`, :math:`T(z)` and :math:`C(z)`
    """
    if not isinstance(A, sp.dense.MutableDenseMatrix):
        raise NotImplementedError(
            "A is not a sympy matrix!")
    if not isinstance(B, sp.dense.MutableDenseMatrix):
        raise NotImplementedError(
            "B is not a sympy matrix!")
    eigvec = A.eigenvects()
    n = len(eigvec)
    r = [None]*n
    Tinv = sp.Matrix()
    for i, vec in enumerate(eigvec[::-1]):
        # negation so that first eigenvalue is negative
        r[i] = -vec[-1][0]
        # calculate the norm of the vector to normalize it
        rsum = 0
        for elem in r[i]:
            rsum += elem**2
        Tinv = sp.Matrix.hstack(Tinv, r[i]) # /sp.sqrt(rsum))
    # calculate the left handed transformation matrix
    T = Tinv.inv()
    # calculate Lambda matrix
    Lamda = T*A*Tinv
    # calculate C matrix and look if T depends on z
    C = T.diff(z)*Tinv + T*B*Tinv
    # calculate characteristics
    lamda = [None]*n
    for i in range(n):
        lamda[i] = Lamda[i, i].simplify()
    char = [None]*n
    for i in range(n):
        # negation because we negated the right hand side
        char[i] = -sp.Integral(lamda[i], (z, z0, z)).doit()
    return Lamda, C, T, lamda, char
