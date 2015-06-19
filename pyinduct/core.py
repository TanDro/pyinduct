# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from scipy import integrate


class Function:
    """
    To ensure the accurateness of numerical handling, areas where it is nonzero have to be provided
    The user can choose between providing a (piecewise) analytical or pure numerical representation of the function
    """

    def __init__(self, eval_handle, domain=(-np.inf, np.inf), nonzero=(-np.inf, np.inf)):
        if not callable(eval_handle):
            raise TypeError("Callable object has to be provided as function_handle")
        self._function_handle = eval_handle

        for kw, val in zip(["domain", "nonzero"], [domain, nonzero]):
            if not isinstance(val, list):
                if isinstance(val, tuple):
                    val = [val]
                else:
                    raise TypeError("List of tuples has to be provided for {0}".format(kw))
            setattr(self, kw, sorted([(min(interval), max(interval))for interval in val], key=lambda x: x[0]))

    def __call__(self, *args):
        """
        handle that is used to evaluate the function on a given point
        :param args: function parameter
        :return: function value
        """
        in_domain = False
        for interval in self.domain:
            # TODO support multi dimensional access
            if interval[0] <= args[0] <= interval[1]:
                in_domain = True
                break

        if not in_domain:
            raise ValueError("Function evaluated outside its domain!")

        return self._function_handle(*args)


class LagrangeFirstOrder(Function):
    """
    Implementation of an lagrangian test function of order 1
      ^
    1-|         ^
      |        /|\
      |       / | \
      |      /  |  \
    0-|-----/   |   \-------------------------> z
            |   |   |
          start,top,end
    """
    def __init__(self, start, top, end):
        if not start <= top <= end or start == end:
            raise ValueError("Input data is nonsense, see Definition.")

        if start == top:
            Function.__init__(self, self._lagrange1st_border_left, nonzero=(start, end))
        elif top == end:
            Function.__init__(self, self._lagrange1st_border_right, nonzero=(start, end))
        else:
            Function.__init__(self, self._lagrange1st_interior, nonzero=(start, end))

        self._start = start
        self._top = top
        self._end = end
        # speed
        self._a = self._top - self._start
        self._b = self._end - self._top

    def _lagrange1st_border_left(self, z):
        """
        left border equation for lagrange 1st order
        """
        if z < self._top or z >= self._end:
            return 0
        else:
            return (self._top - z) / self._b + 1

    def _lagrange1st_border_right(self, z):
        """
        right border equation for lagrange 1st order
        """
        if z <= self._start or z > self._end:
            return 0
        else:
            return (z - self._start) / self._a

    def _lagrange1st_interior(self, z):
        """
        interior equations for lagrange 1st order
        """
        if z < self._start or z > self._end:
            return 0
        elif self._start <= z <= self._top:
            return (z - self._start) / self._a
        else:
            return (self._top - z) / self._b + 1

    # @staticmethod
    # TODO implement correct one
    # def quad_int():
    #     return 2/3

def domain_intersection(first, second):
    """
    calculate intersection two domains
    :param first: domain
    :param second: domain
    :return: intersection
    """
    if isinstance(first, tuple):
        first = [first]
    if isinstance(second, tuple):
        second = [second]

    intersection = []
    first_idx = 0
    second_idx = 0
    last_first_idx = 0
    last_second_idx = 0
    last_first_upper = None
    last_second_upper = None

    while first_idx < len(first) and second_idx < len(second):
        if last_first_upper is not None and first_idx is not last_first_idx:
            if last_first_upper >= first[first_idx][0]:
                raise ValueError("Intervals not ordered!")
        if last_second_upper is not None and second_idx is not last_second_idx:
            if last_second_upper >= second[second_idx][0]:
                raise ValueError("Intervals not ordered!")

        if first[first_idx][0] > first[first_idx][1]:
            raise ValueError("Interval boundaries given in wrong order")
        if second[second_idx][0] > second[second_idx][1]:
            raise ValueError("Interval boundaries given in wrong order")

        # backup for interval order check
        last_first_idx = first_idx
        last_second_idx = second_idx
        last_first_upper = first[first_idx][1]
        last_second_upper = second[second_idx][1]

        # no common domain -> search
        if second[second_idx][0] <= first[first_idx][0] <= second[second_idx][1]:
            # common start found in 1st domain
            start = first[first_idx][0]
        elif first[first_idx][0] <= second[second_idx][0] <= first[first_idx][1]:
            # common start found in 2nd domain
            start = second[second_idx][0]
        else:
            # intervals have no intersection
            first_idx += 1
            continue

        # add end
        if first[first_idx][1] <= second[second_idx][1]:
            end = first[first_idx][1]
            first_idx += 1
        else:
            end = second[second_idx][1]
            second_idx += 1

        # complete domain found
        if not np.isclose(start, end):
            intersection.append((start, end))

    return intersection

def inner_product(first, second):
    """
    calculates the inner product of two functions
    :param first: function
    :param second: function
    :return: inner product
    """
    if not isinstance(first, Function) or not isinstance(second, Function):
        raise TypeError("Wrong type supplied must be a pyinduct.Function")

    # TODO remember not only 1d possible here!
    limits = domain_intersection(first.domain, second.domain)
    nonzero = domain_intersection(first.nonzero, second.nonzero)
    areas = domain_intersection(limits, nonzero)

    # try some shortcuts
    if first == second:
        if hasattr(first, "quad_int"):
            return first.quad_int()

    if type(first) is type(second):
        # TODO let Function Class handle product
        pass

    result = 0
    for area in areas:
        f = lambda z: first(z)*second(z)
        res = integrate.quad(f, area[0], area[1])
        result += res[0]

    return result

    # if isinstance(test_funcs, list):
    #     if not isinstance(test_funcs[0], Function):
    #         raise TypeError("Only pyinduct.Function accepted")
    #     test_funcs = np.asarray(test_funcs)
    # elif not isinstance(test_funcs, Function):
    #         raise TypeError("Only pyinduct.Function accepted")
    # else:
    #     test_funcs = np.asarray([test_funcs])
def project_on_test_functions(func, test_funcs):
    """
    projects given function on testfunctions
    :param func:
    :param test_funcs:
    :return: weights
    """
    if not isinstance(func, Function):
        raise TypeError("Only pyinduct.Function accepted as 'func'")

    if isinstance(test_funcs, Function):  # convenience case
        test_funcs = np.asarray([test_funcs])

    if not isinstance(test_funcs, np.ndarray):
        raise TypeError("Only numpy.ndarray accepted as 'test_funcs'")

    # TODO perform this somewhere else
    handle = np.vectorize(inner_product)

    # compute <x(z, t), phi_i(z)>
    projections = handle(func, test_funcs)

    # compute <phi_j(z), phi_i(z)> for 0 < i, j < n
    n = test_funcs.shape[0]
    i, j = np.mgrid[0:n, 0:n]
    funcs_i = test_funcs[i]
    funcs_j = test_funcs[j]
    scale_mat = handle(funcs_i, funcs_j)

    return np.dot(np.linalg.inv(scale_mat), projections)


def back_project_from_test_functions(weights, test_funcs):
    """
    build handle for function that was expressed in test functions with weights
    :param weights:
    :param test_funcs:
    :return: evaluation handle
    """
    if isinstance(weights, float):
        weights = np.asarray([weights])
    if isinstance(test_funcs, Function):
        test_funcs = np.asarray([test_funcs])

    if not isinstance(weights, np.ndarray) or not isinstance(test_funcs, np.ndarray):
        raise TypeError("Only numpy ndarrays accepted as input")

    if weights.shape[0] is not test_funcs.shape[0]:
        raise ValueError("Lengths of weights and test functions do not match!")

    eval_handle = lambda z: sum([weights[i]*test_funcs[i](z) for i in weights.shape[0]])
    return eval_handle
