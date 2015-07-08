from __future__ import division

__author__ = 'stefan'

import unittest
import numpy as np

from pyinduct import core, utils
import pyqtgraph as pg


class CureTestCase(unittest.TestCase):
    def setUp(self):
        self.node_cnt = 3
        self.nodes = np.linspace(0, 2, self.node_cnt)
        self.dz = (2 - 0) / (self.node_cnt-1)
        self.test_functions = np.array([core.LagrangeFirstOrder(0, 0, 1),
                                        core.LagrangeFirstOrder(0, 1, 2),
                                        core.LagrangeFirstOrder(1, 2, 2)])

    def test_init(self):
        self.assertRaises(TypeError, utils.cure_interval, np.sin, [2, 3])
        self.assertRaises(TypeError, utils.cure_interval, np.sin, (2, 3))
        self.assertRaises(ValueError, utils.cure_interval, core.LagrangeFirstOrder, (0, 2))
        self.assertRaises(ValueError, utils.cure_interval, core.LagrangeFirstOrder, (0, 2), 2, 1)

    def test_rest(self):
        nodes1, funcs1 = utils.cure_interval(core.LagrangeFirstOrder, (0, 2), node_count=self.node_cnt)
        self.assertTrue(np.allclose(nodes1, self.nodes))
        nodes2, funcs2 = utils.cure_interval(core.LagrangeFirstOrder, (0, 2), element_length=self.dz)
        self.assertTrue(np.allclose(nodes2, self.nodes))

        for i in range(self.test_functions.shape[0]):
            self.assertEqual(self.test_functions[i].nonzero, funcs1[i].nonzero)
            self.assertEqual(self.test_functions[i].nonzero, funcs2[i].nonzero)


class FindRoots(unittest.TestCase):

    def setUp(self):
        def eig_eq(w):
            return np.pi * np.cos(2*w) + w*np.sin(w)

        def eig_eq2(w):
            return np.cos(w) + w* np.sin(w)

        def _char_equation(omega):
            return omega * (np.sin(omega) + omega * np.cos(omega))

        self.app = pg.QtGui.QApplication([])
        self.char_eq = eig_eq
        self.char_eq = eig_eq2

    def test_feasible(self):
        roots = utils.find_roots(self.char_eq, 1e1)
        self.assertEqual(len(roots), 1e1)
        for root in roots:
            self.assertAlmostEqual(self.char_eq(root), 0, places=5)

        points = np.arange(0, 100, .1)
        vals = self.char_eq(points)
        pw = pg.plot(title="char equation roots")
        pw.plot(points, vals)
        pw.plot(roots, self.char_eq(roots), pen=None, symbolPen=pg.mkPen("g"))
        self.app.exec_()

    def tearDown(self):
        del self.app
