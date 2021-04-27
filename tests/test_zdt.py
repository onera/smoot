"""
Author: robin grapin

This package is distributed under New BSD license.
"""
import numpy as np
import unittest
from smt.utils.sm_test_case import SMTestCase
from smoot import ZDT


class Test(SMTestCase):
    def run_test(self, problem):
        problem.options["return_complex"] = True

        # Test xlimits
        ndim = problem.options["ndim"]
        xlimits = problem.xlimits
        self.assertEqual(xlimits.shape, (ndim, 2))

        # Test evaluation of multiple points at once
        x = np.zeros((10, ndim))
        for ind in range(10):
            x[ind, :] = 0.5 * (xlimits[:, 0] + xlimits[:, 1])
        y = problem(x)
        if type(y) == list or type(y) == tuple:
            y = y[0]
        self.assertEqual(x.shape[0], y.shape[0])

    def test_zdt(self):
        self.run_test(ZDT(type=1))
        self.run_test(ZDT(type=2))
        self.run_test(ZDT(type=3))
        self.run_test(ZDT(type=4))
        self.run_test(ZDT(type=5))


if __name__ == "__main__":
    unittest.main()
