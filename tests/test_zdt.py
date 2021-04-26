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
        ny = y.shape[1]
        self.assertEqual(x.shape[0], y.shape[0])

        # Test derivatives
        x = np.zeros((4, ndim), complex)
        x[0, :] = 0.2 * xlimits[:, 0] + 0.8 * xlimits[:, 1]
        x[1, :] = 0.4 * xlimits[:, 0] + 0.6 * xlimits[:, 1]
        x[2, :] = 0.6 * xlimits[:, 0] + 0.4 * xlimits[:, 1]
        x[3, :] = 0.8 * xlimits[:, 0] + 0.2 * xlimits[:, 1]
        y0 = problem(x)
        if type(y0) == list or type(y0) == tuple:
            y0 = y0[0]
        dydx_FD = np.zeros(4)
        dydx_CS = np.zeros(4)
        dydx_AN = np.zeros(4)

        print()
        h = 1e-5
        ch = 1e-16
        for iy in range(ny):
            for idim in range(ndim):
                x[:, idim] += h
                y_FD = problem(x)
                x[:, idim] -= h

                x[:, idim] += complex(0, ch)
                y_CS = problem(x)
                x[:, idim] -= complex(0, ch)
                problem_idim = problem(x, idim)

                if type(y_FD) == list or type(y_FD) == tuple:
                    y_FD = y_FD[0]
                    y_CS = y_CS[0]
                    problem_idim = problem(x, idim)[0]

                dydx_FD[:] = (y_FD[:, iy] - y0[:, iy]) / h
                dydx_CS[:] = np.imag(y_CS[:, iy]) / ch
                dydx_AN[:] = problem_idim[:, iy]

                abs_rms_error_FD = np.linalg.norm(dydx_FD - dydx_AN)
                rel_rms_error_FD = np.linalg.norm(dydx_FD - dydx_AN) / np.linalg.norm(
                    dydx_FD
                )

                abs_rms_error_CS = np.linalg.norm(dydx_CS - dydx_AN)
                rel_rms_error_CS = np.linalg.norm(dydx_CS - dydx_AN) / np.linalg.norm(
                    dydx_CS
                )

                msg = (
                    "{:16s} iy {:2} dim {:2} of {:2} "
                    + "abs_FD {:16.9e} rel_FD {:16.9e} abs_CS {:16.9e} rel_CS {:16.9e}"
                )
                print(
                    msg.format(
                        problem.options["name"],
                        iy,
                        idim,
                        ndim,
                        abs_rms_error_FD,
                        rel_rms_error_FD,
                        abs_rms_error_CS,
                        rel_rms_error_CS,
                    )
                )
                self.assertTrue(rel_rms_error_FD < 1e-3 or abs_rms_error_FD < 1e-5)

    def test_zdt(self):
        self.run_test(ZDT(type=1))
        self.run_test(ZDT(type=2))
        self.run_test(ZDT(type=3))
        self.run_test(ZDT(type=4))
        self.run_test(ZDT(type=5))


if __name__ == "__main__":
    unittest.main()
