"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.
"""
import numpy as np
import unittest

from unittest import TestCase
from smt.problems import Rosenbrock, Branin

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.factory import get_performance_indicator

from smoot import MOO

# creation of a Pymoo problem to be able to use NSGA2 on it
class MyProblem(Problem):
    def __init__(self, fun1, fun2):
        super().__init__(
            n_var=2,
            n_obj=2,
            n_constr=0,
            xl=np.array([-2.0, -2.0]),
            xu=np.array([2.0, 2.0]),
            elementwise_evaluation=True,
        )
        self.fun1 = fun1
        self.fun2 = fun2

    def _evaluate(self, x, out, *args, **kwargs):
        xx = np.asarray(x).reshape(1, -1)  # Our functions take array as entry
        f1 = self.fun1(xx)[0][0]
        f2 = self.fun2(xx)[0][0]
        out["F"] = [f1, f2]


class Test(TestCase):
    def test_smoot(self):

        # Problem definition
        ndim = 2
        ny = 2
        fun1 = Rosenbrock(ndim=ndim)
        fun2 = Branin(ndim=ndim)

        # Function to minimize
        def objective(x):
            return [fun1(x), fun2(x)]

        xlimits = np.array([[-2.0, 2.0], [-2.0, 2.0]])

        # Compute true Pareto front
        problem_exact = MyProblem(fun1, fun2)
        algorithm_bis = NSGA2(pop_size=100)
        res_exact = minimize(
            problem_exact,
            algorithm_bis,
            ("n_gen", 100),
            verbose=True,  # False if you do not want the text
            seed=1,
        )

        # Optimize with smoot
        mo = MOO(
            n_iter=10,  # added points to refine the model
            n_start=20,  # points for the initial sampling (default sampling method : LHS)
            xlimits=xlimits,
            n_gen=50,  # number of generations for the genetic algorithm
            pop_size=50,
        )  # number of new individuals at every generation of NSGA2

        mo.optimize(objective)

        gd = get_performance_indicator("gd", res_exact.F)
        dist = gd.calc(mo.result.F)
        self.assertLess(dist, 2.0)


if __name__ == "__main__":
    unittest.main()
