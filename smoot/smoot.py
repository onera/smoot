# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 14:08:54 2021

@author: robin grapin
"""

#%% imports

import numpy as np
from random import uniform
from types import FunctionType

from scipy.optimize import minimize as minimize1D

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.optimize import minimize

from smt.applications.application import SurrogateBasedApplication
from smt.surrogate_models import KPLS, KRG, KPLSK, MGP
from smt.sampling_methods import LHS

from smoot.criterion import criterion

#%% Optimization loop incrementing the surrogates


class MOO(SurrogateBasedApplication):
    def _initialize(self):

        super()._initialize()
        declare = self.options.declare

        declare("fun", None, types=FunctionType, desc="Function to minimize")
        declare(
            "criterion",
            "PI",
            types=str,
            values=["PI", "EHVI", "GA", "WB2S"],
            desc="criterion for next evaluation point determination: Expected Improvement, \
            Surrogate-Based Optimization or genetic algo point",
        )
        declare("n_iter", 10, types=int, desc="Number of optimizer steps")
        declare(
            "n_max_optim",
            20,
            types=int,
            desc="Maximum number of internal optimizations",
        )
        declare("xlimits", None, types=np.ndarray, desc="Bounds of function fun inputs")
        declare("n_start", 20, types=int, desc="Number of optimization start points")
        declare(
            "n_parallel",
            1,
            types=int,
            desc="Number of parallel samples to compute using qEI criterion",
        )
        declare(
            "surrogate",
            KRG(print_global=False),
            types=(KRG, KPLS, KPLSK, MGP),
            desc="SMT kriging-based surrogate model used internaly",
        )  # ne pas utiliser ou adapter au multiobj qu'on aie bien des modeles indep pour chaque objectif
        declare(
            "pop_size",
            100,
            types=int,
            desc="number of individuals for the genetic algorithm",
        )
        declare(
            "n_gen",
            100,
            types=int,
            desc="number generations for the genetic algorithm",
        )
        declare(
            "q",
            0.5,
            types=float,
            desc="importance ratio of design space in comparation to objective space when chosing a point with GA",
        )
        declare("verbose", False, types=bool, desc="Print computation information")
        declare("xdoe", None, types=np.ndarray, desc="Initial doe inputs")
        declare("ydoe", None, types=np.ndarray, desc="Initial doe outputs")

    def optimize(self, fun):
        """
        Optimize the multi-objective function fun. At the end, the object's item
        .modeles is a SMT surrogate_model object with the most precise fun's model
        .result is the result of its optimization thanks to NSGA2

        Parameters
        ----------
        fun : function
            function taking x=ndarray[ne,ndim],
            returning y = [ndarray[ne, 1],ndarray[ne, 1],...]
            where y[i][j][0] = fi(xj).
        """
        if type(self.options["xlimits"]) != np.ndarray:
            try:
                self.options["xlimits"] = fun.xlimits
            except:  # if fun doesn't have "xlimits" attribute
                print("Error : No bounds given")
                return
        x_data, y_data = self._setup_optimizer(fun)
        self.ndim = self.options["xlimits"].shape[0]
        # n_parallel = self.options["n_parallel"]
        try:
            y_data[0][0][0] += 0
            self.ny = len(y_data)
        except:
            self.log("EGO will be used as there is only 1 objective")
            self.use_ego(fun, x_data, y_data)
            self.log(
                "Optimization done, get the front with .result.F and the set with .result.X"
            )
            return
        if self.ny > 2 and self.options["criterion"] != "GA":
            self.log(
                "Only GA is available for more than 2 objectives at the moment, criterion will be switched"
            )
            self.options["criterion"] = "GA"

        # obtaining models for each objective
        self.modelize(x_data, y_data)
        self.probleme = self.def_prob()

        if type(y_data) != list:
            y_data = list(y_data)

        for k in range(self.options["n_iter"]):

            self.log(str("iteration " + str(k + 1)))

            # find next best x-coord point to evaluate
            new_x = self._find_best_point()
            new_y = fun(np.array([new_x]))

            # update model with the new point
            for i in range(len(y_data)):
                y_data[i] = np.atleast_2d(np.append(y_data[i], new_y[i], axis=0))
            x_data = np.atleast_2d(np.append(x_data, np.array([new_x]), axis=0))

            self.modelize(x_data, y_data)

        self.log("Model is well refined, NSGA2 is running...")
        self.result = minimize(
            self.probleme,
            NSGA2(pop_size=self.options["pop_size"]),
            ("n_gen", self.options["n_gen"]),
            verbose=False,
        )
        self.log(
            "Optimization done, get the front with .result.F and the set with .result.X"
        )

    # return result.X , result.F

    def _setup_optimizer(self, fun):
        """
        Parameters
        ----------
        fun : objective function

        Returns
        -------
        xt : array of arrays
            sampling points in the design space.
        yt : list of arrays
            yt[i] = f1(xt).

        """
        if (
            type(self.options["xdoe"]) == np.ndarray
            and type(self.options["ydoe"]) == np.ndarray
        ):
            return self.options["xdoe"], self.options["ydoe"]
        sampling = LHS(xlimits=self.options["xlimits"])
        xt = sampling(self.options["n_start"])
        yt = fun(xt)
        return xt, yt

    def modelize(self, xt, yt):
        """
        Creates and train a krige model with the given datapoints

        Parameters
        ----------
        xt : TYPE
            Design space coordinates of the training points.
        yt : TYPE
            Training outputs.
        """
        self.modeles = []
        for iny in range(self.ny):
            t = KRG(print_global=False)
            t.set_training_values(xt, yt[iny])
            t.train()
            self.modeles.append(t)

    def def_prob(self):
        """
        Creates the pymoo Problem object with the surrogate as objective

        Returns
        -------
        MyProblem : pymoo.problem
        """
        n_obj = self.ny
        n_var = self.ndim
        xbounds = self.options["xlimits"]
        modelizations = self.modeles

        class MyProblem(Problem):
            def __init__(self):
                super().__init__(
                    n_var=n_var,
                    n_obj=n_obj,
                    n_constr=0,
                    xl=np.asarray([i[0] for i in xbounds]),
                    xu=np.asarray([i[1] for i in xbounds]),
                    elementwise_evaluation=True,
                )

            def _evaluate(self, x, out, *args, **kwargs):
                xx = np.asarray(x).reshape(1, -1)  # le modèle prend un array en entrée
                out["F"] = [i.predict_values(xx)[0][0] for i in modelizations]

        return MyProblem()

    def _find_best_point(self):
        """
        Selects the best point to refine the model, according to the chosen method

        Returns
        -------
        ndarray
            next point for the model update.
        """
        criter = self.options["criterion"]

        if criter == "GA":
            res = minimize(
                self.probleme,
                NSGA2(pop_size=self.options["pop_size"]),
                ("n_gen", self.options["n_gen"]),
                verbose=False,
            )
            X = res.X
            Y = res.F
            ydata = np.transpose(
                np.asarray([mod.training_points[None][0][1] for mod in self.modeles])
            )[0]
            xdata = self.modeles[0].training_points[None][0][0]
            # MOBOpt criterion
            q = self.options["q"]
            n = ydata.shape[1]
            d_l_x = [sum([np.linalg.norm(xj - xi) for xj in xdata]) / n for xi in X]
            d_l_f = [sum([np.linalg.norm(yj - yi) for yj in ydata]) / n for yi in Y]
            µ_x = np.mean(d_l_x)
            µ_f = np.mean(d_l_f)
            var_x, var_f = np.var(d_l_x), np.var(d_l_f)
            if var_x == 0 or var_f == 0:
                return X[0, :]
            dispersion = [
                q * (d_l_x[j] - µ_x) / var_x + (1 - q) * (d_l_f[j] - µ_f) / var_f
                for j in range(X.shape[0])
            ]
            i = dispersion.index(max(dispersion))
            return X[i, :]  # , Y[i,:]

        if criter == "PI":
            PI = criterion("PI", self.modeles)
            self.obj_k = lambda x: -PI(x)
        if criter == "EHVI":
            ydata = np.transpose(
                np.asarray([mod.training_points[None][0][1] for mod in self.modeles])
            )[0]
            ref = [ydata[:, 0].max() + 1, ydata[:, 1].max() + 1]
            EHVI = criterion("EHVI", self.modeles, ref)
            self.obj_k = lambda x: -EHVI(x)
        if criter == "WB2S":
            ydata = np.transpose(
                np.asarray([mod.training_points[None][0][1] for mod in self.modeles])
            )[0]
            ref = [ydata[:, 0].max() + 1, ydata[:, 1].max() + 1]
            EHVI = criterion("EHVI", self.modeles, ref)
            self.obj_k_inter = lambda x: -EHVI(x)
            xstart_inter = np.zeros(self.ndim)
            bounds = self.options["xlimits"]
            for i in range(self.ndim):
                xstart_inter[i] = uniform(*bounds[i])
            xmax = minimize1D(self.obj_k_inter, xstart_inter, bounds=bounds).x
            EHVImax = EHVI(xmax)
            if EHVImax == 0:
                s = 1
            else:
                moyennes = [mod.predict_values for mod in self.modeles]
                beta = 100  # to be discussed
                s = (
                    beta
                    * sum(
                        [
                            abs(moy(np.asarray(xmax).reshape(1, -1))[0][0])
                            for moy in moyennes
                        ]
                    )
                    / EHVImax
                )
            WB2S = criterion("WB2S", self.modeles, ref, s)
            self.obj_k = lambda x: -WB2S(x)

        xstart = np.zeros(self.ndim)
        bounds = self.options["xlimits"]
        for i in range(self.ndim):
            xstart[i] = uniform(*bounds[i])
        return minimize1D(self.obj_k, xstart, bounds=bounds).x

    def log(self, msg):
        if self.options["verbose"]:
            print(msg)

    def use_ego(self, fun, xdoe, ydoe):
        """
        Call ego to find the optimum of the 1D-valued funcion fun.
        The set and front are stored, as usual, in the pymoo.model.algorithm
        class result in .result.X and .result.F

        Parameters
        ----------
        fun : function
            function with one output.

        """
        from smt.applications import EGO
        from pymoo.model.algorithm import Algorithm

        ego = EGO(
            xdoe=xdoe,
            ydoe=ydoe,
            n_iter=self.options["n_iter"],
            criterion="EI",
            n_start=self.options["n_start"],
            xlimits=self.options["xlimits"],
            n_parallel=self.options["n_parallel"],
            verbose=self.options["verbose"],
            surrogate=self.options["surrogate"],
        )
        x_opt, y_opt, _, _, _ = ego.optimize(fun)
        self.result = Algorithm()
        self.result.X = np.array([[x_opt]])
        self.result.F = np.array([[y_opt]])
        self.log(
            "Optimization done, get the front with .result.F and the set with .result.X"
        )
