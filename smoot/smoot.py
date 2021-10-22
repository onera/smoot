# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 14:08:54 2021

@author: robin grapin
"""

import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.factory import get_performance_indicator

from smt.applications.application import SurrogateBasedApplication
from smt.surrogate_models import KRG, KPLS
from smt.sampling_methods import LHS

from smoot.criterion import Criterion


class MOO(SurrogateBasedApplication):
    def _initialize(self):

        super()._initialize()
        declare = self.options.declare

        declare(
            "surrogate",
            "KPLS",
            types=str,
            values=["KRG", "KPLS"],
            desc="Surrogate model type",
        )
        declare(
            "const",
            [],
            types=list,
            desc="constraints of the problem, should be <=0 constraints, taking x = ndarray[ne,nx], out ndarray[ne,1] each",
        )
        declare(
            "penal",
            True,
            types=bool,
            desc="True to maximize the criterion with penalty, False for subject to constraint's models",
        )
        declare(
            "subcrit",
            "EHVI",
            types=str,
            values=["PI", "MPI", "EHVI"],
            desc="subcriterion for the formula of wb2s : s*subcrit - transfo(µ)",
        )
        declare(
            "transfo",
            lambda l: sum(l),
            types=type(lambda x: x),
            desc="transfo function for wb2s formula : s*subcrit - transfo(µ)",
        )
        declare(
            "criterion",
            "PI",
            types=str,
            values=["PI", "EHVI", "GA", "WB2S", "MPI"],
            desc="infill criterion",
        )
        declare("n_iter", 10, types=int, desc="Number of optimizer steps")
        declare("xlimits", None, types=np.ndarray, desc="Bounds of function fun inputs")
        declare("n_start", 20, types=int, desc="Number of optimization start points")
        declare(
            "pop_size",
            50,
            types=int,
            desc="number of individuals for the genetic algorithm",
        )
        declare(
            "n_gen",
            50,
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
        declare(
            "xdoe",
            None,
            types=np.ndarray,
            desc="Initial doe inputs. DoE formats are ndarray[n_start,n_dim]",
        )
        declare("ydoe", None, types=np.ndarray, desc="Initial doe outputs")
        declare(
            "ydoe_c", None, types=np.ndarray, desc="initial doe outputs for constraints"
        )
        declare(
            "random_state",
            None,
            types=(type(None), int),
            desc="seed number which controls random draws",
        )

    def optimize(self, fun):
        """
        Optimize the multi-objective function fun. At the end, the object's item
        .modeles is a SMT surrogate_model object with the most precise fun's model
        .result is the result of its optimization thanks to NSGA2

        Parameters
        ----------
        fun : function
            function taking x=ndarray[ne,ndim],
            returning y = ndarray[ne,ny]
            where y[i][j] = fj(xi).
            If fun has only one objective, y = ndarray[ne, 1]

        Returns
        -------
        self.result.X : ndarray[int,n_var]
            Pareto Set.
        self.result.F : ndarray[int,ny]
            Pareto Front.
        """
        if type(self.options["xlimits"]) != np.ndarray:
            try:
                self.options["xlimits"] = fun.xlimits
            except AttributeError:  # if fun doesn't have "xlimits" attribute
                raise AttributeError("Error : No bounds given")
                return

        self.seed = np.random.RandomState(self.options["random_state"])
        self.n_const = len(self.options["const"])
        x_data, y_data, y_data_c = self._setup_optimizer(fun)
        self.ndim = self.options["xlimits"].shape[0]
        self.ny = y_data.shape[-1]

        if self.ny == 1:
            self.log("EGO will be used as there is only 1 objective")
            if self.n_const > 0:
                self.log("EGO doesn't take constraints in account")
            self.use_ego(fun, x_data, y_data)
            self.log(
                "Optimization done, get the front with .result.F and the set with .result.X"
            )
            return

        # obtaining models for each objective
        self.modelize(x_data, y_data, y_data_c)

        if type(y_data) != list:
            y_data = list(y_data)

        for k in range(self.options["n_iter"]):

            self.log(str("iteration " + str(k + 1)))

            # find next best x-coord point to evaluate
            new_x, _ = self._find_best_point(self.options["criterion"])
            new_x = np.array([new_x])
            new_y = fun(new_x)

            # update model with the new point
            y_data = np.atleast_2d(np.append(y_data, new_y, axis=0))
            x_data = np.atleast_2d(np.append(x_data, new_x, axis=0))

            # update the constraints
            if self.n_const > 0:
                new_y_c = np.transpose(
                    np.array(
                        [
                            self.options["const"][i](new_x)[0]
                            for i in range(self.n_const)
                        ]
                    )
                )
                y_data_c = np.atleast_2d(np.append(y_data_c, new_y_c, axis=0))

            self.modelize(x_data, y_data, y_data_c)

        self.log("Model is well refined, NSGA2 is running...")
        self.result = minimize(
            self.def_prob(
                n_var=self.ndim,
                xbounds=self.options["xlimits"],
                n_obj=self.ny,
                obj=self.modeles,
                n_const=self.n_const,
                const=self.const_modeles,
            ),
            NSGA2(
                pop_size=2 * self.options["pop_size"], seed=self.options["random_state"]
            ),
            ("n_gen", 2 * self.options["n_gen"]),
            seed=self.options["random_state"],
        )
        self.log(
            "Optimization done, get the front with .result.F and the set with .result.X"
        )
        return self.result.X, self.result.F

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
            yt[i] = fi(xt).

        """
        xt, yt, yc = self.options["xdoe"], self.options["ydoe"], self.options["ydoe_c"]
        if xt is None and not (yt is None and yc is None):
            print("xdoe must be an array if you want to use ydoe or ydoe_c")
            yt, yc = None, None
        if xt is None:
            sampling = LHS(
                xlimits=self.options["xlimits"],
                random_state=self.options["random_state"],
            )
            xt = sampling(self.options["n_start"])
        if yt is None:
            yt = fun(xt)
        if yc is None and self.n_const > 0:
            yc = [np.array(con(xt)) for con in self.options["const"]]
        return xt, yt, yc

    def modelize(self, xt, yt, yt_const=None):
        """
        Creates and train a krige model with the given datapoints

        Parameters
        ----------
        xt : ndarray[n_points, n_dim]
            Design space coordinates of the training points.
        yt : ndarray[n_points, n_objectives]
            Training outputs.
        yt_const : list of ndarray[nt,ny]
            constraints training outputs
        """
        self.modeles = []
        for iny in range(self.ny):
            t = (
                KRG(print_global=False)
                if self.options["surrogate"] == "KRG"
                else KPLS(print_global=False)
            )
            t.set_training_values(xt, yt[:, iny])
            t.train()
            self.modeles.append(t)

        self.const_modeles = []
        if not (yt_const is None):
            for iny in range(self.n_const):
                t = (
                    KRG(print_global=False)
                    if self.options["surrogate"] == "KRG"
                    else KPLS(print_global=False)
                )
                t.set_training_values(xt, yt_const[:, iny])
                t.train()
                self.const_modeles.append(t)

    def def_prob(self, n_var, xbounds, n_obj, obj, n_const, const):
        """
        Creates the pymoo Problem object with the surrogate as objective

        Returns
        -------
        MyProblem : pymoo.problem
        """

        class MyProblem(ElementwiseProblem):#!!!
            def __init__(self):
                super().__init__(
                    n_var=n_var,
                    n_obj=n_obj,
                    n_constr=n_const,
                    xl=np.asarray([i[0] for i in xbounds]),
                    xu=np.asarray([i[1] for i in xbounds]),
                    #elementwise_evaluation=True,#!!!
                )

            def _evaluate(self, x, out, *args, **kwargs):
                if n_obj > 1:
                    xx = np.asarray(x).reshape(1, -1)
                    out["F"] = [f.predict_values(xx)[0][0] for f in obj]
                    if n_const > 0:
                        out["G"] = [g.predict_values(xx)[0][0] for g in const]

                else:  # 1 obj is for acquisition function
                    out["F"] = obj(x)
                    if n_const > 0:  # case without penalization
                        xx = np.asarray(x).reshape(1, -1)
                        out["G"] = [g.predict_values(xx)[0][0] for g in const]

        return MyProblem()

    def _find_best_point(self, criter):
        """
        Selects the best point to refine the model according to
        the chosen infill criterion.

        Returns
        -------
        ndarray
            next point for the model update.
        """
        if criter == "GA":
            res = minimize(
                self.def_prob(
                    n_var=self.ndim,
                    xbounds=self.options["xlimits"],
                    n_obj=self.ny,
                    obj=self.modeles,
                    n_const=self.n_const,
                    const=self.const_modeles,
                ),
                NSGA2(
                    pop_size=self.options["pop_size"], seed=self.options["random_state"]
                ),
                ("n_gen", self.options["n_gen"]),
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
                return X[self.seed.randint(len(X)), :]
            dispersion = [
                q * (d_l_x[j] - µ_x) / var_x + (1 - q) * (d_l_f[j] - µ_f) / var_f
                for j in range(X.shape[0])
            ]
            i = dispersion.index(max(dispersion))
            return X[i, :]

        if criter == "PI":
            PI = Criterion(
                "PI",
                self.modeles,
                random_state=self.options["random_state"],
            )
            self.obj_k = lambda x: -PI(x)

        if criter == "MPI":
            MPI = Criterion(
                "MPI", self.modeles, random_state=self.options["random_state"]
            )
            self.obj_k = lambda x: -MPI(x)

        if criter == "EHVI":
            ydata = np.transpose(
                np.asarray([mod.training_points[None][0][1] for mod in self.modeles])
            )[0]
            ref = [ydata[:, i].max() + 1 for i in range(self.ny)]  # nadir +1
            hv = get_performance_indicator("hv", ref_point=np.asarray(ref))
            EHVI = Criterion(
                "EHVI",
                self.modeles,
                ref=ref,
                hv=hv,
                random_state=self.options["random_state"],
            )
            self.obj_k = lambda x: -EHVI(x)

        if criter == "WB2S":
            xmax, valmax = self._find_best_point(self.options["subcrit"])
            if valmax == 0:
                s = 1
            else:
                moyennes = [mod.predict_values for mod in self.modeles]
                beta = 100  # hyperparameter
                s = (
                    beta
                    * self.options["transfo"](
                        [moy(np.asarray(xmax).reshape(1, -1))[0][0] for moy in moyennes]
                    )
                    / valmax
                )
            if self.options["subcrit"] == "EHVI":
                ydata = np.transpose(
                    np.asarray(
                        [mod.training_points[None][0][1] for mod in self.modeles]
                    )
                )[0]
                ref = [ydata[:, i].max() + 1 for i in range(self.ny)]  # nadir +1
                hv = get_performance_indicator("hv", ref_point=np.asarray(ref))
            else:
                ref, hv = None, None
            subcriterion = Criterion(
                self.options["subcrit"],
                self.modeles,
                hv=hv,
                ref=ref,
                random_state=self.options["random_state"],
            )
            WB2S = Criterion(
                "WB2S",
                self.modeles,
                s=s,
                random_state=self.options["random_state"],
                subcrit=subcriterion,
                transfo=self.options["transfo"],
            )
            self.obj_k = lambda x: -WB2S(x)

        if self.options["penal"] and self.n_const > 0:
            prob = self.def_prob(
                n_var=self.ndim,
                xbounds=self.options["xlimits"],
                n_obj=1,
                obj=self.penal(self.obj_k),
                n_const=0,
                const=[],
            )

        else:
            prob = self.def_prob(
                n_var=self.ndim,
                xbounds=self.options["xlimits"],
                n_obj=1,
                obj=self.obj_k,
                n_const=self.n_const,
                const=self.const_modeles,
            )

        maximizers = minimize(
            prob,
            NSGA2(pop_size=self.options["pop_size"], seed=self.options["random_state"]),
            ("n_gen", self.options["n_gen"]),
            seed=self.options["random_state"],
        ).X
        x_opt = (
            maximizers
            if len(maximizers.shape) == 1
            else maximizers[self.seed.randint(len(maximizers))]
        )
        self.log(criter + " max value : " + str(-self.obj_k(x_opt)))
        self.log("xopt : " + str(x_opt))
        for i in range(self.n_const):
            self.log(
                "constraint "
                + str(i)
                + " estimated value : "
                + str(self.const_modeles[i].predict_values(np.array([x_opt]))[0][0])
            )
        return x_opt, -self.obj_k(x_opt)

    def penal(self, f):
        """
        "Penalized through weightening" criterion by the probability
        of feasability at each point

        Parameters
        ----------
        f : function
            Criterion to minimize (because f : x -> - criterion(x) ).

        Returns
        -------
        function
            weighted function.
        """
        return lambda x: (f(x) - 0.01) * Criterion.prob_of_feasability(
            x, self.const_modeles
        )
        # 0.01 because the criterion is often equal to 0, so it favorises the reachable points

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
            verbose=self.options["verbose"],
            random_state=self.options["random_state"],
        )
        x_opt, y_opt, _, _, _ = ego.optimize(fun)
        self.result = Algorithm()
        self.result.X = np.array([[x_opt]])
        self.result.F = np.array([[y_opt]])
        self.log(
            "Optimization done, get the front with .result.F and the set with .result.X"
        )
        return x_opt, y_opt
