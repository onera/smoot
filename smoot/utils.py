# -*- coding: utf-8 -*-
"""
Created on Mon May 31 09:35:47 2021

@author: robin
"""

from smoot import MOO
from smoot import ZDT

import ast
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
from smt.sampling_methods import LHS
from pymoo.factory import get_performance_indicator


def write_increase_iter(
    fun,
    path,
    xlimits=None,
    reference=None,
    n_max=10,
    runs=5,
    paraMOO={},
    verbose=True,
    indic="igd",
    start_seed=0,
    criterions=["PI", "EHVI", "GA", "WB2S", "MPI"],
    subcrits=["EHVI", "EHVI", "EHVI", "EHVI", "EHVI"],
    transfos=[
        lambda l: sum(l),
        lambda l: sum(l),
        lambda l: sum(l),
        lambda l: sum(l),
        lambda l: sum(l),
    ],
    titles=None,
):
    """
    write a dictionnary with the results of the runs for each criterion in path.

    Parameters
    ----------
    fun : function
        function to optimize.
    path : str
        path to store datas.
    xlimits : ndarray[n_var,2], if None then it takes fun.xlimits
        bounds limits of fun, optional
    reference : ndarray[ n_points, n_obj], optional
        comparison explicit pareto points or reference point if "hv" is the indicator. The default is None.
    n_max : int
        maximal number of points added during the enrichent process. The default is 10.
    runs : int, optional
        number of runs with different seeds. The default is 5.
    paraMOO : dictionnary, optional
        Non-default MOO parameters for the optimization. The default is {"pop_size" : 50}.
    verbose : Bool, optional
        If informations are given during the process. The default is True.
    subcrits : list of str
        Subcriterions for wb2S
    transfos : list of function
        Transformations for wb2S
    """
    if xlimits is None:
        xlimits = fun.xlimits
    if reference is None and indic != "hv":
        reference = fun.pareto()[1]
    if indic == "hv":
        igd = get_performance_indicator(indic, ref_point=reference)
    else:
        igd = get_performance_indicator(indic, reference)
    if titles is None:
        titles = criterions
    fichier = open(path, "wb")
    mo = MOO(xlimits=xlimits)
    for clef, val in paraMOO.items():
        mo.options._dict[clef] = val

    def obj_profile(
        criterion="PI", n=n_max, seed=3, subcrit="EHVI", transfo=lambda l: sum(l)
    ):
        """
        Intermediate function to run for a specific criterion 1 run
        of MOO, giving the resulting front at each iteration
        """
        fronts = []
        times = []
        sampling = LHS(xlimits=xlimits, random_state=seed)
        xdoe = sampling(mo.options["n_start"])
        mo.options["criterion"] = criterion
        mo.options["random_state"] = seed
        mo.options["xdoe"] = xdoe
        mo.options["n_iter"] = 0
        mo.options["subcrit"] = subcrit
        mo.options["transfo"] = transfo
        mo.optimize(fun)
        fronts.append(mo.result.F)
        mo.options["n_iter"] = 1
        for i in range(1, n):
            stime = time.time()
            mo.options[
                "xdoe"
            ] = xdoe  # the last doe is used for the new iteration to add a point
            X, _ = mo.optimize(fun)
            fronts.append(fun(X))
            times.append(time.time() - stime)
            xdoe = mo.modeles[0].training_points[None][0][0]
        dists = [igd.calc(fr) for fr in fronts]  # - to have the growth as goal

        if verbose:
            print("xdoe", xdoe)
            # print("distances",dists)
        return dists, fronts, times

    dico_res = {crit: {} for crit in titles}
    # plots_moy = []
    for i, crit in enumerate(criterions):
        fronts_pareto = []
        distances = []
        temps = []
        if verbose:
            print("criterion ", titles[i])
        for graine in range(start_seed, start_seed + runs):
            if verbose:
                print("seed", graine)
            di, fr, tmps = obj_profile(
                crit, n=n_max, seed=graine, subcrit=subcrits[i], transfo=transfos[i]
            )
            fronts_pareto.append(fr)
            distances.append(di)
            temps.append(tmps)
        dico_res[titles[i]] = {
            "time": temps.copy(),
            "fronts": fronts_pareto.copy(),
            "dists": distances.copy(),
        }
    pickle.dump(dico_res, fichier)
    fichier.close()


def write_results(fun, path, runs=1, paraMOO={}):
    """
    Run runs times the optimizer on fun using the paraMOO parameters.
    The results of each run are stored in path using the pickle module.
    To get the datas for postprocessing, use read_results(path).

    Parameters
    ----------
    fun : function
        black box function to optimize : ndarray[ne,nx] -> ndarray[ne,ny]
    path : string
        Absolute path to store the datas.
    runs : int, optional
        Number of runs of the MOO solver. The default is 1.
    paraMOO : dictionnary, optional
        parameters for MOO solver. The default is {}.
    """
    fichier = open(path, "wb")
    mo = MOO()
    for clef, val in paraMOO.items():
        mo.options._dict[clef] = val
    pickle.dump(mo.options._dict, fichier)
    dico_res = {}
    for i in range(runs):
        titre = "run" + str(i)
        dico_res[titre] = {}
        mo.optimize(fun)
        dico_res[titre]["F"] = mo.result.F
        dico_res[titre]["X"] = mo.result.X
    pickle.dump(dico_res, fichier)
    fichier.close()


def read_results(path):
    """
    read the results written thanks to write_results in path

    Parameters
    ----------
    path : string
        Absolute path.

    Returns
    -------
    param : dictionnary
        dictionnary of the given parameters for the runs
    results : dictionnary
        contains the datas relatives to the runs. For instance,
        results["run0"]["F"] contains the pareto front of the first run.
    """
    fichier = open(path, "rb")
    param = pickle.load(fichier)
    results = pickle.load(fichier)
    fichier.close()
    return param, results


def pymoo2fun(pb):
    """
    Takes a pymoo problem and makes of it a funtion optimizable thanks to MOO

    Parameters
    ----------
    pb : pymoo.problems
        Pymoo problem, such as those obtained thanks to get_problem from pymoo.factory.

    Returns
    -------
    f_equiv : function
        Callable function, equivalent to the one of the problem : ndarray[ne,nx] -> ndarray[ne,ny].

    """

    def f_equiv(x):
        output = {}
        pb._evaluate(x, output)
        return output["F"]

    return f_equiv


def pymoo2constr(pb):
    """
    Creates the list of the constraints relatives to the pymoo problem in argument.

    Parameters
    ----------
    pb : pymoo.problems
        Constrained pymoo problem to optimize.

    Returns
    -------
    list_con : list
        List of the callable constraints : ndarray[ne,nx] -> ndarray[ne].

    """
    list_con = []
    for i in range(pb.n_constr):

        def g_equiv(x, i=i):
            output = {}
            pb._evaluate(x, output)
            return output["G"][:, i]

        list_con.append(g_equiv)
    return list_con
