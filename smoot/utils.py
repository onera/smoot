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
            print("x", x)
            pb._evaluate(x, output)
            return output["G"][:, i]

        list_con.append(g_equiv)
    return list_con
