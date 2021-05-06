# -*- coding: utf-8 -*-
"""
Created on Mon May  3 14:38:58 2021

@author: robin
"""

import numpy as np

class MonteCarlo(object):
    
    def __init__(self, random_state = None):
        self.seed = np.random.RandomState(random_state)
    
    def sampling(self, x, distrib, points = 300):
        """
        Samples the objective space according to the probability distribution
        Points are uniformly generated on the design space, then their image
        through the model is the output

        Parameters
        ----------
        x : ndarray[n_dim]
            Design point to evaluate thanks to a criteria after the sampling.
        distrib : list of smt models
            models of the objective.
        points : int, optional
            Number of points of the sampling. Should be modulated in function of the number of objectives. The default is 300.

        Returns
        -------
        ndarray[points, n_obj]
            point's distribution in the objective space according to the model(s).

        """
        moyennes = np.asarray([model.predict_values(x)[0][0] for model in distrib])
        sigmas = np.asarray([model.predict_variances(x)[0][0]**0.5 for model in distrib])
        return self.seed.normal(moyennes,sigmas,(points,len(distrib)))

 