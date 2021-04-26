# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 10:26:43 2021

@author: robin
"""
import numpy as np
from scipy.stats import norm

class criterion(object):
    
    def __init__(self, name, models, ref=None, s=None):
        self.modeles = models
        self.name = name
        self.ref = ref
        self.s = s

    def __call__(self, x):
        if self.name == "PI":
            return self.PI(x)
        if self.name == "EHVI":
            return self.EHVI(x,self.ref)
        if self.name == "WB2S":
            return self.WB2S(x, self.ref, self.s)
        
    # Caution !!!! 2-d objective space only for the moment !!!
    def PI(self,x ):
        """
        Parameters
        ----------
        x : list
            coordinates in the design space of the point to evaluate.

        Returns
        -------
        pi_x : float
            PI(x) : probability that x is an improvement € [0,1]
        """

        ydata = np.transpose(np.asarray([mod.training_points[None][0][1] for mod in self.modeles]))[0]
        pareto_index = self.pareto(ydata)
        pareto_front = [ydata[i] for i in pareto_index]
        moyennes = [mod.predict_values for mod in self.modeles]
        variances = [mod.predict_variances for mod in self.modeles]#racine d'une fction ne marche pas,je fais donc en 2 temps pour ecart-type
        x = np.asarray(x).reshape(1, -1)
        sig1, sig2 = variances[0](x)[0][0]**0.5, variances[1](x)[0][0]**0.5
        moy1, moy2 = moyennes[0](x)[0][0], moyennes[1](x)[0][0]
        m = len(pareto_front)
        try :
            pi_x = norm.cdf((pareto_front[0][0] - moy1)/sig1)
            for i in range(1,m-1):
                pi_x += ((norm.cdf((pareto_front[i+1][0] - moy1)/sig1)
                         - norm.cdf((pareto_front[i][0] - moy1)/sig1))
                         * norm.cdf((pareto_front[i+1][1] - moy2)/sig2))
            pi_x += (1 - norm.cdf((pareto_front[m-1][0] - moy1)/sig1))*norm.cdf((pareto_front[m-1][1] - moy2)/sig2)
            return pi_x
        except : #for training points -> having variances = 0
            print("training x called : ", x)
            return 0
    
    def psi(self,a,b,µ,s):
        return s*norm.pdf((b-µ)/s) + (a-µ)*norm.cdf((b-µ)/s)   
      
    # Caution !!!! 2-d objective space only for the moment !!!
    def EHVI(self,x, ref):
        """
        Expected hypervolume improvement if x is the new point added
        
        Parameters
        ----------
        x : list
            coordinates in the design space of the point to evaluate.
        ref : list
            coordinates of the reference point to compute the hypervolume.
            Here, we take 1 + the highest value in the training pool
            for each objective

        Returns
        -------
        res1 + res2 : float
            Expected HVImprovement
        """
        x = np.asarray(x).reshape(1, -1)
        variances = [mod.predict_variances for mod in self.modeles]
        s1, s2 = variances[0](x)[0][0]**0.5, variances[1](x)[0][0]**0.5
        if s1 == 0 or s2 == 0: #training point
            return 0
        ydata = np.transpose(np.asarray([mod.training_points[None][0][1] for mod in self.modeles]))[0]
        pareto_index = self.pareto(ydata)
        f = [ydata[i] for i in pareto_index]# pareto front
        moyennes = [mod.predict_values for mod in self.modeles]
        µ1, µ2 = moyennes[0](x)[0][0], moyennes[1](x)[0][0]
        f.sort(key=lambda x: x[0])
        f.insert(0, np.array([ref[0],-1e15]))#1e15 to approximate infinity
        f.append( np.array([-1e15, ref[1]]))
        res1, res2 = 0, 0
        for i in range(len(f)-1):
            res1 += (f[i][0]-f[i+1][0])*norm.cdf((f[i+1][0] - µ1)/s1)*self.psi(f[i+1][1],f[i+1][1],µ2,s2)
            res2 += (self.psi(f[i][0],f[i][0],µ1,s1) - self.psi(f[i][0],f[i+1][0],µ1,s1) )*self.psi(f[i][1],f[i][1],µ2,s2)
        return res1 + res2
        
    def WB2S(self,x, ref, s):
        """
        Criterion WB2S multi-objective adapted from the paper "Adapated 
        modeling strategy for constrained optimization with application
        to aerodynamic wing design" :
        WB2S(x) = s*EHVI(x) - sum( µi(x) )

        Parameters
        ----------
        x : list
            coordinates in the design space of the point to evaluate.
        ref : list
            reference point to compute the hypervolume.
        xmax : ndarray
            coordinates in the design space of the point with the best EHVI.

        Returns
        -------
        WBS2 : float
        """
        moyennes = [mod.predict_values for mod in self.modeles]
        µ = [moy(x)[0][0] for moy in moyennes]
        y = sum(µ)
        return s*self.EHVI(x, ref) - y
    
    def pareto(self,Y):
        """
        Parameters
        ----------
        Y : list of arrays
            liste of the points to compare.

        Returns
        -------
        index : list
            list of the indexes in Y of the Pareto-optimal points.
        """
        index = [] #indexes of the best points (Pareto)
        n = len(Y)
        dominated = [False]*n
        for y in range(n):
            if not dominated[y]:
                for y2 in range(y+1,n):
                    if not dominated[y2]:#if y2 is dominated (by y0), we already compared y0 to y
                        y_domine_y2 , y2_domine_y = self.dominate_min(Y[y],Y[y2])
                        
                        if y_domine_y2 :
                            dominated[y2]=True
                        if y2_domine_y :
                            dominated[y]=True
                            break
                if not dominated[y]:
                    index.append(y)
        return index
                    
    # returns a-dominates-b , b-dominates-a !! for minimization !!  
    def dominate_min(self,a,b):
        """
        Parameters
        ----------
        a : array or list
            coordinates in the objective space.
        b : array or list
            same thing than a.

        Returns
        -------
        bool
            a dominates b (in terms of minimization !).
        bool
            b dominates a (in terms of minimization !).
        """
        a_bat_b = False
        b_bat_a = False
        for i in range(len(a)):
            if a[i] < b[i]:
                a_bat_b = True
                if b_bat_a :
                    return False, False # same front
            if a[i] > b[i]:
                b_bat_a = True
                if a_bat_b :
                    return False, False    
        if a_bat_b and (not b_bat_a):
            return True, False
        if b_bat_a and (not a_bat_b):
            return False, True
        return False, False # same values