# -*- coding: utf-8 -*-

import math
import numpy as np
from numpy import cos, sin
from sympy import *

class Minimizer:    
    def __init__(self, func, args, props):
        
        self.func = func
        self.args = args 
        self.alpha = props["alpha"] if "alpha" in props else 0.001
        self.iter = props["iter"] if "iter" in props else 500
        self.eps = props["eps"] if "eps" in props else 1e-5
        self.df = self.calculate_partial_derivatives()
        
        
    def calculate_partial_derivatives(self):
        dfs = []
        for arg in self.args:
            dfn = Derivative(self.func, arg)
            dfd = dfn.doit()
            df2l = lambdify(tuple(self.args), dfd)
            dfs.append(df2l)
        return dfs
    
    
    def dfx(self, x):
        derivatives = []
        for i in range(len(x)):
            d = self.df[i](*x)
            derivatives.append(d)
        return derivatives
    
    
    def minimize(self, x0):
        x0 = np.array(x0)
        prev_t = x0-10*self.eps
        t = x0.copy()
        iter = 0
        while np.linalg.norm(t - prev_t) > self.eps and iter < self.iter:
            prev_t = t.copy()
            diff = self.dfx(t)            
            t -= self.alpha*np.array(self.dfx(t))
            iter += 1
        return t
        
         

class BasinHopper:
    
    def __init__(self, funcstr, args, x0, minimizer, T, stepsize, target_accept_rate, interval, factor):
        
        self.x = x0
        self.func = lambdify(args, funcstr)
        self.minimizer = minimizer
        self.T = 1.0/T if T!=0 else float('inf')
        self.stepsize = stepsize        
        
        min = self.minimizer.minimize(x0)   # minimization is done
        self.x = min                        # on initialization 
        self.res = self.x

        # adaptive stepsize props
        self.target_accept_rate = target_accept_rate
        self.interval = interval
        self.factor = factor

        self.nstep = 0
        self.nstep_tot = 0
        self.naccept = 0

        self.lowest = self.res


    def setup_func(self, funcstr):
        return lambdify()


    def monte_carlo_step(self):
        xnew = np.copy(self.x)
        
        xnew = self.take_step(xnew)     # 1) perturbation of coordinates
        
        minres = self.minimizer.minimize(xnew)  # 2) minimization
        
        accept = self.metropolis_criterion(self.x, minres)  # 3) accept/reject step
        if accept:                                          # based on Metropolis criterion
            self.x = minres
            self.naccept += 1
        # update lowest if necessary
        if self.func(*minres) < self.func(*self.lowest): self.lowest = minres
        

    def adjust_step_size(self):
        accept_rate = float(self.naccept) / self.nstep
        if accept_rate > self.target_accept_rate: # accepting too many steps, trapped in basin; take bigger steps
            self.stepsize /= self.factor
        else:   # we're not accepting enought steps, take smaller steps
            self.stepsize *= self.factor


    def take_step(self, x):   
        self.nstep += 1
        self.nstep_tot += 1
        if self.nstep % self.interval == 0:
            self.adjust_step_size()  
        x += np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))
        return x


    def metropolis_criterion(self, x_old, x_new):
        p = math.exp(min(0,-(self.func(*x_new) - self.func(*x_old)) * self.T)) 
        rand = np.random.uniform()
        return p >= rand


def get_args(func):
    sf = sympify(func)
    var_n = len(sf.free_symbols)
    args = list(map(lambda i: 'x' + str(i) , [*range(var_n)]))
    return args
    
    
def basin_hopping(func, x0, minimizer_props={}, niter=200, T=1.0, stepsize=0.5, 
                  target_accept_rate=0.5, interval=50, factor=0.9):    
    
    args = get_args(func)
    
    minimizer =  Minimizer(sympify(func), args, minimizer_props)
    bh = BasinHopper(func, args, x0, minimizer, T, stepsize, target_accept_rate, interval, factor)
    
    for _ in range(niter):
        bh.monte_carlo_step()
   
    return bh.lowest, bh.func(*bh.lowest)

