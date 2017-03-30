import numpy as np
from scipy.integrate import ode
import pylab as pl

# set equation of motion for harmonic oscillator 
def dxdt(t,y,coefficient_matrix):
    r"""Harmonic oscillator:
        m d^2x/dt^2 = -k x
        
        in momentum space:
        dx/dt = p
        dp/dt = (-k/m) x

        as vector equation
         d   / x \     / 0  -k/m \   / x \    dy
        --- |     | = |           | |     | = --
        d t  \ p /     \ 1    0  /   \ p /    dt
    """

    return coefficient_matrix.dot( y )

# set parameters of harmonic oscillator problem 
m = 1.0 # mass
k = 1.0 # spring constant
coefficient_matrix = np.array([ [0,1], [-k/m,0]])

# initial values
x_0 = 0 # intial position
p_0 = 1 # initial momentum
t_0 = 0 # intitial time

# initial y-vector from initial position and momentum
x0 = np.array([x_0,p_0]) 

# initialize integrator
r = ode(dxdt)

# Runge-Kutta with step size control
r.set_integrator('dopri5')

# set initial values
r.set_initial_value(x0,t_0)

# set coefficient matrix to pass to dx/dt
r.set_f_params(coefficient_matrix)

# max value of time and points in time to integrate to
t_max = 10
N_spacing_in_t = 100

# create vector of time points you want to evaluate
t = np.linspace(t_0,t_max,N_spacing_in_t)

# create vector of positions for those times
x_result = np.zeros_like(t)

# loop through all demanded time points
for it, t_ in enumerate(t):

    # get result of ODE integration
    y = r.integrate(t_)

    # write result to result vector
    x_result[it] = y[0]

# plot result
pl.plot(t,x_result)
pl.xlabel('time $t$')
pl.ylabel('position $x$')
pl.show()

