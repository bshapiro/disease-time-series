from config import *
import GPy
import numpy as np
import pdb
import pylab as pl
import matplotlib.pyplot as plt
import sys
pl.ion()


def fit_gp(component, x=None):
    # fit a gp to the time series data for this component
    if config['gp_type'] == 'rbf':
        kernel = GPy.kern.RBF(input_dim=1, variance=config['gp_variance'], lengthscale=config['gp_lengthscale'])
    elif config['gp_type'] == 'exp':
        kernel = GPy.kern.Exponential(input_dim=1, variance=config['gp_variance'], lengthscale=config['gp_lengthscale'])
    elif config['gp_type'] == 'mat32':
        kernel = GPy.kern.Matern32(input_dim=1, variance=config['gp_variance'], lengthscale=config['gp_lengthscale'])
    else:
        sys.exit('No kernel specified.')

    if x is None:
        x = np.reshape(np.asfarray(range(len(component))), (len(component), 1))
    else:
        x = np.reshape(x, (len(component), 1))
    y = np.reshape(component, (len(component), 1))

    y_mean = np.mean(y)
    y_centered = y - y_mean

    m = GPy.models.GPRegression(x, y_centered, kernel)

    print m
    m.plot()
    pdb.set_trace()


def fit_gp_with_priors(component, x=None):
    if x is None:
        x = np.reshape(np.asfarray(range(len(component))), (len(component), 1))
    else:
        x = np.reshape(x, (len(component), 1))

    y = np.reshape(component, (len(component), 1))


    y_mean = np.mean(y)
    y_centered = y - y_mean

    kernel = GPy.kern.RBF(input_dim=1)

    m = GPy.models.GPRegression(x, y_centered, kernel)

    m.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(4, 2))
    m.kern.variance.set_prior(GPy.priors.Gamma.from_EV(4, 0.1))
    m.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(0.0001, 0.001))

    hmc = GPy.inference.mcmc.HMC(m,stepsize=5e-4)
    samples = hmc.sample(num_samples=3000)  # Burnin
    plt.plot(samples)
    import pdb; pdb.set_trace()

    samples = hmc.sample(num_samples=1000)
    plt.clf()
    plt.plot(samples)

    m.kern.variance[:] = samples[:,0].mean()
    m.kern.lengthscale[:] = samples[:,1].mean()
    m.likelihood.variance[:] = samples[:,1].mean()

    import pdb; pdb.set_trace()
    m.plot()
