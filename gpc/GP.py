import GPy
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from config import config
from helpers import generate_output_dir
pl.ion()


def fit_gp(component, x=None, name=None):
    lengthscale = config['gp_lengthscale']
    variance = config['gp_variance']
    if lengthscale is not None and variance is not None:
        kernel = GPy.kern.Exponential(input_dim=1, variance=variance, lengthscale=lengthscale)
    else:
        kernel = GPy.kern.RBF(input_dim=1)

    if x is None:
        x = np.reshape(np.asfarray(range(len(component))), (len(component), 1))
    else:
        x = np.reshape(x, (len(component), 1))
    y = np.reshape(component, (len(component), 1))

    m = GPy.models.GPRegression(x, y, kernel)

    if name.startswith('polya'):
        m.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1, 1), warning=False)
        m.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1, 1), warning=False)
        m.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(0.001, 0.00001), warning=False)
        # m.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(0.001, 0.0001), warning=False)

    if name.startswith('ribosome'):
        m.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1, 1), warning=False)
        m.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(0.5, 0.5), warning=False)
        m.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(0.001, 0.00001), warning=False)

    m.optimize()
    m.plot()

    # print name
    # print m

    plt.savefig(generate_output_dir() + name + '_m.png')
    plt.close()

    return m


def fit_gp_with_priors(component, x=None, name=None):
    if x is None:
        x = np.reshape(np.asfarray(range(len(component))), (len(component), 1))
    else:
        x = np.reshape(x, (len(component), 1))

    y = np.reshape(component, (len(component), 1))

    kernel = GPy.kern.RBF(input_dim=1)

    m = GPy.models.GPRegression(x, y, kernel)

    m.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1, 1))
    m.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1, 1))
    m.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(0.001, 0.0001))

    hmc = GPy.inference.mcmc.HMC(m, stepsize=1e-2)
    samples = hmc.sample(num_samples=500)  # Burnin
    plt.plot(samples)
    plt.savefig(generate_output_dir() + name + '_burnin.png')
    plt.clf()

    samples = hmc.sample(num_samples=1000)
    plt.clf()
    plt.plot(samples)
    plt.savefig(generate_output_dir() + name + '_samples.png')
    plt.clf()

    m.kern.variance[:] = samples[:, 0].mean()
    m.kern.lengthscale[:] = samples[:, 1].mean()
    m.likelihood.variance[:] = samples[:, 2].mean()

    # print m

    m.plot()
    plt.savefig(generate_output_dir() + name + '_m.png')
    plt.close()

    return m
