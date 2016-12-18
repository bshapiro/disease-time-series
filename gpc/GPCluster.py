import numpy as np
from GP import fit_gp_with_priors, fit_gp
from config import config


class GPCluster:

    def __init__(self, gp, name, link=None):
        self.gp = gp
        self.name = name
        self.samples = []
        self.sample_indices = []
        self.link = None

    def add_link(self, cluster):
        self.link = cluster

    def likelihood(self, y, x, index):
        x = np.reshape(np.asarray(x), (len(x), 1))
        y = np.reshape(np.asarray(y), (len(y), 1))
        log_likelihood = self.gp.log_predictive_density(x, y)
        if self.link is not None:
            if self.link.contains_sample(index):
                log_likelihood = log_likelihood * (1.0 - config['strength'])
        return sum(log_likelihood)

    def assign_sample(self, sample, index):
        self.samples.append(sample)
        self.sample_indices.append(index)

    def contains_sample(self, index):
        return index in set(self.sample_indices)

    def clear_samples(self):
        self.samples = []
        self.sample_indices = []

    def reestimate(self, iteration):
        print 'Cluster', self.name + ';\t', '# of samples:', len(self.samples)
        if config['kernel'] == 'stack':
            samples = np.asarray(self.samples).reshape(len(self.samples) * self.samples[0].shape[0])
            gp = fit_gp(samples, range(0, len(self.samples[0])) * len(self.samples), self.name)
        elif config['kernel'] == 'avg':
            samples = np.mean(np.asarray(self.samples).reshape(len(self.samples), self.samples[0].shape[0]), 0)
            # print 'Sample mean: ', samples
            gp = fit_gp(samples, range(len(self.samples[0])), self.name + 'iter' + str(iteration))

        self.gp = gp
