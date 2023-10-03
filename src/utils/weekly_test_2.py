import random
from math import exp, log

class LaplaceDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        diff = abs(x - self.loc)
        return (1 / (2 * self.scale)) * exp(-diff / self.scale)

    def cdf(self, x):
        if x < self.loc:
            return 0.5 * exp((x - self.loc) / self.scale)
        else:
            return 1 - 0.5 * exp(-(x - self.loc) / self.scale)

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("p must be in the range [0, 1]")
        if p < 0.5:
            return self.loc + self.scale * log(2 * p)
        else:
            return self.loc - self.scale * log(2 - 2 * p)

    def gen_random(self):
        u = self.rand.random()
        if u < 0.5:
            return self.loc + self.scale * log(2 * u)
        else:
            return self.loc - self.scale * log(2 - 2 * u)

    def mean(self):
        return self.loc

    def variance(self):
        return 2 * (self.scale ** 2)

    def skewness(self):
        return 0

    def ex_kurtosis(self):
        return 3