import math

class LogisticDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.location = loc
        self.scale = scale

    def pdf(self, x):
        exponent = math.exp(-(x - self.location) / self.scale)
        pdf = exponent / (self.scale * (1 + exponent) ** 2)
        return pdf

    def cdf(self, x):
        cdf = 1 / (1 + math.exp(-(x - self.location) / self.scale))
        return cdf

    def ppf(self, p):
        if 0 < p < 1:
            ppf = self.location - self.scale * math.log(1 / p - 1)
            return ppf
        else:
            raise ValueError("p must be in the range (0, 1)")

    def gen_rand(self):
        u = self.rand.random()
        rand_num = self.location + self.scale * math.log(u / (1 - u))
        return rand_num

    def mean(self):
        return self.location

    def variance(self):
        return (math.pi ** 2) * (self.scale ** 2) / 3

    def skewness(self):
        return 0

    def ex_kurtosis(self):
        return 1.2

    def mvsk(self):
        mean = self.mean()
        variance = self.variance()
        skewness = self.skewness()
        excess_kurtosis = self.ex_kurtosis()
        return [mean, variance, skewness, excess_kurtosis]

import random
import typing
import math
import scipy.special

class ChiSquaredDistribution:
    def __init__(self, rand, dof):
        self.rand = rand
        self.dof = dof

    def pdf(self, x):
        if x < 0:
            return 0
        numerator = x ** ((self.dof / 2) - 1) * math.exp(-x / 2)
        denominator = (2 ** (self.dof / 2)) * scipy.special.gamma(self.dof / 2)
        return numerator / denominator

    def cdf(self, x):
        if x < 0:
            return 0
        return scipy.special.gammainc(self.dof / 2, x / 2)

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("p must be in the range [0, 1]")
        return 2 * scipy.special.gammaincinv(self.dof / 2, p)

    def gen_rand(self):
        # Generate a random number using a method of your choice
        # For example, you can use the inverse transform sampling method
        u = self.rand.random()
        return self.ppf(u)

    def mean(self):
        try:
            return self.dof
        except:
            raise ValueError("Moment undefined")

    def variance(self):
        return 2 * self.dof

    def skewness(self):
        return math.sqrt(8 / self.dof)

    def ex_kurtosis(self):
        return 12 / self.dof

    def mvsk(self):
        mean = self.mean()
        variance = self.variance()
        skewness = self.skewness()
        ex_kurtosis = self.ex_kurtosis()
        try:
            return [mean, variance, skewness, ex_kurtosis]
        except:
            raise ValueError("Moment undefined")

" Javított verzió "

class UniformDistribution:
    pass

class CauchyDistribution:
    pass