1.
import random

class LaplaceDistribution:
    def __init__(self, rand, loc, scale):
        if scale <= 0:
            raise ValueError("A skála (scale) értéke pozitív valós szám kell legyen.")
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def sample(self):
        u = self.rand.random()
        if u < 0.5:
            return self.loc + self.scale * (-(2.0 * u) ** 0.5)
        else:
            return self.loc - self.scale * ((2.0 * (1 - u)) ** 0.5)

# Példa használat
if __name__ == "__main__":
    random_generator = random.Random()
    laplace = LaplaceDistribution(random_generator, loc=0, scale=1.0)

    # Generáljunk 10 Laplace eloszlású mintát
    for _ in range(10):
        sample = laplace.sample()
        print(f"Laplace minta: {sample:.4f}")

2.
import random
import math

class LaplaceDistribution:
    def __init__(self, rand, loc, scale):
        if scale <= 0:
            raise ValueError("A skála (scale) értéke pozitív valós szám kell legyen.")
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def sample(self):
        u = self.rand.random()
        if u < 0.5:
            return self.loc + self.scale * (-(2.0 * u) ** 0.5)
        else:
            return self.loc - self.scale * ((2.0 * (1 - u)) ** 0.5)

    def pdf(self, x):

abs_diff = abs(x - self.loc)
probability = (1 / (2 * self.scale)) * math.exp(-abs_diff / self.scale)
return probability

# Példa használat
if __name__ == "__main__":
random_generator = random.Random()
laplace = LaplaceDistribution(random_generator, loc=0, scale=1.0)

x = 0.5
probability = laplace.pdf(x)
print(f"Az aszimmetrikus Laplace eloszlás valószínűségi sűrűségfüggvénye x={x}: {probability:.4f}")

3.
import math

class LaplaceDistribution:
def __init__(self, mu, b):
self.mu = mu  # Az eloszlás középértéke
self.b = b    # Az eloszlás skálája

def pdf(self, x):
# Laplace-eloszlás valószínűségi sűrűségfüggvénye
return 0.5 * math.exp(-abs(x - self.mu) / self.b) / self.b

def cdf(self, x):
# Laplace-eloszlás kumulatív eloszlásfüggvénye
if x < self.mu:
    return 0.5 * math.exp((x - self.mu) / self.b)
else:
    return 1 - 0.5 * math.exp(-(x - self.mu) / self.b)

# Példa használat:
laplace = LaplaceDistribution(0, 1)
x = 1.5
cdf_value = laplace.cdf(x)
print(f"CDF({x}) = {cdf_value}")

4.
import math

class LaplaceDistribution:
def __init__(self, mu, b):
self.mu = mu  # Az eloszlás középértéke
self.b = b    # Az eloszlás skálája

def pdf(self, x):
# Laplace-eloszlás valószínűségi sűrűségfüggvénye
return 0.5 * math.exp(-abs(x - self.mu) / self.b) / self.b

def cdf(self, x):
# Laplace-eloszlás kumulatív eloszlásfüggvénye
if x < self.mu:
    return 0.5 * math.exp((x - self.mu) / self.b)
else:
    return 1 - 0.5 * math.exp(-(x - self.mu) / self.b)

def ppf(self, p):
# Laplace-eloszlás inverz kumulatív eloszlásfüggvénye
if p < 0.5:
    return self.mu + self.b * math.log(2 * p)
else:
    return self.mu - self.b * math.log(2 - 2 * p)

# Példa használat:
laplace = LaplaceDistribution(0, 1)
p = 0.3
ppf_value = laplace.ppf(p)
print(f"PPF({p}) = {ppf_value}")

5.
import math
import random

class LaplaceDistribution:
def __init__(self, mu, b):
self.mu = mu  # Az eloszlás középértéke
self.b = b    # Az eloszlás skálája

def pdf(self, x):
# Laplace-eloszlás valószínűségi sűrűségfüggvénye
return 0.5 * math.exp(-abs(x - self.mu) / self.b) / self.b

def cdf(self, x):
# Laplace-eloszlás kumulatív eloszlásfüggvénye
if x < self.mu:
    return 0.5 * math.   
    exp((x - self.mu) / self.b)
else:
    return 1 - 0.5 * math.exp(-(x - self.mu) / self.b)

def ppf(self, p):
# Laplace-eloszlás inverz kumulatív eloszlásfüggvénye
if p < 0.5:
    return self.mu + self.b * math.log(2 * p)
else:
    return self.mu - self.b * math.log(2 - 2 * p)

def gen_random(self):
# Generál egy aszimmetrikus Laplace-eloszlású véletlen számot
u = random.uniform(0, 1)
if u < 0.5:
    return self.mu + self.b * math.log(2 * u)
else:
    return self.mu - self.b * math.log(2 - 2 * u)

# Példa használat:
laplace = LaplaceDistribution(0, 1)
random_value = laplace.gen_random()
print(f"Generált véletlen szám: {random_value}")

6.
import math

class LaplaceDistribution:
def __init__(self, mu, b):
self.mu = mu  # Az eloszlás középértéke
self.b = b    # Az eloszlás skálája

def pdf(self, x):
# Laplace-eloszlás valószínűségi sűrűségfüggvénye
return 0.5 * math.exp(-abs(x - self.mu) / self.b) / self.b

def cdf(self, x):
# Laplace-eloszlás kumulatív eloszlásfüggvénye
if x < self.mu:
    return 0.5 * math.exp((x - self.mu) / self.b)
else:
    return 1 - 0.5 * math.exp(-(x - self.mu) / self.b)

def ppf(self, p):
# Laplace-eloszlás inverz kumulatív eloszlásfüggvénye
if p < 0.5:
    return self.mu + self.b * math.log(2 * p)
else:
    return self.mu - self.b * math.log(2 - 2 * p)

def mean(self):
# Várható érték a középérték (mu) alapján
return self.mu

# Példa használat:
laplace = LaplaceDistribution(0, 1)
mean_value = laplace.mean()
print(f"Várható érték: {mean_value}")

7.
import math

class LaplaceDistribution:
def __init__(self, mu, b):
self.mu = mu  # Az eloszlás középértéke
self.b = b    # Az eloszlás skálája

def pdf(self, x):
# Laplace-eloszlás valószínűségi sűrűségfüggvénye
return 0.5 * math.exp(-abs(x - self.mu) / self.b) / self.b

def cdf(self, x):
# Laplace-eloszlás kumulatív eloszlásfüggvénye
if x < self.mu:
    return 0.5 * math.exp((x - self.mu) / self.b)
else:
    return 1 - 0.5 * math.exp(-(x - self.mu) / self.b)

def ppf(self, p):
# Laplace-eloszlás inverz kumulatív eloszlásfüggvénye
if p < 0.5:
    return self.mu + self.b * math.log(2 * p)
else:
    return self.mu - self.b * math.log(2 - 2 * p)

def mean(self):
# Várható érték a középérték (mu) alapján
return self.mu

def variance(self):
# Variancia számítása
return 2 * self.b ** 2

# Példa használat:
laplace = LaplaceDistribution(0, 1)
variance_value = laplace.variance()
print(f"Variancia: {variance_value}")

8.
import math

class LaplaceDistribution:
def __init__(self, mu, b):
self.mu = mu  # Az eloszlás középértéke
self.b = b    # Az eloszlás skálája

def pdf(self, x):
# Laplace-eloszlás valószínűségi sűrűségfüggvénye
return 0.5 * math.exp(-abs(x - self.mu) / self.b) / self.b

def cdf(self, x):
# Laplace-eloszlás kumulatív eloszlásfüggvénye
if x < self.mu:
    return 0.5 * math.exp((x - self.mu) / self.b)
else:
    return 1 - 0.5 * math.exp(-(x - self.mu) / self.b)

def ppf(self, p):
# Laplace-eloszlás inverz kumulatív eloszlásfüggvénye
if p < 0.5:
    return self.mu + self.b * math.log(2 * p)
else:
    return self.mu - self.b * math.log(2 - 2 * p)

def mean(self):
# Várható érték a középérték (mu) alapján
return self.mu

def variance(self):
# Variancia számítása
return 2 * self.b ** 2

def skewness(self):
# Ferdeség számítása
return 0

# Példa használat:
laplace = LaplaceDistribution(0, 1)
skewness_value = laplace.skewness()
print(f"Ferdeség: {skewness_value}")

9.
import math

class LaplaceDistribution:
def __init__(self, mu, b):
self.mu = mu  # Az eloszlás középértéke
self.b = b    # Az eloszlás skálája

def pdf(self, x):
# Laplace-eloszlás valószínűségi sűrűségfüggvénye
return 0.5 * math.exp(-abs(x - self.mu) / self.b) / self.b

def cdf(self, x):
# Laplace-eloszlás kumulatív eloszlásfüggvénye
if x < self.mu:
    return 0.5 * math.exp((x - self.mu) / self.b)
else:
    return 1 - 0.5 * math.exp(-(x - self.mu) / self.b)

def ppf(self, p):
# Laplace-eloszlás inverz kumulatív eloszlásfüggvénye
if p < 0.5:
    return self.mu + self.b * math.log(2 * p)
else:
    return self.mu - self.b * math.log(2 - 2 * p)

def mean(self):
# Várható érték a középérték (mu) alapján
return self.mu

def variance(self):
# Variancia számítása
return 2 * self.b ** 2

def skewness(self):
# Ferdeség számítása
return 0

def ex_kurtosis(self):
# Többlet csúcsosság számítása
return 3

# Példa használat:
laplace = LaplaceDistribution(0, 1)
ex_kurtosis_value = laplace.ex_kurtosis()
print(f"Többlet csúcsosság: {ex_kurtosis_value}")

10.
import math

class LaplaceDistribution:
def __init__(self, mu, b):
self.mu = mu  # Az eloszlás középértéke
self.b = b    # Az eloszlás skálája

def pdf(self, x):
# Laplace-eloszlás valószínűségi sűrűségfüggvénye
return 0.5 * math.exp(-abs(x - self.mu) / self.b) / self.b

def cdf(self, x):
# Laplace-eloszlás kumulatív eloszlásfüggvénye
if x < self.mu:
    return 0.5 * math.exp((x - self.mu) / self.b)
else:
    return 1 - 0.5 * math.exp(-(x - self.mu) / self.b)

def ppf(self, p):
# Laplace-eloszlás inverz kumulatív eloszlásfüggvénye
if p < 0.5:
    return self.mu + self.b * math.log(2 * p)
else:
    return self.mu - self.b * math.log(2 - 2 * p)

def mean(self):
# Várható érték a középérték (mu) alapján
return self.mu

def variance(self):
# Variancia számítása
return 2 * self.b ** 2

def skewness(self):
# Ferdeség számítása
return 0

def ex_kurtosis(self):
# Többlet csúcsosság számítása
return 3

def mvsk(self):
# Eloszlás momentumok és többlet csúcsosságának számítása
if self.b == 0:
    raise Exception("Moment undefined")

mean = self.mean()
variance = self.variance()
skewness = self.skewness()
kurtosis = self.ex_kurtosis()

return [mean, variance, skewness, kurtosis]

# Példa használat:
laplace = LaplaceDistribution(0, 1)
moments = laplace.mvsk()
print(f"Momentumok és többlet csúcsosság: {moments}")

11.
class ParetoDistribution:
    def __init__(self, rand, scale, shape):
        self.rand = rand  # A véletlenszám generátor
        self.scale = scale  # Skála paraméter (scale)
        self.shape = shape  # Alak paraméter (shape)

    # További függvényeket is lehet definiálni az osztályban az eloszlással kapcsolatban.


# Példa használat:
# Az alábbi példa bemutatja, hogyan hozzuk létre egy ParetoDistribution objektumot:
import random

scale = 1.0
shape = 2.1
random_generator = random.Random()
pareto = ParetoDistribution(random_generator, scale, shape)

12.
class ParetoDistribution:
    def __init__(self, rand, loc, scale, shape):
        self.rand = rand
        self.loc = loc
        self.scale = scale
        self.shape = shape

    def pdf(self, x):
        if x < self.loc:
            return 0
        else:
            return (self.shape / self.scale) * (1 + (x - self.loc) / self.scale) ** (-self.shape - 1)


# Példa használat:
# Pareto eloszlás paramétereinek beállítása
loc = 1.0
scale = 2.0
shape = 2.1

random_generator = random.Random()
pareto = ParetoDistribution(random_generator, loc, scale, shape)

# Valószínűségi sűrűségfüggvény értékének kiszámítása egy adott x értékre
x = 3.0
pdf_value = pareto.pdf(x)
print(f"Pareto PDF at x = {x}: {pdf_value}")

13.
class ParetoDistribution:
    def __init__(self, rand, loc, scale, shape):
        self.rand = rand
        self.loc = loc
        self.scale = scale
        self.shape = shape

    def pdf(self, x):
        if x < self.loc:
            return 0
        else:
            return (self.shape / self.scale) * (1 + (x - self.loc) / self.scale) ** (-self.shape - 1)

    def cdf(self, x):
        if x < self.loc:
            return 0
        else:
            return 1 - (self.scale / (x - self.loc)) ** self.shape


# Példa használat:
# Pareto eloszlás paramétereinek beállítása
loc = 1.0
scale = 2.0
shape = 2.1

random_generator = random.Random()
pareto = ParetoDistribution(random_generator, loc, scale, shape)

# Kumulatív eloszlásfüggvény értékének kiszámítása egy adott x értékre
x = 3.0
cdf_value = pareto.cdf(x)
print(f"Pareto CDF at x = {x}: {cdf_value}")

14.
class ParetoDistribution:
    def __init__(self, rand, loc, scale, shape):
        self.rand = rand
        self.loc = loc
        self.scale = scale
        self.shape = shape

    def pdf(self, x):
        if x < self.loc:
            return 0
        else:
            return (self.shape / self.scale) * (1 + (x - self.loc) / self.scale) ** (-self.shape - 1)

    def cdf(self, x):
        if x < self.loc:
            return 0
        else:
            return 1 - (self.scale / (x - self.loc)) ** self.shape

    def ppf(self, p):
        if p <= 0 or p >= 1:
            raise ValueError("Invalid p value. p must be in the open interval (0, 1).")
        return self.loc + self.scale / (p ** (1 / self.shape))


# Példa használat:
# Pareto eloszlás paramétereinek beállítása
loc = 1.0
scale = 2.0
shape = 2.1

random_generator = random.Random()
pareto = ParetoDistribution(random_generator, loc, scale, shape)

# Inverz kumulatív eloszlásfüggvény értékének kiszámítása egy adott p valószámhoz
p = 0.7
ppf_value = pareto.ppf(p)
print(f"Pareto PPF at p = {p}: {ppf_value}")

15.
class ParetoDistribution:
    def __init__(self, rand, loc, scale, shape):
        self.rand = rand
        self.loc = loc
        self.scale = scale
        self.shape = shape

    def pdf(self, x):
        if x < self.loc:
            return 0
        else:
            return (self.shape / self.scale) * (1 + (x - self.loc) / self.scale) ** (-self.shape - 1)

    def cdf(self, x):
        if x < self.loc:
            return 0
        else:
            return 1 - (self.scale / (x - self.loc)) ** self.shape

    def ppf(self, p):
        if p <= 0 or p >= 1:
            raise ValueError("Invalid p value. p must be in the open interval (0, 1).")
        return self.loc + self.scale / (p ** (1 / self.shape))

    def gen_random(self):
        # Aszimmetrikus Laplace eloszlás generálása Pareto eloszlás alapján
        p = self.rand.random()
        return self.ppf(p)


# Példa használat:
# Pareto eloszlás paramétereinek beállítása
loc = 1.0
scale = 2.0
shape = 2.1

random_generator = random.Random()
pareto = ParetoDistribution(random_generator, loc, scale, shape)

# Aszimmetrikus Laplace eloszlású véletlen szám generálása
random_value = pareto.gen_random()
print(f"Aszimmetrikus Laplace eloszlású véletlen szám: {random_value}")

16.
class ParetoDistribution:
    def __init__(self, rand, loc, scale, shape):
        self.rand = rand
        self.loc = loc
        self.scale = scale
        self.shape = shape

    def pdf(self, x):
        if x < self.loc:
            return 0
        else:
            return (self.shape / self.scale) * (1 + (x - self.loc) / self.scale) ** (-self.shape - 1)

    def cdf(self, x):
        if x < self.loc:
            return 0
        else:
            return 1 - (self.scale / (x - self.loc)) ** self.shape

    def ppf(self, p):
        if p <= 0 or p >= 1:
            raise ValueError("Invalid p value. p must be in the open interval (0, 1).")
        return self.loc + self.scale / (p ** (1 / self.shape))

    def gen_random(self):
        p = self.rand.random()
        return self.ppf(p)

    def mean(self):
        if self.shape <= 1:
            raise Exception("Moment undefined")
        return (self.scale * self.shape) / (self.shape - 1) + self.loc


# Példa használat:
loc = 1.0
scale = 2.0
shape = 2.1

random_generator = random.Random()
pareto = ParetoDistribution(random_generator, loc, scale, shape)

mean_value = pareto.mean()
print(f"Pareto Mean: {mean_value}")

17.
class ParetoDistribution:
    def __init__(self, rand, loc, scale, shape):
        self.rand = rand
        self.loc = loc
        self.scale = scale
        self.shape = shape

    def pdf(self, x):
        if x < self.loc:
            return 0
        else:
            return (self.shape / self.scale) * (1 + (x - self.loc) / self.scale) ** (-self.shape - 1)

    def cdf(self, x):
        if x < self.loc:
            return 0
        else:
            return 1 - (self.scale / (x - self.loc)) ** self.shape

    def ppf(self, p):
        if p <= 0 or p >= 1:
            raise ValueError("Invalid p value. p must be in the open interval (0, 1).")
        return self.loc + self.scale / (p ** (1 / self.shape))

    def gen_random(self):
        p = self.rand.random()
        return self.ppf(p)

    def mean(self):
        if self.shape <= 1:
            raise Exception("Moment undefined")
        return (self.scale * self.shape) / (self.shape - 1) + self.loc

    def variance(self):
        if self.shape <= 2:
            raise Exception("Moment undefined")
        return (self.scale ** 2 * self.shape) / ((self.shape - 1) ** 2 * (self.shape - 2))


# Példa használat:
loc = 1.0
scale = 2.0
shape = 2.1

random_generator = random.Random()
pareto = ParetoDistribution(random_generator, loc, scale, shape)

variance_value = pareto.variance()
print(f"Pareto Variance: {variance_value}")

18.
class ParetoDistribution:
    def __init__(self, rand, loc, scale, shape):
        self.rand = rand
        self.loc = loc
        self.scale = scale
        self.shape = shape

    def pdf(self, x):
        if x < self.loc:
            return 0
        else:
            return (self.shape / self.scale) * (1 + (x - self.loc) / self.scale) ** (-self.shape - 1)

    def cdf(self, x):
        if x < self.loc:
            return 0
        else:
            return 1 - (self.scale / (x - self.loc)) ** self.shape

    def ppf(self, p):
        if p <= 0 or p >= 1:
            raise ValueError("Invalid p value. p must be in the open interval (0, 1).")
        return self.loc + self.scale / (p ** (1 / self.shape))

    def gen_random(self):
        p = self.rand.random()
        return self.ppf(p)

    def mean(self):
        if self.shape <= 1:
            raise Exception("Moment undefined")
        return (self.scale * self.shape) / (self.shape - 1) + self.loc

    def variance(self):
        if self.shape <= 2:
            raise Exception("Moment undefined")
        return (self.scale ** 2 * self.shape) / ((self.shape - 1) ** 2 * (self.shape - 2))

    def skewness(self):
        if self.shape <= 4:
            raise Exception("Moment undefined")
        return (2 * (1 + self.shape) / (self.shape - 3)) * (self.shape - 2) / ((self.shape - 4) ** 0.5)


# Példa használat:
loc = 1.0
scale = 2.0
shape = 2.1

random_generator = random.Random()
pareto = ParetoDistribution(random_generator, loc, scale, shape)

skewness_value = pareto.skewness()
print(f"Pareto Skewness: {skewness_value}")

19.
class ParetoDistribution:
    def __init__(self, rand, loc, scale, shape):
        self.rand = rand
        self.loc = loc
        self.scale = scale
        self.shape = shape

    def pdf(self, x):
        if x < self.loc:
            return 0
        else:
            return (self.shape / self.scale) * (1 + (x - self.loc) / self.scale) ** (-self.shape - 1)

    def cdf(self, x):
        if x < self.loc:
            return 0
        else:
            return 1 - (self.scale / (x - self.loc)) ** self.shape

    def ppf(self, p):
        if p <= 0 or p >= 1:
            raise ValueError("Invalid p value. p must be in the open interval (0, 1).")
        return self.loc + self.scale / (p ** (1 / self.shape))

    def gen_random(self):
        p = self.rand.random()
        return self.ppf(p)

    def mean(self):
        if self.shape <= 1:
            raise Exception("Moment undefined")
        return (self.scale * self.shape) / (self.shape - 1) + self.loc

    def variance(self):
        if self.shape <= 2:
            raise Exception("Moment undefined")
        return (self.scale ** 2 * self.shape) / ((self.shape - 1) ** 2 * (self.shape - 2))

    def skewness(self):
        if self.shape <= 4:
            raise Exception("Moment undefined")
        return (2 * (1 + self.shape) / (self.shape - 3)) * (self.shape - 2) / ((self.shape - 4) ** 0.5)

    def ex_kurtosis(self):
        if self.shape <= 4:
            raise Exception("Moment undefined")
        return (6 * (self.shape ** 3 + self.shape ** 2 - 6 * self.shape - 2)) / (
                    self.shape * (self.shape - 3) * (self.shape - 4))


# Példa használat:
loc = 1.0
scale = 2.0
shape = 2.1

random_generator = random.Random()
pareto = ParetoDistribution(random_generator, loc, scale, shape)

ex_kurtosis_value = pareto.ex_kurtosis()
print(f"Pareto Excess Kurtosis: {ex_kurtosis_value}")

20.
class ParetoDistribution:
    def __init__(self, rand, loc, scale, shape):
        self.rand = rand
        self.loc = loc
        self.scale = scale
        self.shape = shape

    def pdf(self, x):
        if x < self.loc:
            return 0
        else:
            return (self.shape / self.scale) * (1 + (x - self.loc) / self.scale) ** (-self.shape - 1)

    def cdf(self, x):
        if x < self.loc:
            return 0
        else:
            return 1 - (self.scale / (x - self.loc)) ** self.shape

    def ppf(self, p):
        if p <= 0 or p >= 1:
            raise ValueError("Invalid p value. p must be in the open interval (0, 1).")
        return self.loc + self.scale / (p ** (1 / self.shape))

    def gen_random(self):
        p = self.rand.random()
        return self.ppf(p)

    def mean(self):
        if self.shape <= 1:
            raise Exception("Moment undefined")
        return (self.scale * self.shape) / (self.shape - 1) + self.loc

    def variance(self):
        if self.shape <= 2:
            raise Exception("Moment undefined")
        return (self.scale ** 2 * self.shape) / ((self.shape - 1) ** 2 * (self.shape - 2))

    def skewness(self):
        if self.shape <= 4:
            raise Exception("Moment undefined")
        return (2 * (1 + self.shape) / (self.shape - 3)) * (self.shape - 2) / ((self.shape - 4) ** 0.5)

    def ex_kurtosis(self):
        if self.shape <= 4:
            raise Exception("Moment undefined")
        return (6 * (self.shape**3 + self.shape**2 - 6*self.shape - 2)) / (self.shape * (self.shape - 3) * (self.shape - 4))

    def mvsk(self):
        if self.shape <= 4:
            raise Exception("Moment undefined")
        mean = self.mean()
        variance = self.variance()
        skewness = self.skewness()
        ex_kurtosis = self.ex_kurtosis()
        return [mean, variance, skewness, ex_kurtosis]

# Példa használat:
loc = 1.0
scale = 2.0
shape = 2.1

random_generator = random.Random()
pareto = ParetoDistribution(random_generator, loc, scale, shape)

mvsk_values = pareto.mvsk()
print("Pareto MVSK values:")
print(f"Mean: {mvsk_values[0]}")
print(f"Variance: {mvsk_values[1]}")
print(f"Skewness: {mvsk_values[2]}")
print(f"Excess Kurtosis: {mvsk_values[3]}")

