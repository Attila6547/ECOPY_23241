"""
1., Hozz létre egy új osztályt, aminek a neve FirstClass
    osztály név: FirstClass
"""
#%%
class FirstClass:
    pass
#%%
"""
2., Hozz létre egy új osztályt, aminek a neve SecondClass, és definiáljon benne egy __init__ nevű függvényt, amely bead egy véletlenszám generátort a self.random adattagnak.
    Osztály név: SecondClass
    függvény név: __init__
    bemenet: self, rand
"""
#%%
import random

class SecondClass:
    def __init__(self, rand):
        self.random = rand

# Példa használat
random_generator = random.Random()
second_instance = SecondClass(random_generator)

# Ellenőrzés, hogy a random attribútum rendelkezésre áll
print(second_instance.random.randint(1, 10))
#%%
"""
3., Hozzan létre egy új osztályt aminek a neve UniformDistribution. Definiáljon benne a __init__ nevű függvényt, amelynek bemenetként kap egy véletlenszám generátort, és az eloszlás alsó (a) és felső határát (b), amely értékeket adattagokba ment le.
    Osztály név: UniformDistribution
    függvény név: __init__
    bemenet: self, rand, a, b
"""
#%%
import random

class UniformDistribution:
    def __init__(self, rand, a, b):
        self.random_generator = rand
        self.lower_limit = a
        self.upper_limit = b

# Példa használat
random_generator = random.Random()
uniform_dist = UniformDistribution(random_generator, 0, 10)

# Ellenőrzés, hogy az adattagok rendelkezésre állnak
print(f"Alsó határ: {uniform_dist.lower_limit}")
print(f"Felső határ: {uniform_dist.upper_limit}")
print(f"Véletlenszerű érték: {uniform_dist.random_generator.uniform(uniform_dist.lower_limit, uniform_dist.upper_limit)}")
#%%
"""

4., Egészítse ki a UniformDistribution osztályt egy új függvénnyel, amely a és b pont közötti értékekhez a hozzájuk tartozó egyenletes eloszlás, eloszlás függvényében hozzárendelt értékét rendeli.
    függvény név: pdf
    bemenet: x
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Continuous_uniform_distribution
"""
#%%
import random

class UniformDistribution:
    def __init__(self, rand, a, b):
        self.random_generator = rand
        self.lower_limit = a
        self.upper_limit = b

    def pdf(self, x):
        if self.lower_limit <= x <= self.upper_limit:
            return 1.0 / (self.upper_limit - self.lower_limit)
        else:
            return 0.0

# Példa használat
random_generator = random.Random()
uniform_dist = UniformDistribution(random_generator, 0, 10)

# Ellenőrzés, hogy a pdf metódus működik
x = 5
probability_density = uniform_dist.pdf(x)
print(f"P({x}) = {probability_density}")

#%%
"""
5., Egészítse ki a UniformDistribution osztályt egy új függvénnyel, amely megvalósítja az egyenletes eloszlás kumulatív eloszlás függvényét.
    függvény név: cdf
    bemenet: x
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Continuous_uniform_distribution
"""
#%%
import random

class UniformDistribution:
    def __init__(self, rand, a, b):
        self.random_generator = rand
        self.lower_limit = a
        self.upper_limit = b

    def pdf(self, x):
        if self.lower_limit <= x <= self.upper_limit:
            return 1.0 / (self.upper_limit - self.lower_limit)
        else:
            return 0.0

    def cdf(self, x):
        if x < self.lower_limit:
            return 0.0
        elif x >= self.upper_limit:
            return 1.0
        else:
            return (x - self.lower_limit) / (self.upper_limit - self.lower_limit)

# Példa használat
random_generator = random.Random()
uniform_dist = UniformDistribution(random_generator, 0, 10)

# Ellenőrzés, hogy a cdf metódus működik
x = 7
cumulative_distribution = uniform_dist.cdf(x)
print(f"CDF({x}) = {cumulative_distribution}")

#%%
"""
6., Egészítse ki a UniformDistribution osztályt egy új függvénnyel, amely implementálja az egyenletes eloszlás inverz kumulatív eloszlás függvényét. (percent-point function)
    függvény név: ppf
    bemenet: p
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Continuous_uniform_distribution
"""
#%%
import random

class UniformDistribution:
    def __init__(self, rand, a, b):
        self.random_generator = rand
        self.lower_limit = a
        self.upper_limit = b

    def pdf(self, x):
        if self.lower_limit <= x <= self.upper_limit:
            return 1.0 / (self.upper_limit - self.lower_limit)
        else:
            return 0.0

    def cdf(self, x):
        if x < self.lower_limit:
            return 0.0
        elif x >= self.upper_limit:
            return 1.0
        else:
            return (x - self.lower_limit) / (self.upper_limit - self.lower_limit)

    def ppf(self, p):
        if 0 <= p <= 1:
            return self.lower_limit + p * (self.upper_limit - self.lower_limit)
        else:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")

# Példa használat
random_generator = random.Random()
uniform_dist = UniformDistribution(random_generator, 0, 10)

# Ellenőrzés, hogy a ppf metódus működik
probability = 0.3
percent_point = uniform_dist.ppf(probability)
print(f"PPF({probability}) = {percent_point}")

#%%
"""
7., Egészítse ki a UniformDistribution osztályt egy új függvénnyel, amely az osztály létrehozásánál megadott paraméterek mellett, egyenletes eloszlású véletlen számokat generál minden meghívásnál.
    függvény név: gen_random
    bemenet: None
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Continuous_uniform_distribution

"""
#%%
import random

class UniformDistribution:
    def __init__(self, rand, a, b):
        self.random_generator = rand
        self.lower_limit = a
        self.upper_limit = b

    def pdf(self, x):
        if self.lower_limit <= x <= self.upper_limit:
            return 1.0 / (self.upper_limit - self.lower_limit)
        else:
            return 0.0

    def cdf(self, x):
        if x < self.lower_limit:
            return 0.0
        elif x >= self.upper_limit:
            return 1.0
        else:
            return (x - self.lower_limit) / (self.upper_limit - self.lower_limit)

    def ppf(self, p):
        if 0 <= p <= 1:
            return self.lower_limit + p * (self.upper_limit - self.lower_limit)
        else:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")

    def mean(self):
        if self.lower_limit == self.upper_limit:
            raise Exception("Moment undefined")
        return (self.lower_limit + self.upper_limit) / 2.0

    def gen_random(self):
        return self.random_generator.uniform(self.lower_limit, self.upper_limit)

# Példa használat
random_generator = random.Random()
uniform_dist = UniformDistribution(random_generator, 0, 10)

# Véletlenszerű szám generálása az osztály paraméterei alapján
random_value = uniform_dist.gen_random()
print(f"Véletlenszerű érték: {random_value}")

#%%
"""
8., Egészítse ki a UniformDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény átlagát.  Ha az eloszlásnak nincsen ilyen értéke, akkor return helyett hívja meg a raise Exception("Moment undefined") parancsot.
    függvény név: mean
    bemenet: None
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Continuous_uniform_distribution
"""
#%%
import random

class UniformDistribution:
    def __init__(self, rand, a, b):
        self.random_generator = rand
        self.lower_limit = a
        self.upper_limit = b

    def pdf(self, x):
        if self.lower_limit <= x <= self.upper_limit:
            return 1.0 / (self.upper_limit - self.lower_limit)
        else:
            return 0.0

    def cdf(self, x):
        if x < self.lower_limit:
            return 0.0
        elif x >= self.upper_limit:
            return 1.0
        else:
            return (x - self.lower_limit) / (self.upper_limit - self.lower_limit)

    def ppf(self, p):
        if 0 <= p <= 1:
            return self.lower_limit + p * (self.upper_limit - self.lower_limit)
        else:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")

    def gen_random(self):
        return self.random_generator.uniform(self.lower_limit, self.upper_limit)

    def mean(self):
        if self.lower_limit == self.upper_limit:
            raise Exception("Moment undefined")
        return (self.lower_limit + self.upper_limit) / 2.0

# Példa használat
random_generator = random.Random()
uniform_dist = UniformDistribution(random_generator, 0, 10)

# Átlag meghívása
try:
    mean_value = uniform_dist.mean()
    print(f"Az eloszlás átlaga: {mean_value}")
except Exception as e:
    print(str(e))

#%%
"""
9., Egészítse ki a UniformDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény mediánját.
    függvény név: median
    bemenet: None
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Continuous_uniform_distribution
"""
#%%
import random

class UniformDistribution:
    def __init__(self, rand, a, b):
        self.random_generator = rand
        self.lower_limit = a
        self.upper_limit = b

    def pdf(self, x):
        if self.lower_limit <= x <= self.upper_limit:
            return 1.0 / (self.upper_limit - self.lower_limit)
        else:
            return 0.0

    def cdf(self, x):
        if x < self.lower_limit:
            return 0.0
        elif x >= self.upper_limit:
            return 1.0
        else:
            return (x - self.lower_limit) / (self.upper_limit - self.lower_limit)

    def ppf(self, p):
        if 0 <= p <= 1:
            return self.lower_limit + p * (self.upper_limit - self.lower_limit)
        else:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")

    def mean(self):
        if self.lower_limit == self.upper_limit:
            raise Exception("Moment undefined")
        return (self.lower_limit + self.upper_limit) / 2.0

    def gen_random(self):
        return self.random_generator.uniform(self.lower_limit, self.upper_limit)

    def median(self):
        return (self.lower_limit + self.upper_limit) / 2.0

# Példa használat
random_generator = random.Random()
uniform_dist = UniformDistribution(random_generator, 0, 10)

# Medián meghívása
median_value = uniform_dist.median()
print(f"Az eloszlás mediánja: {median_value}")

#%%
"""
10., Egészítse ki a UniformDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény varianciáját.  Ha az eloszlásnak nincsen ilyen értéke, akkor return helyett hívja meg a raise Exception("Moment undefined") parancsot.
    függvény név: variance
    bemenet: None
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Continuous_uniform_distribution
"""
#%%
import random

class UniformDistribution:
    def __init__(self, rand, a, b):
        self.random_generator = rand
        self.lower_limit = a
        self.upper_limit = b

    def pdf(self, x):
        if self.lower_limit <= x <= self.upper_limit:
            return 1.0 / (self.upper_limit - self.lower_limit)
        else:
            return 0.0

    def cdf(self, x):
        if x < self.lower_limit:
            return 0.0
        elif x >= self.upper_limit:
            return 1.0
        else:
            return (x - self.lower_limit) / (self.upper_limit - self.lower_limit)

    def ppf(self, p):
        if 0 <= p <= 1:
            return self.lower_limit + p * (self.upper_limit - self.lower_limit)
        else:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")

    def mean(self):
        if self.lower_limit == self.upper_limit:
            raise Exception("Moment undefined")
        return (self.lower_limit + self.upper_limit) / 2.0

    def gen_random(self):
        return self.random_generator.uniform(self.lower_limit, self.upper_limit)

    def variance(self):
        if self.lower_limit == self.upper_limit:
            raise Exception("Moment undefined")
        return ((self.upper_limit - self.lower_limit) ** 2) / 12.0

# Példa használat
random_generator = random.Random()
uniform_dist = UniformDistribution(random_generator, 0, 10)

# Variancia meghívása
variance_value = uniform_dist.variance()
print(f"Az eloszlás varianciája: {variance_value}")

#%%
"""
11., Egészítse ki a UniformDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény ferdeségét.  Ha az eloszlásnak nincsen ilyen értéke, akkor return helyett hívja meg a raise Exception("Moment undefined") parancsot.
    függvény név: skewness
    bemenet: None
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Continuous_uniform_distribution
"""
#%%
import random

class UniformDistribution:
    def __init__(self, rand, a, b):
        self.random_generator = rand
        self.lower_limit = a
        self.upper_limit = b

    def pdf(self, x):
        if self.lower_limit <= x <= self.upper_limit:
            return 1.0 / (self.upper_limit - self.lower_limit)
        else:
            return 0.0

    def cdf(self, x):
        if x < self.lower_limit:
            return 0.0
        elif x >= self.upper_limit:
            return 1.0
        else:
            return (x - self.lower_limit) / (self.upper_limit - self.lower_limit)

    def ppf(self, p):
        if 0 <= p <= 1:
            return self.lower_limit + p * (self.upper_limit - self.lower_limit)
        else:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")

    def mean(self):
        if self.lower_limit == self.upper_limit:
            raise Exception("Moment undefined")
        return (self.lower_limit + self.upper_limit) / 2.0

    def gen_random(self):
        return self.random_generator.uniform(self.lower_limit, self.upper_limit)

    def variance(self):
        if self.lower_limit == self.upper_limit:
            raise Exception("Moment undefined")
        return ((self.upper_limit - self.lower_limit) ** 2) / 12.0

    def skewness(self):
        raise Exception("Moment undefined")

# Példa használat
random_generator = random.Random()
uniform_dist = UniformDistribution(random_generator, 0, 10)

# Ferdeség meghívása
try:
    skewness_value = uniform_dist.skewness()
    print(f"Az eloszlás ferdesége: {skewness_value}")
except Exception as e:
    print(str(e))

#%%
"""
12., Egészítse ki a UniformDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény többlet csúcsosságát.  Ha az eloszlásnak nincsen ilyen értéke, akkor return helyett hívja meg a raise Exception("Moment undefined") parancsot.
    függvény név: ex_kurtosis
    bemenet: None
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Continuous_uniform_distribution
"""
#%%
import random

class UniformDistribution:
    def __init__(self, rand, a, b):
        self.random_generator = rand
        self.lower_limit = a
        self.upper_limit = b

    def pdf(self, x):
        if self.lower_limit <= x <= self.upper_limit:
            return 1.0 / (self.upper_limit - self.lower_limit)
        else:
            return 0.0

    def cdf(self, x):
        if x < self.lower_limit:
            return 0.0
        elif x >= self.upper_limit:
            return 1.0
        else:
            return (x - self.lower_limit) / (self.upper_limit - self.lower_limit)

    def ppf(self, p):
        if 0 <= p <= 1:
            return self.lower_limit + p * (self.upper_limit - self.lower_limit)
        else:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")

    def mean(self):
        if self.lower_limit == self.upper_limit:
            raise Exception("Moment undefined")
        return (self.lower_limit + self.upper_limit) / 2.0

    def gen_random(self):
        return self.random_generator.uniform(self.lower_limit, self.upper_limit)

    def variance(self):
        if self.lower_limit == self.upper_limit:
            raise Exception("Moment undefined")
        return ((self.upper_limit - self.lower_limit) ** 2) / 12.0

    def skewness(self):
        raise Exception("Moment undefined")

    def ex_kurtosis(self):
        raise Exception("Moment undefined")

# Példa használat
random_generator = random.Random()
uniform_dist = UniformDistribution(random_generator, 0, 10)

# Többlet kurtosis meghívása
try:
    kurtosis_value = uniform_dist.ex_kurtosis()
    print(f"Az eloszlás többlet kurtosisa: {kurtosis_value}")
except Exception as e:
    print(str(e))

#%%
"""
13., Egészítse ki a UniformDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény első 3 cetrális momentumát és a többlet csúcsosságot.  Ha az eloszlásnak nincsenek ilyen értékei, akkor return helyett hívja meg a raise Exception("Moments undefined") parancsot.
    függvény név: mvsk
    bemenet: None
    kimeneti típus: List
    link: https://en.wikipedia.org/wiki/Continuous_uniform_distribution
"""
#%%
import random

class UniformDistribution:
    def __init__(self, rand, a, b):
        self.random_generator = rand
        self.lower_limit = a
        self.upper_limit = b

    def pdf(self, x):
        if self.lower_limit <= x <= self.upper_limit:
            return 1.0 / (self.upper_limit - self.lower_limit)
        else:
            return 0.0

    def cdf(self, x):
        if x < self.lower_limit:
            return 0.0
        elif x >= self.upper_limit:
            return 1.0
        else:
            return (x - self.lower_limit) / (self.upper_limit - self.lower_limit)

    def ppf(self, p):
        if 0 <= p <= 1:
            return self.lower_limit + p * (self.upper_limit - self.lower_limit)
        else:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")

    def mean(self):
        if self.lower_limit == self.upper_limit:
            raise Exception("Moments undefined")
        return (self.lower_limit + self.upper_limit) / 2.0

    def gen_random(self):
        return self.random_generator.uniform(self.lower_limit, self.upper_limit)

    def variance(self):
        if self.lower_limit == self.upper_limit:
            raise Exception("Moments undefined")
        return ((self.upper_limit - self.lower_limit) ** 2) / 12.0

    def skewness(self):
        raise Exception("Moments undefined")

    def ex_kurtosis(self):
        raise Exception("Moments undefined")

    def mvsk(self):
        if self.lower_limit == self.upper_limit:
            raise Exception("Moments undefined")

        mean_value = self.mean()
        variance_value = self.variance()
        skewness_value = self.skewness()

        # Az ex_kurtosis a normál eloszlású eloszlás többlet kurtosisa
        ex_kurtosis_value = 0.0

        return [mean_value, variance_value, skewness_value, ex_kurtosis_value]

# Példa használat
random_generator = random.Random()
uniform_dist = UniformDistribution(random_generator, 0, 10)

# Moments meghívása
try:
    moments = uniform_dist.mvsk()
    print(f"Eloszlás moments (mean, variance, skewness, ex_kurtosis): {moments}")
except Exception as e:
    print(str(e))

#%%
"""
14., Hozzan létre egy új osztályt aminek a neve NormalDistribution. Definiáljon benne a __init__ nevű függvényt, amelynek bemenetként kap egy véletlenszám generátort, és az eloszlás várható értékét (location) és varianciáját (scale), amely értékeket adattagokba ment le.
    Osztály név: NormalDistribution
    függvény név: __init__
    bemenet: self, rand, loc, scale
"""
#%%
import random

class NormalDistribution:
    def __init__(self, rand, loc, scale):
        self.random_generator = rand
        self.location = loc
        self.scale = scale

    def gen_random(self):
        return self.random_generator.gauss(self.location, self.scale)

# Példa használat
random_generator = random.Random()
normal_dist = NormalDistribution(random_generator, 0, 1)  # Példa értékekkel inicializálva

# Véletlenszerű szám generálása a normál eloszlás paraméterei alapján
random_value = normal_dist.gen_random()
print(f"Véletlenszerű érték a normál eloszlásból: {random_value}")


#%%
"""
15., Egészítse ki a NormalDistribution osztályt egy új függvénnyel, amely loc várható értékű és scale varianciájú normális eloszlás eloszlásfüggvényéből rendel valószínűségi értéket a bemeneti x valós számhoz.
    függvény név: pdf
    bemenet: x
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Normal_distribution
"""
#%%
import random
import math

class NormalDistribution:
    def __init__(self, rand, loc, scale):
        self.random_generator = rand
        self.location = loc
        self.scale = scale

    def gen_random(self):
        return self.random_generator.gauss(self.location, self.scale)

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        coefficient = 1 / (self.scale * math.sqrt(2 * math.pi))
        exponent = -((x - self.location) ** 2) / (2 * (self.scale ** 2))
        return coefficient * math.exp(exponent)

# Példa használat
random_generator = random.Random()
normal_dist = NormalDistribution(random_generator, 0, 1)  # Példa értékekkel inicializálva

# Valószínűségi sűrűségfüggvény meghívása
x = 0.5
probability = normal_dist.pdf(x)
print(f"Az eloszlás valószínűségi értéke {x}-nél: {probability}")

#%%
"""
16., Egészítse ki a NormalDistribution osztályt egy új függvénnyel, amely megvalósítja az eloszlás kumulatív eloszlás függvényét.
    függvény név: cdf
    bemenet: x
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Normal_distribution
"""
#%%
import random
import math
from scipy.special import erf

class NormalDistribution:
    def __init__(self, rand, loc, scale):
        self.random_generator = rand
        self.location = loc
        self.scale = scale

    def gen_random(self):
        return self.random_generator.gauss(self.location, self.scale)

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        coefficient = 1 / (self.scale * math.sqrt(2 * math.pi))
        exponent = -((x - self.location) ** 2) / (2 * (self.scale ** 2))
        return coefficient * math.exp(exponent)

    def cdf(self, x):
        z = (x - self.location) / (self.scale * math.sqrt(2))
        return 0.5 * (1 + erf(z))

# Példa használat
random_generator = random.Random()
normal_dist = NormalDistribution(random_generator, 0, 1)  # Példa értékekkel inicializálva

# Kumulatív eloszlásfüggvény meghívása
x = 0.5
cumulative_probability = normal_dist.cdf(x)
print(f"Az eloszlás kumulatív valószínűsége {x}-ig: {cumulative_probability}")

#%%
"""
17., Egészítse ki a NormalDistribution osztályt egy új függvénnyel, amely implementálja az eloszlás inverz kumulatív eloszlás függvényét
    függvény név: ppf
    bemenet: p
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Normal_distribution
"""
#%%
import random
import math
from scipy.special import erfinv

class NormalDistribution:
    def __init__(self, rand, loc, scale):
        self.random_generator = rand
        self.location = loc
        self.scale = scale

    def gen_random(self):
        return self.random_generator.gauss(self.location, self.scale)

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        coefficient = 1 / (self.scale * math.sqrt(2 * math.pi))
        exponent = -((x - self.location) ** 2) / (2 * (self.scale ** 2))
        return coefficient * math.exp(exponent)

    def cdf(self, x):
        z = (x - self.location) / (self.scale * math.sqrt(2))
        return 0.5 * (1 + math.erf(z))

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")
        z = math.sqrt(2) * erfinv(2 * p - 1)
        return self.location + self.scale * z

# Példa használat
random_generator = random.Random()
normal_dist = NormalDistribution(random_generator, 0, 1)  # Példa értékekkel inicializálva

# Inverz kumulatív eloszlásfüggvény meghívása
p = 0.5
inverse_cdf_value = normal_dist.ppf(p)
print(f"Az inverz kumulatív eloszlás értéke a valószínűséghez {p}: {inverse_cdf_value}")

#%%
"""
18., Egészítse ki a NormalDistribution osztályt egy új függvénnyel, amely az osztály létrehozásánál megadott paraméterek mellett, normális eloszlású véletlen számokat generál minden meghívásnál
    függvény név: gen_random
    bemenet: None
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Normal_distribution

"""
#%%
import random
import math
from scipy.special import erfinv

class NormalDistribution:
    def __init__(self, rand, loc, scale):
        self.random_generator = rand
        self.location = loc
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        coefficient = 1 / (self.scale * math.sqrt(2 * math.pi))
        exponent = -((x - self.location) ** 2) / (2 * (self.scale ** 2))
        return coefficient * math.exp(exponent)

    def cdf(self, x):
        z = (x - self.location) / (self.scale * math.sqrt(2))
        return 0.5 * (1 + math.erf(z))

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")
        z = math.sqrt(2) * erfinv(2 * p - 1)
        return self.location + self.scale * z

    def gen_random(self):
        return self.random_generator.gauss(self.location, self.scale)

    def mean(self):
        if self.location is None:
            raise Exception("Moment undefined")
        return self.location

# Példa használat
random_generator = random.Random()
normal_dist = NormalDistribution(random_generator, 0, 1)  # Példa értékekkel inicializálva

# Normális eloszlású véletlen szám generálása
random_value = normal_dist.gen_random()
print(f"Generált normális eloszlású véletlen szám: {random_value}")


#%%
"""
19., Egészítse ki a NormalDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény átlagát.  Ha az eloszlásnak nincsen ilyen értéke, akkor return helyett hívja meg a raise Exception("Moment undefined") parancsot.
    függvény név: mean
    bemenet: None
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Normal_distribution
"""
#%%
import random
import math
from scipy.special import erfinv

class NormalDistribution:
    def __init__(self, rand, loc, scale):
        self.random_generator = rand
        self.location = loc
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        coefficient = 1 / (self.scale * math.sqrt(2 * math.pi))
        exponent = -((x - self.location) ** 2) / (2 * (self.scale ** 2))
        return coefficient * math.exp(exponent)

    def cdf(self, x):
        z = (x - self.location) / (self.scale * math.sqrt(2))
        return 0.5 * (1 + math.erf(z))

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")
        z = math.sqrt(2) * erfinv(2 * p - 1)
        return self.location + self.scale * z

    def gen_random(self):
        return self.random_generator.gauss(self.location, self.scale)

    def mean(self):
        if self.location is None:
            raise Exception("Moment undefined")
        return self.location

# Példa használat
random_generator = random.Random()
normal_dist = NormalDistribution(random_generator, 0, 1)  # Példa értékekkel inicializálva

# Eloszlás átlagának lekérdezése
mean_value = normal_dist.mean()
print(f"Az eloszlás átlaga (várható érték): {mean_value}")

#%%
"""
20., Egészítse ki a NormalDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény mediánját
    függvény név: median
    bemenet: None
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Normal_distribution
"""
#%%
import random
import math
from scipy.special import erfinv

class NormalDistribution:
    def __init__(self, rand, loc, scale):
        self.random_generator = rand
        self.location = loc
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        coefficient = 1 / (self.scale * math.sqrt(2 * math.pi))
        exponent = -((x - self.location) ** 2) / (2 * (self.scale ** 2))
        return coefficient * math.exp(exponent)

    def cdf(self, x):
        z = (x - self.location) / (self.scale * math.sqrt(2))
        return 0.5 * (1 + math.erf(z))

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")
        z = math.sqrt(2) * erfinv(2 * p - 1)
        return self.location + self.scale * z

    def gen_random(self):
        return self.random_generator.gauss(self.location, self.scale)

    def mean(self):
        if self.location is None:
            raise Exception("Moment undefined")
        return self.location

    def median(self):
        median_value = self.ppf(0.5)
        return median_value

# Példa használat
random_generator = random.Random()
normal_dist = NormalDistribution(random_generator, 0, 1)  # Példa értékekkel inicializálva

# Eloszlás mediánjának lekérdezése
median_value = normal_dist.median()
print(f"Az eloszlás mediánja: {median_value}")

#%%
"""
21., Egészítse ki a NormalDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény varianciáját.  Ha az eloszlásnak nincsen ilyen értéke, akkor return helyett hívja meg a raise Exception("Moment undefined") parancsot.
    függvény név: variance
    bemenet: None
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Normal_distribution
"""
#%%
import random
import math
from scipy.special import erfinv

class NormalDistribution:
    def __init__(self, rand, loc, scale):
        self.random_generator = rand
        self.location = loc
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        coefficient = 1 / (self.scale * math.sqrt(2 * math.pi))
        exponent = -((x - self.location) ** 2) / (2 * (self.scale ** 2))
        return coefficient * math.exp(exponent)

    def cdf(self, x):
        z = (x - self.location) / (self.scale * math.sqrt(2))
        return 0.5 * (1 + math.erf(z))

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")
        z = math.sqrt(2) * erfinv(2 * p - 1)
        return self.location + self.scale * z

    def gen_random(self):
        return self.random_generator.gauss(self.location, self.scale)

    def mean(self):
        if self.location is None:
            raise Exception("Moment undefined")
        return self.location

    def median(self):
        median_value = self.ppf(0.5)
        return median_value

    def variance(self):
        if self.scale is None:
            raise Exception("Moment undefined")
        return self.scale ** 2

# Példa használat
random_generator = random.Random()
normal_dist = NormalDistribution(random_generator, 0, 1)  # Példa értékekkel inicializálva

# Eloszlás varianciájának lekérdezése
variance_value = normal_dist.variance()
print(f"Az eloszlás varianciája: {variance_value}")

#%%
"""
22., Egészítse ki a NormalDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény ferdeségét.  Ha az eloszlásnak nincsen ilyen értéke, akkor return helyett hívja meg a raise Exception("Moment undefined") parancsot.
    függvény név: skewness
    bemenet: None
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Normal_distribution
"""
#%%
import random
import math
from scipy.special import erfinv

class NormalDistribution:
    def __init__(self, rand, loc, scale):
        self.random_generator = rand
        self.location = loc
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        coefficient = 1 / (self.scale * math.sqrt(2 * math.pi))
        exponent = -((x - self.location) ** 2) / (2 * (self.scale ** 2))
        return coefficient * math.exp(exponent)

    def cdf(self, x):
        z = (x - self.location) / (self.scale * math.sqrt(2))
        return 0.5 * (1 + math.erf(z))

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")
        z = math.sqrt(2) * erfinv(2 * p - 1)
        return self.location + self.scale * z

    def gen_random(self):
        return self.random_generator.gauss(self.location, self.scale)

    def mean(self):
        if self.location is None:
            raise Exception("Moment undefined")
        return self.location

    def median(self):
        median_value = self.ppf(0.5)
        return median_value

    def variance(self):
        if self.scale is None:
            raise Exception("Moment undefined")
        return self.scale ** 2

    def skewness(self):
        if self.scale is None:
            raise Exception("Moment undefined")
        return 0.0  # A normális eloszlás ferdesége mindig 0

# Példa használat
random_generator = random.Random()
normal_dist = NormalDistribution(random_generator, 0, 1)  # Példa értékekkel inicializálva

# Eloszlás ferdeségének lekérdezése
skewness_value = normal_dist.skewness()
print(f"Az eloszlás ferdesége: {skewness_value}")

#%%
"""
23., Egészítse ki a NormalDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény többlet csúcsosságát.  Ha az eloszlásnak nincsen ilyen értéke, akkor return helyett hívja meg a raise Exception("Moment undefined") parancsot.
    függvény név: ex_kurtosis
    bemenet: None
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Normal_distribution
"""
#%%
import random
import math
from scipy.special import erfinv

class NormalDistribution:
    def __init__(self, rand, loc, scale):
        self.random_generator = rand
        self.location = loc
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        coefficient = 1 / (self.scale * math.sqrt(2 * math.pi))
        exponent = -((x - self.location) ** 2) / (2 * (self.scale ** 2))
        return coefficient * math.exp(exponent)

    def cdf(self, x):
        z = (x - self.location) / (self.scale * math.sqrt(2))
        return 0.5 * (1 + math.erf(z))

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")
        z = math.sqrt(2) * erfinv(2 * p - 1)
        return self.location + self.scale * z

    def gen_random(self):
        return self.random_generator.gauss(self.location, self.scale)

    def mean(self):
        if self.location is None:
            raise Exception("Moment undefined")
        return self.location

    def median(self):
        median_value = self.ppf(0.5)
        return median_value

    def variance(self):
        if self.scale is None:
            raise Exception("Moment undefined")
        return self.scale ** 2

    def skewness(self):
        if self.scale is None:
            raise Exception("Moment undefined")
        return 0.0  # A normális eloszlás ferdesége mindig 0

    def ex_kurtosis(self):
        if self.scale is None:
            raise Exception("Moment undefined")
        return 0.0  # A normális eloszlás többlet csúcsossága mindig 0

    def mvsk(self):
        if self.scale is None:
            raise Exception("Moment undefined")
        return [0.0, 0.0, 0.0, 0.0]

# Példa használat
random_generator = random.Random()
normal_dist = NormalDistribution(random_generator, 0, 1)  # Példa értékekkel inicializálva

# Eloszlás többlet csúcsosságának lekérdezése
ex_kurtosis_value = normal_dist.ex_kurtosis()
print(f"Az eloszlás többlet csúcsossága: {ex_kurtosis_value}")

#%%
"""
24., Egészítse ki a NormalDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény első 3 cetrális momentumát és a többlet csúcsosságot.  Ha az eloszlásnak nincsenek ilyen értékei, akkor return helyett hívja meg a raise Exception("Moment undefined") parancsot.
    függvény név: mvsk
    bemenet: None
    kimeneti típus: List
    link: https://en.wikipedia.org/wiki/Normal_distribution
"""
#%%
import random
import math
from scipy.special import erfinv

class NormalDistribution:
    def __init__(self, rand, loc, scale):
        self.random_generator = rand
        self.location = loc
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        coefficient = 1 / (self.scale * math.sqrt(2 * math.pi))
        exponent = -((x - self.location) ** 2) / (2 * (self.scale ** 2))
        return coefficient * math.exp(exponent)

    def cdf(self, x):
        z = (x - self.location) / (self.scale * math.sqrt(2))
        return 0.5 * (1 + math.erf(z))

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")
        z = math.sqrt(2) * erfinv(2 * p - 1)
        return self.location + self.scale * z

    def gen_random(self):
        return self.random_generator.gauss(self.location, self.scale)

    def mean(self):
        if self.location is None:
            raise Exception("Moment undefined")
        return self.location

    def median(self):
        median_value = self.ppf(0.5)
        return median_value

    def variance(self):
        if self.scale is None:
            raise Exception("Moment undefined")
        return self.scale ** 2

    def skewness(self):
        if self.scale is None:
            raise Exception("Moment undefined")
        return 0.0  # A normális eloszlás ferdesége mindig 0

    def ex_kurtosis(self):
        if self.scale is None:
            raise Exception("Moment undefined")
        return 0.0  # A normális eloszlás többlet csúcsossága mindig 0

    def mvsk(self):
        if self.scale is None:
            raise Exception("Moment undefined")
        return [0.0, 0.0, 0.0, 0.0]

# Példa használat
random_generator = random.Random()
normal_dist = NormalDistribution(random_generator, 0, 1)  # Példa értékekkel inicializálva

# Eloszlás első három cetrális momentumának és többlet csúcsosságának lekérdezése
moment_values = normal_dist.mvsk()
print(f"Eloszlás momentumai és többlet csúcsossága: {moment_values}")

#%%
"""
25., Hozzan létre egy új osztályt aminek a neve CauchyDistribution. Definiáljon benne a __init__ nevű függvényt, amelynek bemenetként kap egy véletlenszám generátort, az x0 (location) és gamma (scale) értékeket, amelyeket adattagokba ment le.
    Osztály név: CauchyDistribution
    függvény név: __init__
    bemenet: self, rand, loc, scale
"""
#%%
import random

class CauchyDistribution:
    def __init__(self, rand, loc, scale):
        self.random_generator = rand
        self.location = loc
        self.scale = scale

# Példa használat
random_generator = random.Random()
cauchy_dist = CauchyDistribution(random_generator, 0, 1)  # Példa értékekkel inicializálva

# Az osztály adattagjainak lekérdezése
print(f"Location (x0): {cauchy_dist.location}")
print(f"Scale (gamma): {cauchy_dist.scale}")

#%%
"""
26., Egészítse ki a CauchyDistribution osztályt egy új függvénnyel, amely a bemeneti x értékhez hozzárendeli annak valószínűségi értékét
    függvény név: pdf
    bemenet: x
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Cauchy_distribution
"""
#%%
import random
import math

class CauchyDistribution:
    def __init__(self, rand, loc, scale):
        self.random_generator = rand
        self.location = loc
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        numerator = self.scale
        denominator = math.pi * ((x - self.location) ** 2 + self.scale ** 2)
        return numerator / denominator

# Példa használat
random_generator = random.Random()
cauchy_dist = CauchyDistribution(random_generator, 0, 1)  # Példa értékekkel inicializálva

# Valószínűségi érték kiszámítása adott x értékhez
x_value = 2.0
probability = cauchy_dist.pdf(x_value)
print(f"Az x = {x_value} érték valószínűsége: {probability}")

#%%
"""
27., Egészítse ki a CauchyDistribution osztályt egy új függvénnyel, amely megvalósítja az eloszlás kumulatív eloszlás függvényét.
    függvény név: cdf
    bemenet: x
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Cauchy_distribution
"""
#%%
import random
import math

class CauchyDistribution:
    def __init__(self, rand, loc, scale):
        self.random_generator = rand
        self.location = loc
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        numerator = self.scale
        denominator = math.pi * ((x - self.location) ** 2 + self.scale ** 2)
        return numerator / denominator

    def cdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        z = (x - self.location) / self.scale
        return 0.5 + (1 / math.pi) * math.atan(z)

# Példa használat
random_generator = random.Random()
cauchy_dist = CauchyDistribution(random_generator, 0, 1)  # Példa értékekkel inicializálva

# Kumulatív eloszlás függvény értékének kiszámítása adott x értékhez
x_value = 2.0
cumulative_probability = cauchy_dist.cdf(x_value)
print(f"Az x = {x_value} értékhez tartozó kumulatív eloszlás értéke: {cumulative_probability}")

#%%
"""
28., Egészítse ki a CauchyDistribution osztályt egy új függvénnyel, amely implementálja az eloszlás inverz kumulatív eloszlás függvényét
    függvény név: ppf
    bemenet: p
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Cauchy_distribution
"""
#%%
import random
import math

class CauchyDistribution:
    def __init__(self, rand, loc, scale):
        self.random_generator = rand
        self.location = loc
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        numerator = self.scale
        denominator = math.pi * ((x - self.location) ** 2 + self.scale ** 2)
        return numerator / denominator

    def cdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        z = (x - self.location) / self.scale
        return 0.5 + (1 / math.pi) * math.atan(z)

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")
        return self.location + self.scale * math.tan(math.pi * (p - 0.5))

# Példa használat
random_generator = random.Random()
cauchy_dist = CauchyDistribution(random_generator, 0, 1)  # Példa értékekkel inicializálva

# Inverz kumulatív eloszlás függvény értékének kiszámítása adott p valószínűségi értékhez
probability = 0.25
x_value = cauchy_dist.ppf(probability)
print(f"Az {probability} valószínűséghez tartozó inverz kumulatív eloszlás értéke: {x_value}")

#%%
 """
29., Egészítse ki a CauchyDistribution osztályt egy új függvénnyel, amely az osztály létrehozásánál megadott paraméterek mellett, cauchy eloszlású véletlen számokat generál minden meghívásnál
    függvény név: gen_random
    bemenet: None
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Cauchy_distribution
"""
#%%
import random
import math

class CauchyDistribution:
    def __init__(self, rand, loc, scale):
        self.random_generator = rand
        self.location = loc
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        numerator = self.scale
        denominator = math.pi * ((x - self.location) ** 2 + self.scale ** 2)
        return numerator / denominator

    def cdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        z = (x - self.location) / self.scale
        return 0.5 + (1 / math.pi) * math.atan(z)

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")
        return self.location + self.scale * math.tan(math.pi * (p - 0.5))

    def gen_random(self):
        u1 = self.random_generator.random()
        u2 = self.random_generator.random()
        return self.location + self.scale * math.tan(math.pi * (u1 - 0.5))

# Példa használat
random_generator = random.Random()
cauchy_dist = CauchyDistribution(random_generator, 0, 1)  # Példa értékekkel inicializálva

# Cauchy eloszlású véletlen szám generálása
random_value = cauchy_dist.gen_random()
print(f"A generált Cauchy eloszlású véletlen szám: {random_value}")

#%%
"""
30., Egészítse ki a CauchyDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény átlagát. Ha az eloszlásnak nincsen ilyen értéke, akkor return helyett hívja meg a raise Exception("Moment undefined") parancsot.
    függvény név: mean
    bemenet: None
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Cauchy_distribution
"""
#%%
import random
import math

class CauchyDistribution:
    def __init__(self, rand, loc, scale):
        self.random_generator = rand
        self.location = loc
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        numerator = self.scale
        denominator = math.pi * ((x - self.location) ** 2 + self.scale ** 2)
        return numerator / denominator

    def cdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        z = (x - self.location) / self.scale
        return 0.5 + (1 / math.pi) * math.atan(z)

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")
        return self.location + self.scale * math.tan(math.pi * (p - 0.5))

    def gen_random(self):
        u1 = self.random_generator.random()
        u2 = self.random_generator.random()
        return self.location + self.scale * math.tan(math.pi * (u1 - 0.5))

    def mean(self):
        raise Exception("Moment undefined")

# Példa használat
random_generator = random.Random()
cauchy_dist = CauchyDistribution(random_generator, 0, 1)  # Példa értékekkel inicializálva

# Átlag lekérdezése (moment undefined)
try:
    mean = cauchy_dist.mean()
except Exception as e:
    print(str(e))

#%%
"""
31., Egészítse ki a CauchyDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény mediánját.
    függvény név: median
    bemenet: None
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Cauchy_distribution
"""
#%%
import random
import math

class CauchyDistribution:
    def __init__(self, rand, loc, scale):
        self.random_generator = rand
        self.location = loc
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        numerator = self.scale
        denominator = math.pi * ((x - self.location) ** 2 + self.scale ** 2)
        return numerator / denominator

    def cdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        z = (x - self.location) / self.scale
        return 0.5 + (1 / math.pi) * math.atan(z)

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")
        return self.location + self.scale * math.tan(math.pi * (p - 0.5))

    def gen_random(self):
        u1 = self.random_generator.random()
        u2 = self.random_generator.random()
        return self.location + self.scale * math.tan(math.pi * (u1 - 0.5))

    def mean(self):
        raise Exception("Moment undefined")

    def median(self):
        return self.location  # A medián mindig a hely (location) értéke

# Példa használat
random_generator = random.Random()
cauchy_dist = CauchyDistribution(random_generator, 0, 1)  # Példa értékekkel inicializálva

# Medián lekérdezése
median = cauchy_dist.median()
print(f"A Cauchy eloszlás mediánja: {median}")

#%%
"""
32., Egészítse ki a CauchyDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény varianciáját.  Ha az eloszlásnak nincsen ilyen értéke, akkor return helyett hívja meg a raise Exception("Moment undefined") parancsot.
    függvény név: variance
    bemenet: None
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Cauchy_distribution
"""
#%%
import random
import math

class CauchyDistribution:
    def __init__(self, rand, loc, scale):
        self.random_generator = rand
        self.location = loc
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        numerator = self.scale
        denominator = math.pi * ((x - self.location) ** 2 + self.scale ** 2)
        return numerator / denominator

    def cdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        z = (x - self.location) / self.scale
        return 0.5 + (1 / math.pi) * math.atan(z)

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")
        return self.location + self.scale * math.tan(math.pi * (p - 0.5))

    def gen_random(self):
        u1 = self.random_generator.random()
        u2 = self.random_generator.random()
        return self.location + self.scale * math.tan(math.pi * (u1 - 0.5))

    def mean(self):
        raise Exception("Moment undefined")

    def median(self):
        return self.location

    def variance(self):
        raise Exception("Moment undefined")

# Példa használat
random_generator = random.Random()
cauchy_dist = CauchyDistribution(random_generator, 0, 1)  # Példa értékekkel inicializálva

# Variancia lekérdezése (moment undefined)
try:
    variance = cauchy_dist.variance()
except Exception as e:
    print(str(e))

#%%
"""
33., Egészítse ki a CauchyDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény ferdeségét.  Ha az eloszlásnak nincsen ilyen értéke, akkor return helyett hívja meg a raise Exception("Moment undefined") parancsot.
    függvény név: skewness
    bemenet: None
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Cauchy_distribution
"""
#%%
import random
import math

class CauchyDistribution:
    def __init__(self, rand, loc, scale):
        self.random_generator = rand
        self.location = loc
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        numerator = self.scale
        denominator = math.pi * ((x - self.location) ** 2 + self.scale ** 2)
        return numerator / denominator

    def cdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        z = (x - self.location) / self.scale
        return 0.5 + (1 / math.pi) * math.atan(z)

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")
        return self.location + self.scale * math.tan(math.pi * (p - 0.5))

    def gen_random(self):
        u1 = self.random_generator.random()
        u2 = self.random_generator.random()
        return self.location + self.scale * math.tan(math.pi * (u1 - 0.5))

    def mean(self):
        raise Exception("Moment undefined")

    def median(self):
        return self.location

    def variance(self):
        raise Exception("Moment undefined")

    def skewness(self):
        raise Exception("Moment undefined")

# Példa használat
random_generator = random.Random()
cauchy_dist = CauchyDistribution(random_generator, 0, 1)  # Példa értékekkel inicializálva

# Ferdeség (skewness) lekérdezése (moment undefined)
try:
    skewness = cauchy_dist.skewness()
except Exception as e:
    print(str(e))

#%%
"""
34., Egészítse ki a CauchyDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény többlet csúcsosságát.  Ha az eloszlásnak nincsen ilyen értéke, akkor return helyett hívja meg a raise Exception("Moment undefined") parancsot.
    függvény név: ex_kurtosis
    bemenet: None
    kimeneti típus: float
    link: https://en.wikipedia.org/wiki/Cauchy_distribution
"""
#%%
import random
import math

class CauchyDistribution:
    def __init__(self, rand, loc, scale):
        self.random_generator = rand
        self.location = loc
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        numerator = self.scale
        denominator = math.pi * ((x - self.location) ** 2 + self.scale ** 2)
        return numerator / denominator

    def cdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        z = (x - self.location) / self.scale
        return 0.5 + (1 / math.pi) * math.atan(z)

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")
        return self.location + self.scale * math.tan(math.pi * (p - 0.5))

    def gen_random(self):
        u1 = self.random_generator.random()
        u2 = self.random_generator.random()
        return self.location + self.scale * math.tan(math.pi * (u1 - 0.5))

    def mean(self):
        raise Exception("Moments undefined")

    def median(self):
        return self.location

    def variance(self):
        raise Exception("Moments undefined")

    def skewness(self):
        raise Exception("Moments undefined")

    def ex_kurtosis(self):
        raise Exception("Moments undefined")

# Példa használat
random_generator = random.Random()
cauchy_dist = CauchyDistribution(random_generator, 0, 1)  # Példa értékekkel inicializálva

# Többlet csúcsosság (excess kurtosis) lekérdezése (moments undefined)
try:
    ex_kurtosis = cauchy_dist.ex_kurtosis()
except Exception as e:
    print(str(e))

#%%
"""
35., Egészítse ki a CauchyDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény első 3 cetrális momentumát és a többlet csúcsosságot.  Ha az eloszlásnak nincsenek ilyen értékei, akkor return helyett hívja meg a raise Exception("Moments undefined") parancsot.
    függvény név: mvsk
    bemenet: None
    kimeneti típus: List
    link: https://en.wikipedia.org/wiki/Cauchy_distribution
"""
#%%
import random
import math

class CauchyDistribution:
    def __init__(self, rand, loc, scale):
        self.random_generator = rand
        self.location = loc
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        numerator = self.scale
        denominator = math.pi * ((x - self.location) ** 2 + self.scale ** 2)
        return numerator / denominator

    def cdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórás (scale) értéke pozitív számnak kell lennie.")
        z = (x - self.location) / self.scale
        return 0.5 + (1 / math.pi) * math.atan(z)

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")
        return self.location + self.scale * math.tan(math.pi * (p - 0.5))

    def gen_random(self):
        u1 = self.random_generator.random()
        u2 = self.random_generator.random()
        return self.location + self.scale * math.tan(math.pi * (u1 - 0.5))

    def mean(self):
        raise Exception("Moments undefined")

    def median(self):
        return self.location

    def variance(self):
        raise Exception("Moments undefined")

    def skewness(self):
        raise Exception("Moments undefined")

    def ex_kurtosis(self):
        raise Exception("Moments undefined")

    def mvsk(self):
        raise Exception("Moments undefined")

# Példa használat
random_generator = random.Random()
cauchy_dist = CauchyDistribution(random_generator, 0, 1)  # Példa értékekkel inicializálva

# Az eloszlás függvény első 3 cetrális momentumai és többlet csúcsossága lekérdezése (moments undefined)
try:
    mvsk = cauchy_dist.mvsk()
except Exception as e:
    print(str(e))
