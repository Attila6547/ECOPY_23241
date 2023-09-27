"""
0. Importáld a random modult
"""
#%%
import random

#%%
"""
 1.,  1., Hívd meg a random szám generátort, és generálj egy 0 és 1 közötti véletlen számot
"""
#%%
import random

veletlen_szam = random.random()
print(veletlen_szam)
#%%
"""
2., Generálj 1 és 100 közötti egész véletlen számokat
"""
#%%
import random

veletlen_egesz = random.randint(1, 100)
print(veletlen_egesz)

#%%
"""
3. Generálj 1 és 100 közötti véletlen egész számot, olyan módon, hogy a random seed értéke 42 legyen
"""
#%%
import random

random.seed(42)  # Beállítjuk a kezdeti magot 42-re
veletlen_egesz = random.randint(1, 100)
print(veletlen_egesz)

"""
4., Készíts egy függvényt, ami visszaad egy véletlenül kiválasztott elemet egy egész számokat tartalmazó listából.
    függvény neve: random_from_list
    bemenet: input_list
    kimenetí típus: int
"""
#%%
import random

def random_from_list(input_list):
    if len(input_list) == 0:
        raise ValueError("Az input_list üres lista, nincs mit kiválasztani.")

    random_element = random.choice(input_list)
    return random_element
# Tesztelés
szamok = list(range(1, 11))  # 1-től 10-ig terjedő számok listája
veletlen_szam = random_from_list(szamok)
print(veletlen_szam)
#%%
"""
5., Készíts egy függvényt, ami visszaad egy n darab véletlen számot tartalmazó listát egy egész számokat tartalmazó bemenenti listából.
    függvény neve: random_sublist_from_list
    bemenet: input_list, number_of_elements
    kimenetí típus: List
"""
#%%
import random

def random_sublist_from_list(input_list, number_of_elements):
    if not input_list:
        raise ValueError("Az input_list üres lista, nincs mit kiválasztani.")
    if number_of_elements <= 0:
        raise ValueError("A number_of_elements értéke pozitív egész szám kell legyen.")
    if number_of_elements > len(input_list):
        raise ValueError("number_of_elements nem lehet nagyobb, mint az input_list hossza.")

    random_sublist = random.sample(input_list, number_of_elements)
    return random_sublist

# Bemeneti lista: 1-től 100-ig véletlenszerűen kiválasztott 10 szám
input_list = random.sample(range(1, 101), 10)
number_of_elements = 5  # Példa: 5 véletlen számot szeretnénk kiválasztani
veletlen_lista = random_sublist_from_list(input_list, number_of_elements)
print(veletlen_lista)

#%%
import random

def random_sublist_from_list(input_list, number_of_elements):
    if not input_list:
        raise ValueError("Az input_list üres lista, nincs mit kiválasztani.")
    if number_of_elements <= 0:
        raise ValueError("A number_of_elements értéke pozitív egész szám kell legyen.")

    random_sublist = random.sample(input_list, number_of_elements)
    return random_sublist

my_list = list(range(1, 101))  # Az 1-től 100-ig terjedő számok listája
quantity = 3
print(random_sublist_from_list(my_list, quantity))

#%%
"""
6., Készíts egy függvényt, ami visszaad egy véletlenül kiválasztott elemet egy string-ből.
    függvény neve: random_from_string
    bemenet: input_string
    kimeneti típus: string
"""
#%%
import random

def random_from_string(input_string):
    if not input_string:
        raise ValueError("Az input_string üres, nincs mit kiválasztani.")

    random_char = random.choice(input_string)
    return random_char
input_string = "Hello, World!"
random_character = random_from_string(input_string)
print(random_character)

#%%

#%%
'''
7., Készíts egy függvényt, amely visszaad egy 100 elemű, 0 és 1 közötti véletlen számokat tartalmazó listát
    függvény név: hundred_small_random
    bemenet: None
    kimeneti típus: List
'''
#%%
import random

def hundred_small_random():
    random_list = [random.randint(0, 1) for _ in range(100)]
    return random_list
random_numbers = hundred_small_random()
print(random_numbers)

#%%
"""
8., Készíts egy függvényt, amely visszaad egy 100 elemű, 10 és 1000 közötti véletlen számokat tartalmazó listát
    függvény név: hundred_large_random
    bemenet: None
    kimeneti típus: List
"""
#%%
import random

def hundred_large_random():
    random_list = [random.randint(10, 1000) for _ in range(100)]
    return random_list
random_numbers = hundred_large_random()
print(random_numbers)

#%%

#%%
"""
9., Készíts egy függvényt, amely visszaad 5 elemű, 9 és 1000 közötti, 3-al osztható egész számokat tartalmazó listát
    függvény név: five_random_number_div_three
    bemenet: None
    kimeneti típus: List

"""
#%%
import random

def five_random_number_div_three():
    random_list = [random.randint(9, 1000) for _ in range(5)]
    random_list = [x for x in random_list if x % 3 == 0]
    return random_list
random_numbers = five_random_number_div_three()
print(random_numbers)

#%%

#%%
"""
10., Készíts egy függvényt amely a bemeneti lista elemeit véletlenszerűen összekeveri
    függvény név: random_reorder
    bemenet: input_list
    kimeneti típus: List
"""
#%%
import random

def random_reorder(input_list):
    if not input_list:
        return []  # Üres bemeneti lista esetén visszaadunk egy üres listát

    shuffled_list = random.sample(input_list, len(input_list))
    return shuffled_list
input_list = [1, 2, 3, 4, 5]
shuffled_list = random_reorder(input_list)
print(shuffled_list)

#%%
"""
11., Készíts egy függvényt, amely 1 és 5 közötti egyenletes eloszlású valós véletlen számot ad vissza minden meghívás esetén.
    függvény név: uniform_one_to_five
    bemenet: None
    kimeneti típus: float
"""
#%%
import random

def uniform_one_to_five():
    return random.uniform(1, 5)
random_number = uniform_one_to_five()
print(random_number)
