# 1., Írjon egy függvényt ami vissza adja a bemeneti lista páros elemeit
# függvény név: evens_from_list
# bemeneti paraméterek: input_list
# kimeneti típus: List
def evens_from_list(input_list):
    even_numbers = [x for x in input_list if x % 2 == 0]
    return even_numbers
input_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
even_numbers = evens_from_list(input_list)
print(even_numbers)

# 2., Írjon egy függvényt ami megvizsgálja, hogy a listában minden elem páratlan-e
# függvény név: every_element_is_odd
# bemeneti paraméterek: input_list
# kimeneti típus: bool
def every_element_is_odd(input_list):
    for num in input_list:
        if num % 2 == 0:
            return False
    return True
list1 = [1, 3, 5, 7, 9]
result1 = every_element_is_odd(list1)
print(result1)

list2 = [1, 3, 5, 6, 9]
result2 = every_element_is_odd(list2)
print(result2)

# 3., Írjon egy függvényt ami visszaadja a k. legnagyobb elemet listában
# függvény név: kth_largest_in_list
# bemeneti paraméterek: input_list, kth_largest
# kimeneti típus: int
# 3., Írjon egy függvényt ami visszaadja a k. legnagyobb elemet listában
# függvény név: kth_largest_in_list
# bemeneti paraméterek: input_list, kth_largest
# kimeneti típus: int
# %%
import heapq

def kth_largest_in_list(input_list, kth_largest):
    if kth_largest <= 0:
        raise ValueError("kth_largest értéke csak pozitív lehet.")

    heap_size = 0  # A min heap mérete

    min_heap = []  # Min heap létrehozása

    for num in input_list:
        if heap_size < kth_largest:
            heapq.heappush(min_heap, num)
            heap_size += 1
        elif num > min_heap[0]:
            heapq.heappop(min_heap)
            heapq.heappush(min_heap, num)

    if heap_size < kth_largest:
        raise ValueError("A k. legnagyobb elem nem létezik a listában.")

    return min_heap[0]

input_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
k = 3
result = kth_largest_in_list(input_list, k)
print(result)

# 4., Írjon egy függvényt ami kiszámítja a lista elemek gördülő átlagát
# függvény név: cumavg_list
# bemeneti paraméterek: input_list
# kimeneti típus: List
def cumavg_list(input_list):
    if not input_list:
        return []

    cumsum = [input_list[0]]
    for i in range(1, len(input_list)):
        cumsum.append(cumsum[i - 1] + input_list[i])

    cumavg = [cumsum[i] / (i + 1) for i in range(len(cumsum))]

    return cumavg
input_list = [1, 2, 3, 4, 5]
result = cumavg_list(input_list)
print(result)

# 5., Írjon egy függvényt ami kiszámítja 2 lista elemenként vett szorzatát
# függvény név: element_wise_multiplication
# bemeneti paraméterek: input_list1, input_list2
# kimeneti típus: List
def element_wise_multiplication(input_list1, input_list2):
    if len(input_list1) != len(input_list2):
        raise ValueError("A bemeneti listák hossza nem egyezik meg.")

    result = [a * b for a, b in zip(input_list1, input_list2)]
    return result
list1 = [1, 2, 3, 4, 5]
list2 = [2, 3, 4, 5, 6]
result = element_wise_multiplication(list1, list2)
print(result)

# 6., Írjon egy függvényt amely összekapcsol n listát 1 listába
# függvény név: merge_lists
# bemeneti paraméterek: *lists
# kimeneti típus: List
def merge_lists(*lists):
    merged_list = []
    for l in lists:
        merged_list.extend(l)
    return merged_list
list1 = [1, 2, 3]
list2 = [4, 5, 6]
list3 = [7, 8, 9]

result = merge_lists(list1, list2, list3)
print(result)

# 7., Írjon egy függvényt amely visszaadja a lista páratlan elemeinek a négyzetét
# függvény név: squared_odds
# bemeneti paraméterek: input_list
# kimeneti típus: List
def squared_odds(input_list):
    squared_odds_list = [x ** 2 for x in input_list if x % 2 != 0]
    return squared_odds_list
input_list = [1, 2, 3, 4, 5, 6]
result = squared_odds(input_list)
print(result)

# 8., Írjon egy függvényt amely a fordítottan sorba rendezi a kulcs-érték párokat a kulcs értéke szerint egy dictionary-ben
# függvény név: reverse_sort_by_key
# bemeneti paraméterek: input_dict
# kimeneti típus: Dict
# 8., Írjon egy függvényt amely a fordítottan sorba rendezi a kulcs-érték párokat a kulcs értéke szerint egy dictionary-ben
# függvény név: reverse_sort_by_key
# bemeneti paraméterek: input_dict
# kimeneti típus: Dict
#%%
def reverse_sort_by_key(input_dict):
    sorted_dict = dict(sorted(input_dict.items(), key=lambda item: item[0], reverse=True))
    return sorted_dict
input_dict = {'c': 3, 'a': 1, 'b': 2}
result = reverse_sort_by_key(input_dict)
print(result)

# 9., Írjon egy függvényt amely a bemeneti, pozitív egész számokat tartalmazó listát kiválogatja 2-vel, 5-el, 2-vel és 5-el, és egyikkel sem osztható számokat, és visszaad egy olyan dictionary-t, amelyben a kulcsok a 'by_two', 'by_five', 'by_two_and_five', és a 'by_none', az értékek, pedig a listák. (2 pont)
# függvény név: sort_list_by_divisibility
# bemeneti paraméterek: input_list
# kimeneti típus: Dict
def sort_list_by_divisibility(input_list):
    by_two = [x for x in input_list if x % 2 == 0 and x % 5 != 0]
    by_five = [x for x in input_list if x % 2 != 0 and x % 5 == 0]
    by_two_and_five = [x for x in input_list if x % 2 == 0 and x % 5 == 0]
    by_none = [x for x in input_list if x % 2 != 0 and x % 5 != 0]

    result_dict = {
        'by_two': by_two,
        'by_five': by_five,
        'by_two_and_five': by_two_and_five,
        'by_none': by_none
    }

    return result_dict
input_list = [1, 2, 3, 4, 5, 6, 10, 15]
result = sort_list_by_divisibility(input_list)
print(result)