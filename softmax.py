import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    numeradores = [np.exp(n) for n in L]
    denominador = np.sum(numeradores)
    return [n / denominador for n in numeradores]

print(softmax((2, 1, 0)))

