import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
#
#  entropia = - SUM( (yi * ln(pi)) + ((1 - yi) * ln(1 - pi)) )
#              i=1 -> m
#
def cross_entropy(Y, P):
    return -1 * np.sum( [ (Y[i] * np.log(P[i])) + ((1 - Y[i]) * np.log(1 - P[i])) for i in range(0, len(P)) ] )


Y = [1,0,1,1]
P = [0.4,0.6,0.1,0.5]

print('Y = {y}'.format(y=Y))
print('P = {p}'.format(p=P))
print('cross_entropy = {c}'.format(c=cross_entropy(Y, P)))