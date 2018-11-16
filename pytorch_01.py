import torch

def activation(x):
    """ Sigmoid activation function 
    """
    return 1/(1+torch.exp(-x))


### Generate some data
torch.manual_seed(7) # Set the random seed so things are predictable

# Features are 3 random normal variables
features = torch.randn((1, 5))
# True weights for our data, random normal variables again
weights = torch.randn_like(features)
# and a true bias term
bias = torch.randn((1, 1))

"""
Exercise: 
Calculate the output of the network with input features, weights, and bias.
Similar to Numpy, PyTorch has a torch.sum() function, as well as a .sum() 
method on tensors, for taking sums. 
Use the function activation defined above as the activation function.
"""
y = activation( torch.sum(features * weights) + bias )
print(y)
y = activation( (features * weights).sum() + bias )
print(y)

"""
Exercise:
Calculate the output of our little network using matrix multiplication.
Usamos:  
*** .reshape(a, b) o .resize_(a, b) o .view(a, b) para hacer traspuesta de la matriz ([1,5] a [5,1])
*** torch.mm(entradas, pesos) ---> Multiplica matrices obligando a las dimensiones correctas.
"""

# El método size() devuelve dimensiones de tensor, en subclase de tupla: torch.Size
# Calculamos las nuevas dimensiones haciendo la reversa de la dimensión anterior.
nuevas_dimensiones = torch.Size(reversed(weights.size()))
y = activation( torch.mm(features, weights.view(nuevas_dimensiones) ).sum() + bias )
print(y)


