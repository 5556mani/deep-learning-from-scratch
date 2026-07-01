import numpy as np
from f_001_loss_functions import MSE

# We have to write following functions to use later
'''
i)   Initialisation
ii)  Prediction
iii) Use imported Loss function
iv)  Hardcode all derivatives for BP
v)   Update

'''

# Initialisation function:
def initialise(dimension):
    '''
    Dimension=[no of layer,array with no of neurons in ith layer,for i=0 it will be input dimension]
    => All weight will be initialised with 1 and all bias will be initialised with 0
    
    '''

    parameter=[]

    for i in range(0,dimension[0]):

        # Weight matrix as no of row = neurons in current layer
        weight_matrix = np.random.randn(dimension[1][i+1], dimension[1][i]) * np.sqrt(2. / dimension[1][i])  # using He initialisation
        bias          = np.zeros((dimension[1][i+1],1))

        layer_parameter=[weight_matrix,bias]
        parameter.append(layer_parameter)
    '''
    Output Structure:
                    i)  array Parameter will be on len=num of layer given and all element will of size=2
                    ii) in those subarrays first will be Weight matrix and its size will depend on num of neurons in previous layers and in it
                    iii)len of bias will number of neurona in that layer'''

    return parameter

# We need Feed forward function to write clean code of the Prediction function:
def feed_fwd(Weight,bias,input):
    result=Weight@input + bias
    result=np.maximum(0,result) # acting as relu
    return result

# Now we will be writing code for Prediction:
def pred(parameter,input):
    activation=input
    layer_no=len(parameter)
    for i in range(0,layer_no-1):
        activation=feed_fwd(parameter[i][0],parameter[i][1],activation)

    result=parameter[layer_no-1][0]@activation + parameter[layer_no-1][1] # for output layer
    return result

# Writing loss function
def loss(pred,label):
    # LOSS function in more clean words
    return MSE(pred,label)

# Now we have to write the update function