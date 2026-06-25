# I will writing all the loss function in this file and may use later
# initially there will be few loss functions , but i will add them i will use them

import numpy as np

# Mean squared error:
def MSE(pred,label):
    '''
    i)   We are squaring to make sure that error are added not get cancelled.
    ii)  loss fucntions will be always differentiable.
    iii) Will have gauranteed local minima.
    iv)  Not robust to outlier.
    v)   Last neuron activation function should be linear,for better stablity of the model.
    
    '''
    loss= (label-pred)**2
    return loss

def MAE(pred,label):
    '''
    i)  Absolute so that error dont get cancelled.
    ii) Not always differentiable, we have to take sub-gradients.
    iii)Robust to outlier beacuse linear in nature wrt error.
    iv) Error units will be conserved

    '''
    loss=abs(pred-label)

    return loss

def HubberLoss(pred,label,delta):
    '''
    i) Mixture of MSE & MAE , for error less than delta , it will be MSE
       but for error more than delta it will be MAE , it give robustness to outlier
       with the benifit of MSE.   
        
    ii) delta is hyperparameter
     
    '''
    
    error=abs(pred-label)
    if error<delta:
        loss=((error)**2)/2
    else:
        loss=delta*error-(delta**2)/2

    return loss

def BCE(pred,label):
    '''
    i)   For binary classification
    ii)  output activation should be sigmoid
    iii) diffrentiable
    iv)  multiple local minima
    
    '''
    pred= np.array(pred)
    label= np.array(label)
    pred = np.clip(pred, 1e-15, 1 - 1e-15)
    loss= -label*np.log(pred)-(1-label)*np.log(1-pred)

    return loss

def CCE(pred,label):
    '''
    i)   For multi class classificiation.
    ii)  Activation function should be Softmax.
    iii) One hot encoding.
    
    '''
    pred= np.array(pred)
    pred = np.clip(pred, 1e-15, 1 - 1e-15)
    label= np.array(label)

    loss=-np.dot(label,np.log(pred))

    return loss

def SCE(pred,label_index):
    '''
    it is similar to CCE, but used when we have done integer assignment , not One hot encoding.
    
    '''
    pred= np.array(pred)
    pred = np.clip(pred, 1e-15, 1 - 1e-15)

    loss=-np.log(pred[label_index])

    return loss