from f_001_loss_functions import *
import numpy as np


# we will be writing for for two hidden layers(#neuron=3) and one input layer(#neuron=2)  and output layer(#neuron=1)
# and will test this on AND dataset(will write this in generic manner)

# our input will be act in input layer (shaper=(2,1))

# our first hidden layer will have three neuron , as input have dimmension of 2(#neuron) so weight will be (3,2)and weight will be (3,1)
# let initiae this randomly

hid_layer_1_w = np.random.randn(3,2)  # there are various method to initialise the weights but that is not main concern now
hid_layer_1_b = np.random.randn(3,1)

# activation function to introduce non linearity

def acti_fn_1(activation):
    # choosing relu as activation function for first layer
    activation=np.maximum(0,activation)
    return activation

# our second hidden layer have also 3 neurons but it take input of size three, writing weights according to that
hid_layer_2_w = np.random.randn(3,3)  
hid_layer_2_b = np.random.randn(3,1)  

def act_fn_2(activation):
    activation=np.maximum(0,activation)
    return activation

# now it is turn of output layer which have one neuron only
out_layer_w = np.random.randn(1,3)
out_layer_b = np.random.randn(1,1)

def act_fn_out(out):
    out = 1/(1+np.exp(-out))
    return out



# forward network is written clearly now we have do forward pass , save activation calculate loss and then update
#-------------------> BACKPROPOGATION

# our loss function will be binary cross entropy  which is imported from other file at the top with name: BCE

#-------------------> DATASET CREATION (AND gate data, 1 and 0 only)
 
# AND gate has only 4 unique combinations (0,0)(0,1)(1,0)(1,1)
# but we need 40 data points, so we randomly generate these 4 combinations again and again
 
def create_dataset(num_samples=40):
    # keeping X shape as (2,num_samples) since our input layer takes 2 neurons
    # keeping y shape as (1,num_samples) since output layer has only 1 neuron
    X = np.zeros((2,num_samples))
    y = np.zeros((1,num_samples))
 
    for i in range(num_samples):
        a = np.random.randint(0,2)   # random 0 or 1
        b = np.random.randint(0,2)   # random 0 or 1
        X[0,i] = a
        X[1,i] = b
        y[0,i] = 1 if (a==1 and b==1) else 0   # AND gate rule
 
    return X,y
 
X,y = create_dataset(40)
 
 
#-------------------> FORWARD PASS (now caching activations here, so backward pass can use them)
# as we derived earlier, backward pass needs Z1,A1,Z2,A2,A3 and X
# so forward pass will not just return the output, it will return the cache as well
 
def forward_pass(X):
 
    # ------ layer 1 ------
    z1 = np.dot(hid_layer_1_w,X) + hid_layer_1_b     # (3,2)x(2,m) + (3,1) = (3,m)
    a1 = acti_fn_1(z1)
 
    # ------ layer 2 ------
    z2 = np.dot(hid_layer_2_w,a1) + hid_layer_2_b    # (3,3)x(3,m) + (3,1) = (3,m)
    a2 = act_fn_2(z2)
 
    # ------ output layer ------
    z3 = np.dot(out_layer_w,a2) + out_layer_b        # (1,3)x(3,m) + (1,1) = (1,m)
    a3 = act_fn_out(z3)
 
    # caching everything that will be needed in backward pass
    cache = (X,z1,a1,z2,a2,a3)
 
    return a3,cache
 
 
#-------------------> RELU DERIVATIVE
# backward pass needs relu's derivative, and we get it from Z (not from A, this was the important point)
 
def relu_derivative(z):
    return np.where(z>0,1,0)
 
 
#-------------------> BACKWARD PASS
# writing the same equations here that we derived on paper
# dZ3 = A3 - y   (this is the combined simplification of sigmoid + BCE)
# then we propagate this dZ3 backward through W3 to get dZ2, then through W2 to get dZ1
 
def backward_pass(cache,y):
 
    X,z1,a1,z2,a2,a3 = cache
    m = X.shape[1]     # number of examples in the batch, used to average the gradients
 
    # ------ output layer grad ------
    dz3 = a3 - y                              # (1,m)
    dw3 = np.dot(dz3,a2.T)/m                  # (1,3)
    db3 = np.sum(dz3,axis=1,keepdims=True)/m  # (1,1)   axis=1 sums across all examples
 
    # ------ hidden layer 2 grad ------
    da2 = np.dot(out_layer_w.T,dz3)           # (3,m)   transpose of W because going backward means traversing the connections in reverse
    dz2 = da2*relu_derivative(z2)             # (3,m)   Z2 is used here, not A2
    dw2 = np.dot(dz2,a1.T)/m                  # (3,3)
    db2 = np.sum(dz2,axis=1,keepdims=True)/m  # (3,1)
 
    # ------ hidden layer 1 grad ------
    da1 = np.dot(hid_layer_2_w.T,dz2)         # (3,m)
    dz1 = da1*relu_derivative(z1)             # (3,m)
    dw1 = np.dot(dz1,X.T)/m                   # (3,2)
    db1 = np.sum(dz1,axis=1,keepdims=True)/m  # (3,1)
 
    return dw1,db1,dw2,db2,dw3,db3
 
 
#-------------------> TRAINING LOOP (running epochs, each epoch does forward->cache->backward->update)
 
lr = 0.1
epochs = 2000
 
for epoch in range(epochs):
 
    # forward pass, also gives us the cache we discussed
    a3,cache = forward_pass(X)
 
    # calculating loss just for monitoring purpose (not used directly in gradient
    # since we already derived the simplified form dZ3 = A3-y)
    loss = BCE(a3,y)
    avg_loss = np.mean(loss)
 
    # backward pass, passing the same cache we got from forward pass
    dw1,db1,dw2,db2,dw3,db3 = backward_pass(cache,y)
 
    # update rule : new = old - lr*grad     (gradient descent)
    hid_layer_1_w = hid_layer_1_w - lr*dw1
    hid_layer_1_b = hid_layer_1_b - lr*db1
 
    hid_layer_2_w = hid_layer_2_w - lr*dw2
    hid_layer_2_b = hid_layer_2_b - lr*db2
 
    out_layer_w = out_layer_w - lr*dw3
    out_layer_b = out_layer_b - lr*db3
 
    if epoch%200==0:
        print("epoch:",epoch," loss:",avg_loss)
 
 
#-------------------> FINAL CHECK (after training, see if AND gate is predicted correctly or not)
 
print("\nfinal predictions after training:")
final_pred,_ = forward_pass(X)
print(np.round(final_pred[:, :10],3))   # showing predictions for first 10 samples
print("actual:")
print(y[:, :10])