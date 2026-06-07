# BATCH GRADIENT DESCENT FROM SCRATCH


# importig imp libraries
import numpy as np
rng = np.random.default_rng()

# Generating and separating training features and explicit column labels
data = rng.uniform(high=1, low=0, size=(5, 6))
print(data)
X_train = data[:, 0:5]
print(X_train)
label = data[:, 5:6]
print(label)
print(label.shape)

# Writing loss function
def loss_val(pred, label):
    # for simplicity in the code i will be using the MSE , but we can change here for other loss functions
    loss = 0.5 * ((label) - (pred)) ** 2
    return loss

# writing gradient function
# next thing that we need diffrential of the loss with the parameters , as it will depend on the specific function, but here we will be writng it for the perceptron
def grad_wrt(input, label, prediction):
    return input.T @ (prediction - label)

# Initialising W and b
def initialize_W(size=(5, 1), high=1, low=0):
    W = rng.uniform(high=high, low=low, size=size)
    print("Weights are following:\n", W, "\n")
    b = rng.random()
    print("Iniitial bias:", b)
    return W, b

W, b = initialize_W(size=(5, 1))

# Perceptron
# wrting perceptron as a function for ease of code
def pred(X, W, b):
    output = X @ W + b
    return output

# Hyperparameters
lr = 0.01

#  BGD (Batch Gradient Descent) Training Loop
# lets write the BGD
def BGD_BP(train_X, epochs, W, b, label):
    N = train_X.shape[0]
    for epoch in range(epochs):
        prediction = pred(X_train, W, b)
        loss = loss_val(label, prediction)
        W -= (1 / N) * lr * grad_wrt(X_train, label, prediction)  # (input,label,prediction)
        b -= (1 / N) * lr * np.sum(prediction - label)  # as it is simple perceptron , grad is -1
        print("updated weights\n", W)
        print("updated bias : ", b)


# Execution Call
BGD_BP(X_train, 5, W, b, label)