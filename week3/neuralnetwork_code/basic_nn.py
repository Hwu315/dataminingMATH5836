 


import numpy as np

np.random.seed()

Top = [2,2,1] # NN topology for XOR gate problem
# lets set the weights and biases
#W1 = np.random.uniform(-0.5, 0.5, (Top[0] , Top[1])) 
W1 = np.array([[0.5,0.9 ], [0.4,1.0]]) # overwritten
#B1 = np.random.uniform(-0.5,0.5, (1, Top[1])  ) # bias first layer
B1 = np.array([[0.8,-0.1]]) # overwritten bias of hidden layer

#W2 = np.random.uniform(-0.5, 0.5, (Top[1] , Top[2]))   
W2 = np.array([[-1.2],[1.1]]) # overwritten
#B2 = np.random.uniform(-0.5,0.5, (1,Top[2]))
B2 = np.array([[0.3]]) # overwritten bias of hidden layer

print(W1,  ' W1')
print(B1, ' B1') 
print(W2,  ' W2')
print(B2, ' B2') 


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ForwardPass(X): 
    z1 = X.dot(W1) - B1 
    print(z1, ' z1')  
    hidout = sigmoid(z1) # output of first hidden layer
    print(hidout, ' hidout')   
    z2 = hidout.dot(W2)  - B2 
    print(z2, ' z2')  
    out = sigmoid(z2)  # output second hidden layer
    print(out, ' out')
    return out, hidout

def BackwardPass(input_vec, learn_rate, prediction, hidout, desired):   
    out = prediction 
    out_delta =   (desired - out)*(out*(1-out))  
    print(out_delta, ' is gradient at output')
    #hid_delta = out_delta.dot(W2) * (hidout * (1-hidout))  

    #W2+= hidout.T.dot(out_delta) * learn_rate
    #B2+=  (-1 * learn_rate * out_delta)
    #W1 += (input_vec.T.dot(hid_delta) * learn_rate) 
    #B1+=  (-1 * learn_rate * hid_delta) 

#main


training_inputs = []
training_inputs.append(np.array([1, 1]))
training_inputs.append(np.array([1, 0]))
training_inputs.append(np.array([0, 1]))
training_inputs.append(np.array([0, 0]))

# set the dataset class labels
labels = np.array([0, 1, 1, 0])

print(training_inputs, ' list of training features')

learn_rate = 0.1

max_epochs = 1

for epochs in range(max_epochs):
    for X, y in zip(training_inputs, labels): 
        print(X,y, ' instance') 
        prediction, hidout = ForwardPass(X)
        BackwardPass(X, learn_rate, prediction, hidout, y)


