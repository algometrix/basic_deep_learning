import numpy as np

# Creating network for the equation z = ax + by

dataset_size = 20

dataset = np.zeros(shape=(dataset_size,dataset_size,1))
weights = np.zeros(shape=(1,2))
learning_rate = 0.001

# Generate dataset
for x in range(0, dataset_size):
    for y in range(0, dataset_size):
        a = 2 # z = ax + by
        b = 3 # z = ax + by
        z = a*x + b*y
        dataset[x,y] = z
        #print('X : {} | Y : {} | Z : {}'.format(x,y,z))

def forward(input, weights):
    np_input = np.array(input)
    result = np.matmul(np_input, weights.T)
    return result

def backpropogate(input, result, target, weights):
    #print('input:{} | result:{} | target:{} | weights:{}'.format(input, result, target, weights))
    gradient = result - target
    #print('Gradient : {}'.format(gradient))
    update_value = gradient * input
    #print('Update Value : {}'.format(update_value))
    return update_value

#weights[0,0] = 1
#weights[0,1] = 1

print('Weights : {}'.format(weights))

for x in range(0, dataset_size):
    for y in range(0, dataset_size):
        print('---------------------------------------------------------------------------------------------------------')
        input = [x,y]
        target = dataset[x,y]
        result = forward(input, weights)
        error = ((result - target)**2) / 2
        weight_delta = backpropogate(input, result, target, weights)
        weights = weights - (learning_rate * weight_delta)
        w1 = weights[0,0]
        w2 = weights[0,1]
        print("x: {} \t| y: {} \t| target: {} \t| result: {} \t| E: {} \t| W1: {} \t| W2: {}"
        .format(x,       y,  target,      result,    error,    w1,      w2,))