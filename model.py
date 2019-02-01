import numpy as np

# Creating network for the equation z = ax + by

dataset_size = 20

dataset = np.zeros(shape=(dataset_size,dataset_size,1))
weights = np.zeros(shape=(1,2))
learning_rate = 0.001

# Generate dataset
for x in range(0, dataset_size):
    a = 4 # z = ax + by
    b = 2 # z = ax + by
        
    for y in range(0, dataset_size):
        z = (a*x + b*y)
        dataset[x,y] = z
        print('X : {} | Y : {} | Z : {}'.format(x,y,z))

def forward(input, weights):
    np_input = np.array(input)
    result = np.dot(np_input, weights[0])
    return result

def backpropogate(input, result, target, weights):
    #print('input:{} | result:{} | target:{} | weights:{}'.format(input, result, target, weights))
    gradient = result - target
    #print('Gradient : {}'.format(gradient))
    update_value = gradient * input
    #print('Update Value : {}'.format(update_value))
    return update_value

weights[0,0] = 9
weights[0,1] = 7

print('Weights : {}'.format(weights))

for x in range(0, dataset_size):
    for y in range(0, dataset_size):
        print('---------------------------------------------------------------------------------------------------------')
        input = [x,y]
        target = dataset[x,y]
        result = forward(input, weights)
        error = ((result - target)**2) / 2
        weight_delta = backpropogate(input, result, target, weights)
        #print('Delta : {}'.format(weight_delta))
        #print('Weights : {}'.format(weights))
        weights = weights - (learning_rate * weight_delta)
        w1 = weights[0,0]
        w2 = weights[0,1]
        print("Step {:2d} \t| x: {:2d} \t| y: {:2d} \t| target: {:10.4f} \t| result: {:10.4f} \t| E: {:10.4f} \t| W1: {:10.4f} \t| W2: {:10.4f}"
        .format( (x+1)*(y+1) , x,       y,  float(target),      float(result),    float(error),    float(w1),      float(w2),))