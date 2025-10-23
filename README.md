# Neural_S

## Todo
- Finish rewrite in C

## Resources
- Basic neural network: https://www.geeksforgeeks.org/numpy/implementation-of-neural-network-from-scratch-using-numpy/
- Dataset download link: https://systemds.apache.org/datasets/mnist
- Explanation video/tutorial: https://www.youtube.com/watch?v=w8yWXqWQYmU&t=218s
- Reference notebook: https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras

## Implementation
### Layers
- m training images
- 28x28 images -> 784 pixels (all in range [0, 255])
- 0th layer: input layer, 784 nodes (1 per pixel)
- 1st layer: hidden layer, 20 nodes
- 2nd layer: output layer, 10 nodes (1 per digit)

### Feed forward
```py
A0 = x # Activated 0th layer
Z1 = W1*A0 # Unactivated 1st layer, W1 is weights
A1 = g(Z1) = ReLU(Z1) # Activated 1st layer, ReLU is activation function
Z2 = W2*A1 # Unactivated 2nd layer, W2 is weights
A2 = Softmax(Z2) # Activated 2nd layer, Softmax is activation function
```

### Back propagation
```py
dZ2 = A2-Y
dW2 = (A1).T * dZ2
dZ1 = dZ2 * W2^T * g_prime(Z1) # g' is derivate of activation function
dW1 = (x).T * dZ1
```

### Updating weights
```py
W1 -= alpha * dW1 # alpha is the learning rate
W2 -= alpha * dW2
```

### Training the model
We loop through all the images, and follow the above steps for each image. We repeat this process multiple times to further improve the accuracy.