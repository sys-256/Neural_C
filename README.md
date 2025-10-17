# Neural_S

## Todo
- Use different number of nodes for hidden layer

dataset download link: https://systemds.apache.org/datasets/mnist
explanation video/tutorial: https://www.youtube.com/watch?v=w8yWXqWQYmU&t=218s
reference notebook: https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras

28x28 images -> 784 pixels (all between 0 and 256). m training images
0th layer, input layer, 784 nodes (1 per pixel)
1st layer, hidden layer, 10 nodes
2nd layer, output layer, 10 nodes (1 per digit)

A0 = x (784*m) -> activated 0th layer
Z1 = W1*A0+B1 -> unactivated 1st layer. W1 is weights, B1 is biases
A1 = g(Z1) = ReLU(Z1) -> activated 1st layer. ReLU is activation function
Z2 = W2*A1+B2 -> unactivated 2nd layer. W2 is weights. B2 is biases
A2 = Softmax(Z2) -> activated 2nd layer

dZ2 = A2-Y
dW2 = (1/m) * dZ2 * (A1)^T
dB2 = (1/m) * sum(dZ2)

dZ1 = (W2)^T * dZ2 .* g'(Z1) -> g' is derivate of activation function, not sure what .* does, maybe dot product?
dW1 = (1/m) * dZ1 * (x)^T
dB1 = (1/m) * sum(dZ1)

W1 = W1 - alpha * dW1
B1 = B1 - alpha * dB1
W2 = W2 - alpha * dW2
B2 = B2 - alpha * dB2
alpha -> learning rate TODO what's a common value