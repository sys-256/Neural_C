# Neural_C

## About
- A simple neural network written in C, using only libc and libm
- Recognizes handwritten digits from [the MNIST database](https://systemds.apache.org/datasets/mnist).
- Converts a 28x28 image to an input layer with 784 nodes
- Uses [ReLU](https://en.wikipedia.org/wiki/Rectified_linear_unit) to activate the hidden layer of 20 nodes
- Uses [softmax](https://en.wikipedia.org/wiki/Softmax_function) to generate predictions for each digit

## Why
I wanted to learn more about neural networks, linear algebra and programming in C (like pointers, 2D arrays, file reading and memory allocation), so I decided to implement a neural network in C. I started by [learning more about neural networks](#resources), and then [implementing one in Python](src_py/main.py) using only numpy. After that, I gradually rewrote each Python function in C. The result can be found in [src](src). The main function inside [main.c](src/main.c) demonstrates the functionality of each function, including the ones that train the model and test it.

## Accuracy
### Biases
Currently, the model only uses weights and no biases. This was done to make it easier to implement, but it reduces the accuracy. By adding biases, the model would be able to better fit itself to the data.
### Floating point inaccuracies
When comparing the intermediate arruracy and weights from the C implementation to the Python implementation, we see that the C implementation suffers from [floating point accuracy problems](https://en.wikipedia.org/wiki/Floating-point_arithmetic#Accuracy_problems), whereas numpy does not. Resolving this issue would result in more accurate weights, which would improve the accuracy of the model.
### Overfitting
When trained with 20 nodes in the hidden layer on the full training set (60 000 images), the model reaches around 96% accuracy after 25 iterations, and 93% accuracy on the test set. This means that the model is overfitting itself on the training data. We could solve this issue by implementing techniques such as [data augmentation](https://en.wikipedia.org/wiki/Data_augmentation#Data_augmentation_for_image_classification), [early stopping](https://en.wikipedia.org/wiki/Early_stopping) or [dropout](https://en.wikipedia.org/wiki/Dilution_(neural_networks)).
### Learning rate optimizations
When doing 50 iterations, the accuracy starts to oscillate at an accuracy of around 99.5%. This indicates that we are overstepping the minimum of the loss function, which prevents us from fine-tuning the weights. We could solve this use by implementing a [learning rate schedule](https://en.wikipedia.org/wiki/Learning_rate#Learning_rate_schedule), which decreases the learning rate as we train the model.

## Performance
|Implementation|Time to train and test|
|--------------|----------------------|
|C, without -O3|6m42s                 |
|C, with -O3   |1m06s                 |
|Python        |2m03s                 |

## Possible performance improvements
- Reuse `z1`, `a1` and `a2` from feed forward as a back propagation parameter during training to avoid doing the same calculations twice
- Create a general training state to avoid repeated `malloc()` and `free()` of variables like `w2T`, `dz1` and `dw2`
- Combine all training images into (for example) a 60000x784 matrix to improve training speed by reducing overhead
- Create a separate matrix multiplication function that acts on a to be transposed matrix by using [column-major order](https://en.wikipedia.org/wiki/Transpose#Implementation_of_matrix_transposition_on_computers)
- Parallelize functions that act on matrices
- Use a faster [matrix multiplication algorithm](https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm)
- Use CUDA to take advantage of the GPU

## Resources
- Basic neural network: https://www.geeksforgeeks.org/numpy/implementation-of-neural-network-from-scratch-using-numpy/
- Explanation video: https://www.youtube.com/watch?v=w8yWXqWQYmU
- Reference notebook: https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras
- File reading in C: https://www.man7.org/linux/man-pages/man3/fread.3.html
- Memory allocation in C: https://www.man7.org/linux/man-pages/man3/malloc.3.html

## License
See [the license file](LICENSE).
