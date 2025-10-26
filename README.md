# Neural_C

## TODO:
- check todos (in project)
- update performance figures
- add license

## About
- A simple neural network written in C, using only libc and libm
- Recognizes handwritten digits from [the MNIST database](https://systemds.apache.org/datasets/mnist).
- Converts a 28x28 image to an input layer with 784 nodes
- Only uses weights, so no biases (reduces accuracy but makes it easier to implement)
- Uses [ReLU](https://en.wikipedia.org/wiki/Rectified_linear_unit) to activate the hidden layer of 20 nodes
- Uses [softmax](https://en.wikipedia.org/wiki/Softmax_function) to generate predictions for each digit

## Why
I wanted to learn more about neural networks, linear algebra and programming in C (like pointers, 2D arrays, file reading and memory allocation), so I decided to implement a neural network in C. I started by [learning more about neural networks](#resources), and then [implementing one in Python](src_py/main.py) using only numpy. After that, I gradually rewrote each Python function in C. The result can be found in [src](src). The main function inside [main.c](src/main.c) demonstrates the functionality of each function, including the ones that train the model and test it. Spoiler: the accuracy after 25 iterations is around 94%.

## Performance
|Implementation|Time to train and test|
|--------------|----------------------|
|C, without -O3|6m59s                 |
|C, with -O3   |1m12s                 |
|Python        |2m00s                 |

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
