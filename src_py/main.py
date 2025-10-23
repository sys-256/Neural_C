# Import packages
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

# Function to read the data set
def read_idx(filename: str) -> NDArray[np.uint8]:
    with open(filename, 'rb') as f:
        # Read the magic number (first 4 bytes)
        magic = int.from_bytes(f.read(4), 'big')
        
        # Determine the number of dimensions
        if magic in (2051, 2049):  # 2051 for images, 2049 for labels
            num_dims = 3 if magic == 2051 else 1
            # Read the remaining dimensions
            dims: list[int] = []
            while True:
                dim = int.from_bytes(f.read(4), 'big')
                dims.append(dim)
                if len(dims) == num_dims:
                    break
            
            # Read the data
            data: NDArray[np.uint8] = np.fromfile(f, dtype=np.uint8)
            
            # Reshape the data into the appropriate dimensions
            if magic == 2051:
                # Images have shape (num_images, height * width)
                return data.reshape(dims[0], 1, dims[1] * dims[2])
            elif magic == 2049:
                # Labels have shape (num_labels,)
                return data
        else:
            raise ValueError("Invalid IDX file")
    return np.array(0) # Will never be reached

# Read data and labels and convert to numpy array
data = read_idx("data/train-images-idx3-ubyte")
data = data.astype(np.float64) / np.max(data) # Divide by max value to prevent int overflow
labels = read_idx("data/train-labels-idx1-ubyte")
labels = np.array([np.array([1 if i == labels[j] else 0 for i in range(10)]).reshape(1,10) for j in range(len(labels))])

data_test = read_idx("data/t10k-images-idx3-ubyte")
data_test = data_test.astype(np.float64) / np.max(data_test) # Divide by max value to prevent int overflow
labels_test = read_idx("data/t10k-labels-idx1-ubyte")
labels_test = np.array([np.array([1 if i == labels_test[j] else 0 for i in range(10)]).reshape(1,10) for j in range(len(labels_test))])

# Initialize random weights
def generate_weights(rows: int, columns: int):
	return np.random.rand(rows, columns)

# Activation functions
# Must be preserved: z
def ReLU(z: NDArray[np.float64]) -> NDArray[np.float64]:
	if z.shape != (1, n_hidden):
		raise ValueError()
	return np.maximum(0, z)

# Can be changed: z
def ReLU_prime(z: NDArray[np.float64]) -> NDArray[np.bool]:
	if z.shape != (1, n_hidden):
		raise ValueError()
	return z > 0

# Can be changed: z
def softmax(z: NDArray[np.float64]) -> NDArray[np.float64]:
	if z.shape != (1, n_output):
		raise ValueError()
	z -= np.max(z)
	return np.exp(z) / np.sum(np.exp(z))

# Get most likely number
# Must be preserved: z
def get_predictions(z: NDArray[np.float64]):
	if z.shape != (1, n_output):
		raise ValueError()
	return int(np.argmax(z))

# Feed forward algorithm
# Must be preserved: x, w1, w2
def feed_forward(x: NDArray[np.float64], w1: NDArray[np.float64], w2: NDArray[np.float64]
                 ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
	if x.shape != (1, n_input):
		raise ValueError()
	if w1.shape != (n_input, n_hidden):
		raise ValueError()
	if w2.shape != (n_hidden, n_output):
		raise ValueError()
	z1 = x.dot(w1)		# Unactivated hidden layer
	a1 = ReLU(z1)		# Activated hidden layer
	z2 = a1.dot(w2)		# Unactivated output layer
	a2 = softmax(z2)	# Activated output layer
	return z1, a1, a2 # z2 isn't used in any of the function calls

# Back propagation of error algorithm
# Must be preserved: x, y
# Can be changed: w1, w2
def back_propagation(x: NDArray[np.float64], y: NDArray[np.uint8], w1: NDArray[np.float64], w2: NDArray[np.float64], alpha: float):
	if x.shape != (1, n_input):
		raise ValueError()
	if y.shape != (1, n_output):
		raise ValueError()
	if w1.shape != (n_input, n_hidden):
		raise ValueError()
	if w2.shape != (n_hidden, n_output):
		raise ValueError()
	z1, a1, a2 = feed_forward(x, w1, w2)
	
	# Error in output layer
	dz2 = a2 - y
	dz1 = dz2.dot(w2.T) * ReLU_prime(z1)
	
	# Gradient for w1 and w2
	dw2 = a1.T.dot(dz2)
	dw1 = x.T.dot(dz1)
	
	# Updating parameters
	w2 -= alpha * dw2
	w1 -= alpha * dw1

	return w1, w2

# Must be preserved: x, y
# Can be changed: w1, w2
def train(x: NDArray[np.float64], y: NDArray[np.uint8], w1: NDArray[np.float64], w2: NDArray[np.float64], alpha: float, iterations: int):
	if x.shape != (m, 1, n_input):
		raise ValueError()
	if y.shape != (m, 1, n_output):
		raise ValueError()
	if w1.shape != (n_input, n_hidden):
		raise ValueError()
	if w2.shape != (n_hidden, n_output):
		raise ValueError()
	accuracy: list[float] = []

	for i in range(iterations):
		predictions: list[bool] = []
		for j in range(len(x)):
			_, _, a2 = feed_forward(x[j], w1, w2)

			# On the last iteration, show all numbers it got incorrectly
			# if (i+1 == iterations and get_predictions(a2) != np.argmax(y[j])):
			# 	print("Prediction:", get_predictions(a2), "Actual:", np.argmax(y[j]))
			# 	plt.imshow(x[j].reshape(28, 28))
			# 	plt.show()
			predictions.append(get_predictions(a2) == np.argmax(y[j]))
			w1, w2 = back_propagation(x[j], y[j], w1, w2, alpha)

		if (i+1) % 5 == 0:
			print("Iteration:", i + 1, "\tAccuracy:", 100 * np.sum(predictions) / len(x)) 
		accuracy.append(100 * np.sum(predictions) / len(x))

	return accuracy, w1, w2

# Must be preserved: x, y, w1, w2
def test_model(x: NDArray[np.float64], y: NDArray[np.uint8], w1: NDArray[np.float64], w2: NDArray[np.float64]):
	if x.shape != (m_test, 1, n_input):
		raise ValueError()
	if y.shape != (m_test, 1, n_output):
		raise ValueError()
	if w1.shape != (n_input, n_hidden):
		raise ValueError()
	if w2.shape != (n_hidden, n_output):
		raise ValueError()

	predictions: list[bool] = []
	for j in range(len(x)):
		_, _, a2 = feed_forward(x[j], w1, w2)
		predictions.append(get_predictions(a2) == np.argmax(y[j]))

	print("\nAccuracy on test set:", 100 * np.sum(predictions) / len(x))
	return

# Must be preserved: x, w1, w2
def predict(x: NDArray[np.float64], w1: NDArray[np.float64], w2: NDArray[np.float64]):
	if x.shape != (1, n_input):
		raise ValueError()
	if w1.shape != (n_input, n_hidden):
		raise ValueError()
	if w2.shape != (n_hidden, n_output):
		raise ValueError()
	_, _, a2 = feed_forward(x, w1, w2)
	k = get_predictions(a2)
	print("Number:", k)

# n_input = 3
# n_hidden = 5
# n_output = 2
# input = np.array((0.840, 0.394, 0.783)).reshape(1,3)
# w1 = np.array((0.197, 0.996, 0.422, 0.208, 0.689, 0.566, 0.392, 0.981, 0.768, 0.748, 0.734, 0.445, 0.759, 0.965, 0.452)).reshape(3,5)
# w2 = np.array((0.806, 0.800, 0.804, 0.131, 0.141, 0.912, 0.084, 0.624, 0.238, 0.373)).reshape(5,2)
# output = np.array((0,1)).reshape(1,2)
# z1, a1, a2 = feed_forward(input, w1, w2)
# print(z1, a1, a2)
# w1, w2 = back_propagation(input, output, w1, w2, 0.5)
# print(w1, w2)

# Generate weights
m = len(data)
m_test = len(data_test)
n_input = data.shape[2]
n_hidden = 20
n_output = labels.shape[2]
w1 = generate_weights(n_input, n_hidden)
w2 = generate_weights(n_hidden, n_output)

# Train model
accuracy, w1, w2 = train(data, labels, w1, w2, 0.01, 50)

# Plot accuracy
plt.plot(accuracy)
plt.ylabel("Accuracy")
plt.xlabel("Iterations")
plt.show()

test_model(data_test, labels_test, w1, w2)
