import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.random.randn(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0  # Step function

    def predict(self, x):
        x = np.insert(x, 0, 1)  # Add bias input
        return self.activation(np.dot(self.weights, x))

    def train(self, X, y):
        for epoch in range(self.epochs):
            for i in range(len(X)):
                x_i = np.insert(X[i], 0, 1)  # Add bias input
                y_pred = self.activation(np.dot(self.weights, x_i))
                error = y[i] - y_pred
                self.weights += self.learning_rate * error * x_i  # Weight update

    def evaluate(self, X, y):
        correct = sum(self.predict(x) == y_true for x, y_true in zip(X, y))
        accuracy = correct / len(y)
        return accuracy

# NAND Truth Table
X_NAND = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_NAND = np.array([1, 1, 1, 0])  # NAND output

# XOR Truth Table
X_XOR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_XOR = np.array([0, 1, 1, 0])  # XOR output

# Train perceptron on NAND
perceptron_nand = Perceptron(input_size=2)
perceptron_nand.train(X_NAND, y_NAND)
print("NAND Perceptron Accuracy:", perceptron_nand.evaluate(X_NAND, y_NAND))

# Train perceptron on XOR (Perceptron will fail due to non-linearity)
perceptron_xor = Perceptron(input_size=2)
perceptron_xor.train(X_XOR, y_XOR)
print("XOR Perceptron Accuracy:", perceptron_xor.evaluate(X_XOR, y_XOR))
