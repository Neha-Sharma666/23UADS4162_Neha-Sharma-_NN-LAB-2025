import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)  # Derivative of sigmoid

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5, epochs=10000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Initialize weights randomly
        self.W1 = np.random.randn(self.input_size, self.hidden_size)  
        self.b1 = np.random.randn(self.hidden_size)  
        self.W2 = np.random.randn(self.hidden_size, self.output_size)  
        self.b2 = np.random.randn(self.output_size)  

    def forward(self, X):
        # Forward pass
        self.hidden_input = np.dot(X, self.W1) + self.b1
        self.hidden_output = sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.W2) + self.b2
        self.final_output = sigmoid(self.final_input)
        return self.final_output

    def backward(self, X, y):
        # Compute error
        output_error = y - self.final_output
        output_delta = output_error * sigmoid_derivative(self.final_output)

        hidden_error = np.dot(output_delta, self.W2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.W2 += np.dot(self.hidden_output.T, output_delta) * self.learning_rate
        self.b2 += np.sum(output_delta, axis=0) * self.learning_rate
        self.W1 += np.dot(X.T, hidden_delta) * self.learning_rate
        self.b1 += np.sum(hidden_delta, axis=0) * self.learning_rate

    def train(self, X, y):
        for epoch in range(self.epochs):
            self.forward(X)
            self.backward(X, y)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - self.final_output))
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    def predict(self, X):
        return self.forward(X) > 0.5  # Convert probabilities to binary outputs

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
y = np.array([[0], [1], [1], [0]])  # Expected XOR outputs

# Create MLP with 2 input neurons, 2 hidden neurons, and 1 output neuron
mlp = MLP(input_size=2, hidden_size=2, output_size=1)

# Train MLP
mlp.train(X, y)

# Test the trained MLP on XOR inputs
predictions = mlp.predict(X)
print("\nPredictions for XOR:")
for i in range(len(X)):
    print(f"Input: {X[i]} -> Predicted: {int(predictions[i])} | Expected: {y[i][0]}")
