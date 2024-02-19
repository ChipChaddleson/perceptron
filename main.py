# work time from 16:48 till 19:26
# two hours 38 mineuts

import random
from math import log10
import matplotlib.pyplot as plt


back = -3

def broke(s, l):
    b = []
    for i in range(len(s)):
        b.append(int(s[i]))
    while len(b) < l:
        b.insert(0, 0)
    return(b)

l = 11

trainingInputs = [broke(str(bin(i).removeprefix("0b")), l) for i in range(200)]
labels = [1 if item[back] == 1 else 0 for item in trainingInputs]

testData = [broke(str(bin(i).removeprefix("0b")), l) for i in range(250, 350)]
testLabels = [1 if item[back] == 1 else 0 for item in testData]

def predict(weights, bias, inputs):


    activation = bias
    for i, inputValue in enumerate(inputs):
        # calculate the weighted sum of inputs and add the bias
        activation += inputValue * weights[i]

    # apply the activation function (threshold at 0)
    return 1 if activation >= 0 else 0

# Combine training data and labels
trainingDataWithLabels = list(zip(trainingInputs, labels))

# Combine test data and labels
testDataWithLabels = list(zip(testData, testLabels))

# Shuffle both training and test data while maintaining alignment
random.shuffle(trainingDataWithLabels)
random.shuffle(testDataWithLabels)

# Split shuffled data and labels
trainingInputs, labels = zip(*trainingDataWithLabels)
testData, testLabels = zip(*testDataWithLabels)



def train(weights, bias, trainingInputs, labels, learningRate=0.01, epochs=1):
    weightHistory = []
    accuracyHistory = []
    for _ in range(epochs):
        for inputs, label in zip(trainingInputs, labels):
            # make a prediction
            prediction = predict(weights, bias, inputs)

            # calculate the error
            error = label - prediction

            # update the bias using gradient descent
            bias += learningRate * error

            # update the weights using gradient descent
            for i, inputValue in enumerate(inputs):
                weights[i] += learningRate * error * inputValue

            # Calculate and print accuracy after each step
            accuracy = calculate_accuracy(trainingInputs, labels)
            accuracyHistory.append(accuracy)
            weightHistory.append(weights.copy())
            formattedWeights = [round(num, round(log10(1/learningRate))) for num in weights]
                                            # this looks complex because it is but just ignore it ok
                                            # its just a way of convering learning rate to decimal places
            
            print(f"{accuracy}% {formattedWeights}")

    return weights, bias, weightHistory, accuracyHistory

def calculate_accuracy(inputs, labels):
    correct_predictions = 0
    total_predictions = len(inputs)
    for input_data, label in zip(inputs, labels):
        prediction = predict(weights, bias, input_data)
        if prediction == label:
            correct_predictions += 1
    accuracy = correct_predictions / total_predictions * 100
    return accuracy


weights = [0] * l   # only two inputs
# weights = [0, 0, 0, 0, 0, 0, -1, 5, 2, 0, 0]   # only two inputs

bias = 0
weights, bias, weightHistory, accuracyHistory = train(weights, bias, trainingInputs, labels)

plt.figure(figsize=(10, 5))

plt.subplot(1, 1, 1)
plt.plot(range(len(accuracyHistory)), accuracyHistory)
plt.title('Accuracy History')
plt.xlabel('Iterations')
plt.ylabel('Accuracy (%)')

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1, projection='3d')

for i in range(len(weights)):
    ax.plot(range(len(weightHistory)), [i+1] * len(weightHistory), [wh[i] for wh in weightHistory], label=f'Weight {i+1}')

ax.set_title('Weight History')
ax.set_xlabel('Iterations')
ax.set_ylabel('Position')
ax.set_zlabel('Weight Value')
ax.legend()

plt.tight_layout()
plt.show()

for inputs in trainingInputs[0:5]:
    print(f"Input: {inputs}, Predicted: {predict(weights, bias, inputs)}, expected: {inputs[back]}")
print("Training accuracy:", calculate_accuracy(trainingInputs, labels))

print("\n\n")

# example of overfitting
for inputs in testData[0:5]:
    print(f"Input: {inputs}, Predicted: {predict(weights, bias, inputs)}, expected: {inputs[back]}")
print("Test accuracy:", calculate_accuracy(testData, testLabels))



print(weights)
print(bias)