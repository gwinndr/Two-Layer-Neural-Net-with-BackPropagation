import random
import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + math.e ** -x)


class PerceptronUnit:
    def __init__(self, numDimensions):
        self.y = None # Output from threshold
        self.z = None # Raw output

        self.weightVector =  normalize(np.array([random.randint(0, 5000) for i in range(numDimensions)]))


    # Getting the raw output z and calling runThreshold
    def runForward(self, challenge):
        self.z = np.dot(self.weightVector, challenge)
        self.runThreshold()


    # Using tanh(z) as threshold
    def runThreshold(self):
        self.y = sigmoid(self.z)


    # Derivative of tanh
    def derivativeThreshold(self):
        return sigmoid(self.z) * (1-sigmoid(self.z))

    # Normalizes weight vector
    def normalizeWeights(self):
        self.weightVector = normalize(self.weightVector)




# Starts and runs a 2 layer neural network
def runTwoLayerNN(trainingSet, correctAnswers, numHiddenUnits, learnRate=1, verbose=False):
    numDimensions = len(trainingSet[0])

    # Creating output unit (outputs from hidden units connected so thats the number of "dimensions")
    outputUnit = PerceptronUnit(numHiddenUnits)

    # Creating hidden units on layer 2 (all inputs connected to this unit so numdimensions is the "real" dimensions
    hiddenUnits = [PerceptronUnit(numDimensions) for i in range(numHiddenUnits)]

    try:
        # Running the algorithm for all rows in the training set
        numChanges = 0
        i = 0
        while (i < len(trainingSet)):
            challengeVector = trainingSet[i]

            # Running challenge vector through all hidden units
            hiddenUnitOutputs = []
            for unit in hiddenUnits:
                unit.runForward(challengeVector)
                hiddenUnitOutputs.append(unit.y)

            # Running hiddenUnit outputs into outputUnit
            outputUnit.runForward(hiddenUnitOutputs)

            # Getting the guess and correct answer
            if (outputUnit.y >= 0.5):
                guess = 1
            else:
                guess = -1


            if (correctAnswers[i] == "1.0"):
                correctAnswer = 1
            else:
                correctAnswer = -1


            # If the guess was wrong back propogate to update weights
            if (not guess == correctAnswer):
                outputDelta = -2 * (correctAnswer - outputUnit.y) * outputUnit.derivativeThreshold()

                outputUnitSensitivities = np.zeros(numHiddenUnits)
                for i in range(numHiddenUnits):
                    outputUnitSensitivities[i] = outputDelta * hiddenUnits[i].y

                hiddenSensitivities = []
                for i in range(numHiddenUnits):
                    deltaHidden = outputDelta * outputUnit.weightVector[i] * hiddenUnits[i].derivativeThreshold()


                    sensitivities = np.zeros(numDimensions)
                    for j in range(len(challengeVector)):
                        sensitivities[j] = deltaHidden * challengeVector[j]

                    hiddenSensitivities.append(sensitivities)

                # Weight update for output unit
                for i in range(len(outputUnit.weightVector)):
                    weight = outputUnit.weightVector[i]
                    sensitivity = outputUnitSensitivities[i]

                    outputUnit.weightVector[i] = weight - (learnRate * sensitivity)
                # outputUnit.normalizeWeights()


                # Weight update for hidden units
                for i in range(numHiddenUnits):
                    unit = hiddenUnits[i]
                    sensitivities = hiddenSensitivities[i]
                    for j in range(len(unit.weightVector)):
                        weight = unit.weightVector[j]
                        sensitivity = sensitivities[j]

                        unit.weightVector[j] = weight - (learnRate * sensitivity)
                    # unit.normalizeWeights()


                i = 0
                numChanges = numChanges + 1

                if (verbose and numChanges > 50):
                    print(outputUnit.weightVector), i
                    numChanges = 0

                continue

            i = i + 1

    except KeyboardInterrupt:
        print "Interrupted"
        print outputUnit.weightVector


    return (outputUnit, hiddenUnits)




# Normalizes a vector v
def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
       return v
    return v/norm


