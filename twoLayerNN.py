from perceptronClass import runTwoLayerNN, normalize
import numpy as np

# Computes the dot product and returns 1 if >= 0 and -1 otherwise
def makeGuess(weightVector, challengeVector):
    innerProd = np.dot(weightVector, challengeVector)

    if(innerProd >= 0):
        return 1
    else:
        return -1


# Reads in from the training file
def readTrainingSet(trainFileStream):
    trainingSet = []
    answers = []

    for line in trainFileStream:
        splitLine = line.split()

        answer = splitLine.pop()
        point = np.array([float(x) for x in splitLine])

        trainingSet.append(point)
        answers.append(answer)


    return(trainingSet, answers)


# Reads in from test file
def readTestSet(testFileStream):
    testSet = []

    for line in testFileStream:
        splitLine = line.split()
        point = np.array([float(x) for x in splitLine])

        testSet.append(point)


    return(testSet)


# Makes guesses on unknown elements
def doTest(testSet, weightVector, answers = None):

    if (answers == None):
        for i in range(len(testSet)):
            point = testSet[i]
            guess = makeGuess(weightVector, point)

            print("Guess: " + str(guess))
    else:
        for i in range(len(testSet)):
            point = testSet[i]
            answer = answers[i]

            guess = makeGuess(weightVector, point)

            print("Guess: " + str(guess) + "   Answer: " + str(answer))


# Saves the outputUnit and the hiddenUnits to a file as specified by Tino
def saveNN(outputUnit, hiddenUnits):
    file = open("./Files/NN.txt", "wb")

    file.write(str(len(hiddenUnits)) + "\n")

    for weight in outputUnit.weightVector:
        file.write(str(weight) + " ")
    file.write("\n")

    for unit in hiddenUnits:
        for weight in unit.weightVector:
            file.write(str(weight) + " ")
        file.write("\n")

    file.close()


# Saves the boolean predictions
def saveGuesses(guesses):
    file = open("./Files/Predictions.txt", "wb")

    for guess in guesses:
        file.write(guess + ' ')

    file.write('\n')

    file.close()






##### MAIN SCRIPT #####
trainFileStream = open("./Files/TrainSet.txt", "rb")
fullSet, answers = readTrainingSet(trainFileStream)
trainFileStream.close()

fullSetNorm = [normalize(v) for v in fullSet]

outputUnit, hiddenUnits = runTwoLayerNN(fullSetNorm, answers, numHiddenUnits= 2, verbose=True, learnRate=0.5)

for i in range(len(fullSet)):
    challengeVector = fullSet[i]

    # Running challenge vector through all hidden units
    hiddenUnitOutputs = []
    for unit in hiddenUnits:
        unit.runForward(challengeVector)
        hiddenUnitOutputs.append(unit.y)

    # Running hiddenUnit outputs into outputUnit
    outputUnit.runForward(hiddenUnitOutputs)

    # Getting the guess and correct answer
    if (outputUnit.y >= 0.5):
        guess = "Y"
    else:
        guess = "N"

    if (answers[i] == "1.0"):
        correctAnswer = "Y"
    else:
        correctAnswer = "N"

    if(not guess == correctAnswer):
        print "FAIL"

testFileStream = open("./Files/TestSet.txt")
testSet = readTestSet(testFileStream)
testFileStream.close()

guesses = []
for i in range(len(testSet)):
    challenge = testSet[i]

    hiddenUnitOutputs = []
    for unit in hiddenUnits:
        unit.runForward(challenge)
        hiddenUnitOutputs.append(unit.y)

    outputUnit.runForward(hiddenUnitOutputs)

    if outputUnit.y >= 0.5:
        guesses.append("+1")
    else:
        guesses.append("-1")




while True:
    answer = raw_input("Would you like to save? (y/n): ")

    if answer == "y":
        saveNN(outputUnit, hiddenUnits)
        saveGuesses(guesses)
        break
    elif answer == "n":
        break
    else:
        print "Input not recognized!"





