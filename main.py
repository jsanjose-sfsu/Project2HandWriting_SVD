import numpy as np


def informationListConversion(set, labels):
    '''
    This is for formatting our information into lists so we can
    manipulate the data into matrices, DO NOT TOUCH; ask for johnsanjose
    for explanation on how this works.
    :param set: this is the string-ified version of the set
    :param labels: this is the string-ified version of the labels
    :return: set and label.
    '''
    trainingSetStringList = set.split(",")
    trainingSet = []
    for i in range(0, len(trainingSetStringList)):
        if "\n" in trainingSetStringList[i]:
            currStringList = trainingSetStringList[i].split("\n")

            for j in range(0, len(currStringList)):
                if currStringList[j] != '':
                    trainingSet.append(float(currStringList[j]))
        else:
            trainingSet.append(float(trainingSetStringList[i]))

    labelsStringList = labels.split("\n")
    labelsStringList.pop(len(labelsStringList) - 1)

    trainingLabels = []
    for i in range(0, len(labelsStringList)):
        trainingLabels.append(float(labelsStringList[i]))

    return [trainingSet, trainingLabels]

def matrixSetConversion(set):
    '''
    This function creates a set of matrices based off of the
    set file its been given. DO NOT TOUCH. Again, ask johnsanjose
    for questions.
    :param set: Set list (a list of values from the set file).
    :return: 9 numpy matrices.
    '''

    ListOfMatrices = []

    count = 0
    #we need to do 10 times
    for i in range(0, 10):
        tempRow =[]
        for k in range(0, 400):
            tempCol = []
            for j in range(0, 400):
                tempCol.append(set[count])
                count += 1
            tempRow.append(tempCol)

        A_transposed = np.matrix(tempRow)
        ListOfMatrices.append(A_transposed.T)

    return ListOfMatrices


def main():

    trainingSetMatrixFile = open("HandWrittenDataFiles/handwriting_training_set.txt", "r")
    trainingSetLabels = open("HandWrittenDataFiles/handwriting_training_set_labels.txt", "r")

    testSetFile= open("HandWrittenDataFiles/handwriting_test_set.txt")
    testLabelsFile = open("HandWrittenDataFiles/handwriting_test_set_labels.txt")

    stringTrainingSet = trainingSetMatrixFile.read()
    stringTrainingSetLabels = trainingSetLabels.read()

    stringTestSet = testSetFile.read()
    stringTestLabelsSet = testLabelsFile.read()

    [trainingSet, trainingLabels] = informationListConversion(stringTrainingSet, stringTrainingSetLabels)
    [testSet, testLabels] = informationListConversion(stringTrainingSet, stringTestLabelsSet)

    A = matrixSetConversion(trainingSet)

    return 0


if __name__ == '__main__':
    main()