import numpy as np
import matplotlib as mat


def informationListConversion(set, labels):
    """
    This is for formatting our information into lists so we can
    manipulate the data into matrices, DO NOT TOUCH; ask for johnsanjose
    for explanation on how this works.
    :param set: this is the string-ified version of the set
    :param labels: this is the string-ified version of the labels
    :return: set and label.
    """
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
        if labelsStringList[i] == '10':
            trainingLabels.append(float(0))
        else:
            trainingLabels.append(float(labelsStringList[i]))

    return [trainingSet, trainingLabels]

def matrixSetConversion(set):
    """
    This function creates a set of matrices based off of the
    set file its been given. DO NOT TOUCH. Again, ask johnsanjose
    for questions.
    :param set: Set list (a list of values from the set file).
    :return: 9 numpy matrices.
    """

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

def matrixLabelConversion(label):
    """
    This function creates 1x400 column vectors from the label thats been passed in.
    :param label: test label
    :return: 10 1x400 matrices.
    """
    ListOfMatrices = []

    count = 0
    for i in range(0, 10):
        tempCol = []
        for j in range(0, 400):
            tempCol.append(label[count])
            count += 1
        tempArrayMatrix = np.matrix(tempCol)
        ListOfMatrices.append(tempArrayMatrix)

    return ListOfMatrices


def testConversion(setString, labelString):
    """
    Converts test set and test label into matrices to be processed. Again do not touch, assume that
    this function returns matrices for the test set and the test label.
    :param setString:
    :param labelString:
    :return: 2 numpy matrices corresponding to testSet and testLabels.
    """
    tempSetListString = setString.split(',')
    tempSet = []
    for i in range(0, len(tempSetListString)):
        if '\n' in tempSetListString[i]:
            currStringList = tempSetListString[i].split("\n")
            for j in range(0, len(currStringList)):
                if currStringList[j] != '':
                    tempSet.append(float(currStringList[j]))
        else:
            tempSet.append(float(tempSetListString[i]))


    count = 0
    tempMatrixRow = []
    for i in range(0, 1000):
        tempMatrixCol = []
        for j in range(0, 400):
            tempMatrixCol.append(tempSet[count])
            count += 1
        tempMatrixRow.append(tempMatrixCol)

    matrixTestSet = np.matrix(tempMatrixRow)

    tempLabel = labelString.split('\n')
    tempLabel.pop(len(tempLabel) - 1)

    print(len(tempLabel))

    colArray = []
    for i in range(0, len(tempLabel)):
        colArray.append(float(tempLabel[i]))

    matrixTestLabel = np.matrix(colArray)
    matrixTestLabel = matrixTestLabel.T

    return matrixTestSet, matrixTestLabel


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


    [matrixTestSet, matrixTestLabels] = testConversion(stringTestSet, stringTestLabelsSet)
    A = matrixSetConversion(trainingSet)
    y = matrixLabelConversion(trainingLabels)


    """-------------------------DO NOT TOUCH ANYTHING FROM THIS POINT-------------------------"""
    #TODO: Perform SVD and find the corresponding z_hat for all matrices A (ten of them).
    #NOTE: use variables, A, b, matrixTestSet, and matrixTestLabels.
    #A is a list of 10 matrices, (400x400)
    #b is a list of 10 column vector matrix (400x1)
    #matrixTestSet is a matrix from the test set txt file. (1000x400)
    #matrixTestLabel is a column matrix from test label txt file (1000x1)

    return 0


if __name__ == '__main__':
    main()