from typing import List

import numpy as np
import matplotlib as mat


def informationListConversion(set, labels):
    """
    This is for formatting our information into lists so we can
    manipulate the data into matrices.
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
    set file its been given. DO NOT TOUCH.
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
    This function creates 1x400 column vectors from the label thats been passed in. A
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
    this function returns matrices for the test set and the test labe.
    :param setString:
    :param labelString:
    :return: 2 numpy matrices corresponding to testSet (1000x400)and testLabels(1000X1).
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
    matrixTestSet = []
    for i in range(0, 1000):
        tempMatrixCol = []
        for j in range(0, 400):
            tempMatrixCol.append(tempSet[count])
            count += 1

        tempMatrix = np.matrix(tempMatrixCol)
        matrixTestSet.append(tempMatrix.T)

    tempLabel = labelString.split('\n')
    tempLabel.pop(len(tempLabel) - 1)


    colArray = []
    for i in range(0, len(tempLabel)):
        colArray.append(float(tempLabel[i]))

    return matrixTestSet, colArray

def orthonormalProjection(U, y):
    """
    This operation represents the orthonormal projection from y->y_hat.
    UxU.Txy = y_hat
    :param U:
    :param y:
    :return: y_hat calulated from operation described.
    """
    U_cross_UT = np.matmul(U, U.T)
    y_hat = np.matmul(U, y)

    return y_hat

def approx_Equal(x, y, tolerance=0.001):
    """
    Determines if result is approximately the right answer.
    :param x: z_norm
    :param y: test label
    :param tolerance:
    :return: True if result is close enough with tolerance false if not
    """
    return abs(x-y) <= 0.5 * tolerance * (x + y)

def main():

    #We open the files
    trainingSetMatrixFile = open("HandWrittenDataFiles/handwriting_training_set.txt", "r")
    trainingSetLabels = open("HandWrittenDataFiles/handwriting_training_set_labels.txt", "r")
    testSetFile= open("HandWrittenDataFiles/handwriting_test_set.txt")
    testLabelsFile = open("HandWrittenDataFiles/handwriting_test_set_labels.txt")

    #Parse them
    stringTrainingSet = trainingSetMatrixFile.read()
    stringTrainingSetLabels = trainingSetLabels.read()
    stringTestSet = testSetFile.read()
    stringTestLabelsSet = testLabelsFile.read()

    #Convert them to usable matrices and lists
    [trainingSet, trainingLabels] = informationListConversion(stringTrainingSet, stringTrainingSetLabels)
    [matrixTestSet, matrixTestLabels] = testConversion(stringTestSet, stringTestLabelsSet)
    A = matrixSetConversion(trainingSet)
    y = matrixLabelConversion(trainingLabels)


    """-------------------------DO NOT TOUCH ANYTHING FROM THIS POINT-------------------------"""
    #TODO: Perform SVD and find the corresponding z_hat for all matrices A (ten of them).
    #NOTE: use variables, A, b, matrixTestSet, and matrixTestLabels.
    #A is a list of 10 matrices, (400x400) where we need to get U matrices by SVD.
    #b is a list of 10 column vector matrix (400x1)
    #matrixTestSet is a matrix from the test set txt file. (1000x400) this is our set of y vectors
    #matrixTestLabel is a column matrix from test label txt file (1000x1)

    '''Gathering all orthonormal matrices from A'''
    U = []
    for i in range(0, len(A)):
        #use of SVD and grabbing the orthogonal matrix U from current matrix A
        [tempU, tempE, tempV] = np.linalg.svd(A[i])
        U.append(tempU)

    """For each test digit, compute the 10 y_hat vectors and the corresponding 10 z vectors."""
    #number of correct guesses with an adjustable tolerance
    numCorrectGuess = 0
    tolerance = .05
    #for each test digit
    for i in range(0, len(matrixTestLabels)):
        z_Mag = []
        y_hat = []
        z_hat = []
        #compute the 10 y_hat vectors
        for j in range(0, len(U)):
            y_hat.append(orthonormalProjection(U[j], matrixTestSet[i]))

        #and the corresponding 10 z vectors.
        for k in range(0, len(y_hat)):
            z_hat.append(matrixTestSet[i] - y_hat[k])

        #computing norms
        for m in range(0, len(z_hat)):
            magnitude_z = np.linalg.norm(z_hat[m])
            z_Mag.append(magnitude_z)

        #grabbing the minimum which is the prediction
        minimum = z_Mag[0]
        for l in range(0, len(z_Mag)):
            if(z_Mag[l] < minimum):
                minimum = z_Mag[l]

        #we now see if our prediction is correct by
        #comparing it with the test labels.
        print("----------------------------------------------------------")
        print("Prediction = {} : Label = {}".format(minimum,
                                                    matrixTestLabels[i]))
        approximatelyAccurate = approx_Equal(minimum, matrixTestLabels[i], tolerance)
        print("Does it approximately match?: {}".format(approximatelyAccurate))

        if approximatelyAccurate:
            numCorrectGuess += 1

    print("----------------------------------------------------------\n")

    print("Number of correct guesses: {}".format(numCorrectGuess))

    return 0


if __name__ == '__main__':
    main()