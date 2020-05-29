#Filename: topsis.py
#Description: Implementation of MCDA method TOPSIS
#Author: Elissaios Sarmas. November 3, 2019.
import csv
import numpy as np
import matplotlib.pyplot as plt
import math

inputname = "topsis2.csv"

file =  open(inputname, "rt")
list1 = list(csv.reader(file))

#Section 1: Read Input Data such as Alternatives, Criteria, Weights, Veto Thresholds and the Decision Matrix

criteria = int(list1[0][1])
alternatives = int(list1[1][1])
weights = [0 for y in range(criteria)]
optimizationType = [0 for y in range(criteria)]
for i in range(criteria):
    optimizationType[i] = int(list1[2][i+1])
    weights[i] = float(list1[3][i+1])
decisionMatrix = [[0 for y in range(criteria)] for x in range(alternatives)]
companyName = ["" for i in range(alternatives)]
for i in range(alternatives):
    companyName[i] = list1[i+16][0]
    for j in range(criteria):
        decisionMatrix[i][j] = float(list1[i+16][j+1])

#Section 2: Calculation of the Normalised Decision Matrix

normalisedDecisionMatrix = [[0 for i in range(criteria)] for y in range(alternatives)]
for j in range(criteria):
    sumOfPows = 0
    for i in range(alternatives):
        sumOfPows = sumOfPows + math.pow(decisionMatrix[i][j],2)
        sqSumOfPows =  math.sqrt(sumOfPows)
        for i in range(alternatives):
            normalisedDecisionMatrix[i][j] = decisionMatrix[i][j]*1.0 / sqSumOfPows

#Section 3: Calculation of Weighted Decision Matrix

weightedDecisionMatrix = [[0 for i in range(criteria)] for y in range(alternatives)]
for j in range(criteria):
    for i in range(alternatives):
        weightedDecisionMatrix[i][j] = normalisedDecisionMatrix[i][j] * weights[j]

#Section 4: Calculation of Ideal and Non-Ideal Solutions

idealSolution = [0 for i in range(criteria)]
nonIdealSolution = [0 for i in range(criteria)]
for j in range(criteria):
    maxValue = -100000000000
    minValue = 100000000000
    for i in range(alternatives):
        if weightedDecisionMatrix[i][j] < minValue:
            minValue = weightedDecisionMatrix[i][j]
        if weightedDecisionMatrix[i][j] > maxValue:
            maxValue = weightedDecisionMatrix[i][j]
    if optimizationType[j] == 0:
        idealSolution[j] = maxValue
        nonIdealSolution[j] = minValue
    elif optimizationType[j] == 1:
        idealSolution[j] = minValue
        nonIdealSolution[j] = maxValue

#Section 5: Calculation of Separation Distance of each alternative

sPlus = [0 for i in range(alternatives)]
sMinus = [0 for i in range(alternatives)]
for i in range(alternatives):
    sumPlusTemp = 0
    sumMinusTemp = 0
    for j in range(criteria):
        sumPlusTemp = sumPlusTemp + math.pow(idealSolution[j]-weightedDecisionMatrix[i][j],2)
        sumMinusTemp = sumMinusTemp + math.pow(nonIdealSolution[j]-weightedDecisionMatrix[i][j],2)
    sPlus[i] = math.sqrt(sumPlusTemp)
    sMinus[i] = math.sqrt(sumMinusTemp)

#Section 6: Relative Closeness of each alternative to the ideal solution

C = [0 for i in range(alternatives)]
C2 = [0 for i in range(alternatives)]
for i in range(alternatives):
    C2[i] = round(round(sMinus[i]*1.0 / (sMinus[i] + sPlus[i]),4) * 100,2) #percentage
    C[i] = sMinus[i]*1.0 / (sMinus[i] + sPlus[i])

print(C)

plt.bar(companyName, C, color = 'b', edgecolor = 'black')
plt.xlabel('Alternatives')
plt.ylabel('Relative Closeness')
plt.title('TOPSIS Method', fontsize=16)
ax = plt.gca()
ax.set_facecolor('red')
plt.grid()

result = [[0 for x in range(2)] for y in range(alternatives)]
for i in range(alternatives):
    result[i][0] = companyName[i]
    result[i][1] = C2[i]

result = sorted(result, key=lambda tup: tup[1], reverse=True)

print(result)
plt.show()