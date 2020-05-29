#Filename: topsis.py
#Description: Implementation of MCDA method TOPSIS
#Author: Elissaios Sarmas. November 3, 2019.
import csv
import numpy as np
import matplotlib.pyplot as plt
import math

#Section 1: Read Input Data such as Alternatives, Criteria, Weights, Veto Thresholds and the Decision Matrix

criteria = 3
alternatives = 4

optimizationType = [0, 0, 0]
weights = [0.4, 0.1, 0.5]
alternativeName = ['Security1', 'Security2', 'Security3', 'Security4']
decisionMatrix = [[20, 5, 20], [15, 3, 5], [12, 2, 10], [10, 4, 35]]

#Section 2: Calculation of the Normalised Decision Matrix

normalisedDecisionMatrix = [[0 for i in range(criteria)] for y in range(alternatives)]
for j in range(criteria):
    sumOfPows = 0
    for i in range(alternatives):
        sumOfPows = sumOfPows + math.pow(decisionMatrix[i][j],2)
        sqSumOfPows =  math.sqrt(sumOfPows)
        for i in range(alternatives):
            normalisedDecisionMatrix[i][j] = decisionMatrix[i][j]*1.0 / sqSumOfPows
print("Normalised Matrix")
print(normalisedDecisionMatrix)

#Section 3: Calculation of Weighted Decision Matrix

weightedDecisionMatrix = [[0 for i in range(criteria)] for y in range(alternatives)]
for j in range(criteria):
    for i in range(alternatives):
        weightedDecisionMatrix[i][j] = normalisedDecisionMatrix[i][j] * weights[j]
print("weighted Matrix")
print(weightedDecisionMatrix)

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

print("Ideal Solution")
print(idealSolution)
print("Non Ideal Solution")
print(nonIdealSolution)

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

print("sPlus")
print(sPlus)
print("sMinus")
print(sMinus)

#Section 6: Relative Closeness of each alternative to the ideal solution

C = [0 for i in range(alternatives)]
C2 = [0 for i in range(alternatives)]
for i in range(alternatives):
    C2[i] = round(round(sMinus[i]*1.0 / (sMinus[i] + sPlus[i]),4) * 100,2) #percentage
    C[i] = sMinus[i]*1.0 / (sMinus[i] + sPlus[i])

print("C")
print(C)

plt.bar(alternativeName, C, color = 'b', edgecolor = 'black')
plt.xlabel('Alternatives')
plt.ylabel('Relative Closeness')
plt.title('TOPSIS Method', fontsize=16)
ax = plt.gca()
ax.set_facecolor('red')
plt.grid()

result = [[0 for x in range(2)] for y in range(alternatives)]
for i in range(alternatives):
    result[i][0] = alternativeName[i]
    result[i][1] = C2[i]

result = sorted(result, key=lambda tup: tup[1], reverse=True)

print(result)
plt.show()