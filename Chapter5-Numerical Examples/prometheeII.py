#Filename: prometheeII.py
#Description: Implementation of MCDA method PROMETHEE II
#Author: Elissaios Sarmas. November 3, 2019.

import csv
import numpy as np
import matplotlib.pyplot as plt
import math


def usualCriterion(evaluationTable, k, alternatives, decisionMatrix, indifferenceThreshold, preferenceThreshold, weights, optimizationType):
    if optimizationType[k] == 0:
        for i in range(alternatives):
            for j in range(alternatives):
                if i!=j and decisionMatrix[i][k] >= decisionMatrix[j][k]:
                    if decisionMatrix[i][k] > decisionMatrix[j][k]:
                        evaluationTable[i][j] = evaluationTable[i][j] + 1.0 * weights[k]
    elif optimizationType[k] == 1:
        for i in range(alternatives):
            for j in range(alternatives):
                if i!=j and decisionMatrix[j][k] >= decisionMatrix[i][k]:
                    if decisionMatrix[j][k] > decisionMatrix[i][k]:
                        evaluationTable[i][j] = evaluationTable[i][j] + 1.0 * weights[k]

def quasiCriterion(evaluationTable, k, alternatives, decisionMatrix, indifferenceThreshold, preferenceThreshold, weights, optimizationType):
    if optimizationType[k] == 0:
        for i in range(alternatives):
            for j in range(alternatives):
                if i!=j and decisionMatrix[i][k] >= decisionMatrix[j][k]:
                    if decisionMatrix[i][k] - decisionMatrix[j][k] > indifferenceThreshold[k]:
                        evaluationTable[i][j] = evaluationTable[i][j] + 1.0 * weights[k]
    elif optimizationType[k] == 1:
        for i in range(alternatives):
            for j in range(alternatives):
                if i!=j and decisionMatrix[j][k] >= decisionMatrix[i][k]:
                    if decisionMatrix[j][k] - decisionMatrix[i][k] > indifferenceThreshold[k]:
                        evaluationTable[i][j] = evaluationTable[i][j] + 1.0 * weights[k]

def linearPreferenceCriterion(evaluationTable, k, alternatives, decisionMatrix, indifferenceThreshold, preferenceThreshold, weights, optimizationType):
    if optimizationType[k] == 0:
        for i in range(alternatives):
            for j in range(alternatives):
                if i!=j and decisionMatrix[i][k] >= decisionMatrix[j][k]:
                    if decisionMatrix[i][k] - decisionMatrix[j][k] > preferenceThreshold[k]:
                        evaluationTable[i][j] = evaluationTable[i][j] + 1.0 * weights[k]
                    else:
                        evaluationTable[i][j] = evaluationTable[i][j] + ((decisionMatrix[i][k] - decisionMatrix [j][k])*1.0 / preferenceThreshold[k]) * weights[k]
    elif optimizationType[k] == 1:
        for i in range(alternatives):
            for j in range(alternatives):
                if i!=j and decisionMatrix[j][k] >= decisionMatrix[i][k]:
                    if decisionMatrix[j][k] - decisionMatrix[i][k] > preferenceThreshold[k]:
                        evaluationTable[i][j] = evaluationTable[i][j] + 1.0 * weights[k]
                    else:
                        evaluationTable[i][j] = evaluationTable[i][j] + ((decisionMatrix[j][k] - decisionMatrix [i][k])*1.0 / preferenceThreshold[k]) * weights[k]



def levelCriterion(evaluationTable, k, alternatives, decisionMatrix, indifferenceThreshold, preferenceThreshold, weights, optimizationType):
    if optimizationType[k] == 0:
        for i in range(alternatives):
            for j in range(alternatives):
                if i!=j and decisionMatrix[i][k] >= decisionMatrix[j][k]:
                    if decisionMatrix[i][k] - decisionMatrix[j][k] > preferenceThreshold[k]:
                        evaluationTable[i][j] = evaluationTable[i][j] + 1.0 * weights[k]
                    elif decisionMatrix[i][k] - decisionMatrix[j][k] <= indifferenceThreshold[k]:
                        evaluationTable[i][j] = evaluationTable[i][j] + 0.0 * weights[k]
                    else:
                        evaluationTable[i][j] = evaluationTable[i][j] + 0.5 * weights[k]
    elif optimizationType[k] == 1:
        for i in range(alternatives):
            for j in range(alternatives):
                if i!=j and decisionMatrix[j][k] >= decisionMatrix[i][k]:
                    if decisionMatrix[j][k] - decisionMatrix[i][k] > preferenceThreshold[k]:
                        evaluationTable[i][j] = evaluationTable[i][j] + 1.0 * weights[k]
                    elif decisionMatrix[j][k] - decisionMatrix[i][k] <= indifferenceThreshold[k]:
                        evaluationTable[i][j] = evaluationTable[i][j] + 0.0 * weights[k]
                    else:
                        evaluationTable[i][j] = evaluationTable[i][j] + 0.5 * weights[k]


def linearPreferenceAndIndifferenceCriterion(evaluationTable, k, alternatives, decisionMatrix, indifferenceThreshold, preferenceThreshold, weights, optimizationType):
    if optimizationType[k] == 0:
        for i in range(alternatives):
            for j in range(alternatives):
                if i!=j and decisionMatrix[i][k] >= decisionMatrix[j][k]:
                    if decisionMatrix[i][k] - decisionMatrix[j][k] > preferenceThreshold[k]:
                        evaluationTable[i][j] = evaluationTable[i][j] + 1.0 * weights[k]
                    elif decisionMatrix[i][k] - decisionMatrix[j][k] > indifferenceThreshold[k]:
                        evaluationTable[i][j] = evaluationTable[i][j] + ((decisionMatrix[i][k] - decisionMatrix [j][k] - indifferenceThreshold[k])*1.0 / (preferenceThreshold[k]-indifferenceThreshold[k])) * weights[k]
                    else:
                        evaluationTable[i][j] = evaluationTable[i][j] + 0.0 * weights[k]
    elif optimizationType[k] == 1:
        for i in range(alternatives):
            for j in range(alternatives):
                if i!=j and decisionMatrix[j][k] >= decisionMatrix[i][k]:
                    if decisionMatrix[j][k] - decisionMatrix[i][k] > preferenceThreshold[k]:
                        evaluationTable[i][j] = evaluationTable[i][j] + 1.0 * weights[k]
                    elif decisionMatrix[j][k] - decisionMatrix[i][k] > indifferenceThreshold[k]:
                        evaluationTable[i][j] = evaluationTable[i][j] + ((decisionMatrix[j][k] - decisionMatrix [i][k] - indifferenceThreshold[k])*1.0 / (preferenceThreshold[k]-indifferenceThreshold[k])) * weights[k]
                    else:
                        evaluationTable[i][j] = evaluationTable[i][j] + 0.0 * weights[k]


#Read Input

criteria = 3
alternatives = 4

optimizationType = [0, 0, 0]
criterion = [5, 5, 5]
preferenceThreshold = [10, 2, 15]
indifferenceThreshold = [5, 1, 5]
weights = [0.4, 0.1, 0.5]
alternativeName = ['Security1', 'Security2', 'Security3', 'Security4']
decisionMatrix = [[20, 5, 20], [15, 3, 5], [12, 2, 10], [10, 4, 35]]

evaluationTable = [[0.0 for i in range(alternatives)] for y in range(alternatives)]

for k in range(criteria):
    if criterion[k] == 1:
        usualCriterion(evaluationTable, k, alternatives, decisionMatrix, indifferenceThreshold, preferenceThreshold, weights, optimizationType)
    elif criterion[k] == 2:
        quasiCriterion(evaluationTable, k, alternatives, decisionMatrix, indifferenceThreshold, preferenceThreshold, weights, optimizationType)
    elif criterion[k] == 3:
        linearPreferenceCriterion(evaluationTable, k, alternatives, decisionMatrix, indifferenceThreshold, preferenceThreshold, weights, optimizationType)
    elif criterion[k] == 4:
        levelCriterion(evaluationTable, k, alternatives, decisionMatrix, indifferenceThreshold, preferenceThreshold, weights, optimizationType)
    elif criterion[k] == 5:
        linearPreferenceAndIndifferenceCriterion(evaluationTable, k, alternatives, decisionMatrix, indifferenceThreshold, preferenceThreshold, weights, optimizationType)


for i in range(alternatives):
    for j in range(alternatives):
        evaluationTable[i][j] = round(evaluationTable[i][j],2)

print(evaluationTable)

sumOfLines = np.sum(evaluationTable, axis=1)
sumOfColumns = np.sum(evaluationTable, axis=0)

phiPlus = sumOfLines*1.0 / (alternatives - 1)
phiMinus = sumOfColumns*1.0 / (alternatives - 1)
phi = phiPlus - phiMinus

print("Positive Flow")
print(phiPlus)
print("Negative Flow")
print(phiMinus)
print("Net Flow")
print(phi)

i = np.arange(alternatives)
plt.bar(alternativeName, phi, color = 'b', edgecolor = 'black')
plt.xlabel('Alternatives')
plt.ylabel('Net Flow')
plt.title('PROMETHEE Method', fontsize=16)
ax = plt.gca()
ax.set_facecolor('red')
plt.grid()

result = [[0 for x in range(2)] for y in range(alternatives)]
for i in range(alternatives):
    result[i][0] = alternativeName[i]
    result[i][1] = round(phi[i],3)

result = sorted(result, key=lambda tup: tup[1], reverse=True)
print(result)
plt.show()