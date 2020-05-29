#Filename: prometheeII.py
#Description: Implementation of MCDA method PROMETHEE II
#Author: Elissaios Sarmas. November 3, 2019.

import csv
import numpy as np
import matplotlib.pyplot as plt
import math

inputname = "promethee.csv"

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

file =  open(inputname, "rt")
list1 = list(csv.reader(file))

#Read Input

criteria = int(list1[0][1])
alternatives = int(list1[1][1])
weights = [0 for y in range(criteria)]
optimizationType = [0 for y in range(criteria)]
criterion = [0 for y in range(criteria)]
preferenceThreshold = [0 for y in range(criteria)]
indifferenceThreshold = [0 for y in range(criteria)]
for i in range(criteria):
    optimizationType[i] = int(list1[2][i+1])
    weights[i] = float(list1[3][i+1])
    criterion[i] = int(list1[7][i+1])
    preferenceThreshold[i] = float(list1[5][i+1])
    indifferenceThreshold[i] = float(list1[6][i+1])
decisionMatrix = [[0 for y in range(criteria)] for x in range(alternatives)]
companyName = ["" for i in range(alternatives)]
for i in range(alternatives):
    companyName[i] = list1[i+16][0]
    for j in range(criteria):
        decisionMatrix[i][j] = float(list1[i+16][j+1])

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

sumOfLines = np.sum(evaluationTable, axis=1)
sumOfColumns = np.sum(evaluationTable, axis=0)

phiPlus = sumOfLines*1.0 / (alternatives - 1)
phiMinus = sumOfColumns*1.0 / (alternatives - 1)
phi = phiPlus - phiMinus

print(phi)



i = np.arange(alternatives)
plt.bar(i+1, phi, color = 'b', edgecolor = 'black')
plt.xlabel('Alternatives')
plt.ylabel('Net Flow')
plt.title('PROMETHEE Method', fontsize=16)
ax = plt.gca()
ax.set_facecolor('red')
plt.grid()

result = [[0 for x in range(2)] for y in range(alternatives)]
for i in range(alternatives):
    result[i][0] = companyName[i]
    result[i][1] = round(phi[i],2)

result = sorted(result, key=lambda tup: tup[1], reverse=True)
print(result)
plt.show()
