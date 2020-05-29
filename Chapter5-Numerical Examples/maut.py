#Filename: maut.py
#Description: Implementation of MCDA method MAUT
#Author: Elissaios Sarmas. November 3, 2019.
import csv
import numpy as np
import matplotlib.pyplot as plt
import math


criteria = 3
alternatives = 4

optimizationType = [0, 0, 0]
weights = [0.4, 0.1, 0.5]
alternativeName = ['Security1', 'Security2', 'Security3', 'Security4']
decisionMatrix = [[20, 5, 20], [15, 3, 5], [12, 2, 10], [10, 4, 35]]

maxValue = np.max(decisionMatrix, axis = 0)
minValue = np.min(decisionMatrix, axis = 0)

normalisedMatrix = [[0 for y in range(criteria)] for x in range(alternatives)]
for i in range(alternatives):
    for j in range(criteria):
        if optimizationType[j] == 0:
            normalisedMatrix[i][j] = (decisionMatrix[i][j] - minValue[j])*1.0 / (maxValue[j] - minValue[j])
        elif optimizationType[j] == 1:
            normalisedMatrix[i][j] = (maxValue[j] - decisionMatrix[i][j])*1.0 / (maxValue[j] - minValue[j])

print(normalisedMatrix)

utilityScore = [0 for x in range(alternatives)]
utilityScorePer = [0 for x in range(alternatives)]

for i in range(alternatives):
    tempSum = 0
    for j in range(criteria):
        tempSum += normalisedMatrix[i][j] * weights[j]
    utilityScore[i] = tempSum
    utilityScorePer[i] = round(round(tempSum,4) * 100,2)

print(utilityScore)

plt.bar(alternativeName, utilityScore, color = 'b', edgecolor = 'black')
plt.xlabel('Alternatives')
plt.ylabel('Utility Score')
plt.title('MAUT Method', fontsize=16)
ax = plt.gca()
ax.set_facecolor('red')
plt.grid()

result = [[0 for x in range(2)] for y in range(alternatives)]
for i in range(alternatives):
    result[i][0] = alternativeName[i]
    result[i][1] = utilityScore[i]

result = sorted(result, key=lambda tup: tup[1], reverse=True)

print(result)
plt.show()