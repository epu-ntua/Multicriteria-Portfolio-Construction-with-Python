#Filename: maut.py
#Description: Implementation of MCDA method MAUT
#Author: Elissaios Sarmas. November 3, 2019.
import csv
import numpy as np
import matplotlib.pyplot as plt
import math

inputname = "topsis.csv"

file =  open(inputname, "rt")
list1 = list(csv.reader(file)) 

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

maxValue = np.max(decisionMatrix, axis = 0)
minValue = np.min(decisionMatrix, axis = 0)

normalisedMatrix = [[0 for y in range(criteria)] for x in range(alternatives)]
for i in range(alternatives):
    for j in range(criteria):
        if optimizationType[j] == 0:
            normalisedMatrix[i][j] = (decisionMatrix[i][j] - minValue[j])*1.0 / (maxValue[j] - minValue[j])
        elif optimizationType[j] == 1:
            normalisedMatrix[i][j] = (maxValue[j] - decisionMatrix[i][j])*1.0 / (maxValue[j] - minValue[j])

utilityScore = [0 for x in range(alternatives)]
utilityScorePer = [0 for x in range(alternatives)]

for i in range(alternatives):
    tempSum = 0
    for j in range(criteria):
        tempSum += normalisedMatrix[i][j] * weights[j]
    utilityScore[i] = tempSum
    utilityScorePer[i] = round(round(tempSum,4) * 100,2)

print(utilityScore)

plt.bar(companyName, utilityScore, color = 'b', edgecolor = 'black')
plt.xlabel('Alternatives')
plt.ylabel('Utility Score')
plt.title('MAUT Method', fontsize=16)
ax = plt.gca()
ax.set_facecolor('red')
plt.grid()

result = [[0 for x in range(2)] for y in range(alternatives)]
for i in range(alternatives):
    result[i][0] = companyName[i]
    result[i][1] = utilityScore[i]

result = sorted(result, key=lambda tup: tup[1], reverse=True)

print(result)
plt.show()