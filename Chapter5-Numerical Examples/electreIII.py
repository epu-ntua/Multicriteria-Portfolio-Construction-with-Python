#Filename: electreIII.py
#Description: Implementation of MCDA method ELECTRE III
#Author: Elissaios Sarmas. November 3, 2019.
import csv
import numpy as np
import matplotlib.pyplot as plt
import math

#Section 1: Read Input Data such as Alternatives, Criteria, Weights and the Decision Matrix

criteria = 3
alternatives = 4

optimizationType = [0, 0, 0]
vetoThreshold = [20, 5, 30]
preferenceThreshold = [10, 2, 15]
indifferenceThreshold = [5, 1, 5]
weights = [0.4, 0.1, 0.5]
alternativeName = ['Security1', 'Security2', 'Security3', 'Security4']
decisionMatrix = [[20, 5, 20], [15, 3, 5], [12, 2, 10], [10, 4, 35]]

#Section 2: Calculation of the Agreement Table for Electre ΙΙΙ Method

concordanceTable = [[0 for i in range(alternatives)] for y in range(alternatives)]
for k in range(criteria):
    if optimizationType[k] == 0:
        for i in range(alternatives):
            for j in range(alternatives):
                if i!=j:
                    if decisionMatrix[j][k] - decisionMatrix[i][k] <= indifferenceThreshold[k]:
                        concordanceTable[i][j] = round(concordanceTable[i][j] + 1.0 * weights[k],2)
                    elif decisionMatrix[j][k] - decisionMatrix[i][k] <= preferenceThreshold[k]:
                        concordanceTable[i][j] = round(concordanceTable[i][j] + ((decisionMatrix[i][k] - decisionMatrix [j][k] + preferenceThreshold[k])*1.0 / (preferenceThreshold[k]-indifferenceThreshold[k])) * weights[k],2)
                    else:
                        concordanceTable[i][j] = round(concordanceTable[i][j] + 0.0 * weights[k],2)
    elif optimizationType[k] == 1:
        for i in range(alternatives):
            for j in range(alternatives):
                if i!=j:
                    if decisionMatrix[i][k] - decisionMatrix[j][k] <= indifferenceThreshold[k]:
                        concordanceTable[i][j] = round(concordanceTable[i][j] + 1.0 * weights[k],2)
                    elif decisionMatrix[i][k] - decisionMatrix[j][k] <= preferenceThreshold[k]:
                        concordanceTable[i][j] = round(concordanceTable[i][j] + ((decisionMatrix[j][k] - decisionMatrix [i][k] + preferenceThreshold[k])*1.0 / (preferenceThreshold[k]-indifferenceThreshold[k])) * weights[k],2)
                    else:
                        concordanceTable[i][j] = round(concordanceTable[i][j] + 0.0 * weights[k],2)

print("Agreement Table")
print(concordanceTable)

#Section 3: Calculation of the Disagreement Tables for Electre III Method

sumOfWeights = sum(weights)
discordanceTable = [[[0 for k in range(criteria)] for i in range(alternatives)] for j in range(alternatives)]
for k in range(criteria):
    if optimizationType[k] == 0:
        for i in range(alternatives):
            for j in range(alternatives):
                if i!=j:
                    if decisionMatrix[j][k] - decisionMatrix[i][k] <= preferenceThreshold[k]:
                        discordanceTable[i][j][k] = 0
                    elif decisionMatrix[j][k] - decisionMatrix[i][k] <= vetoThreshold[k]:
                        discordanceTable[i][j][k] = round(((decisionMatrix[j][k] - decisionMatrix [i][k] - preferenceThreshold[k])*1.0 / (vetoThreshold[k]-preferenceThreshold[k])),2)
                    else:
                        discordanceTable[i][j][k] = 1
    elif optimizationType[k] == 1:
        for i in range(alternatives):
            for j in range(alternatives):
                if i!=j:
                    if decisionMatrix[i][k] - decisionMatrix[j][k] <= preferenceThreshold[k]:
                        discordanceTable[i][j][k] = 0
                    elif decisionMatrix[i][k] - decisionMatrix[j][k] <= vetoThreshold[k]:
                        discordanceTable[i][j][k] = round(((decisionMatrix[j][k] - decisionMatrix [i][k] + preferenceThreshold[k])*1.0 / (vetoThreshold[k]-preferenceThreshold[k])),2)
                    else:
                        discordanceTable[i][j][k] = 1

print("Disagreement Table")
print(discordanceTable)

#Section 4: Calculation of reliability indexes

reliabilityTable = [[0 for i in range(alternatives)] for y in range(alternatives)]
for i in range(alternatives):
    for j in range(alternatives):
        if i!=j:
            reliabilityTable[i][j] = concordanceTable[i][j]
            for k in range(criteria):
                if concordanceTable[i][j] < discordanceTable[i][j][k]:
                    reliabilityTable[i][j] = reliabilityTable[i][j] * ((1 - discordanceTable[i][j][k]) / (1 - concordanceTable[i][j]))
        else:
            reliabilityTable[i][j] = 1

print("Reliability Table")
print(reliabilityTable)

#Section 5: Calculation of Dominance Table

d = 0.8
dominanceTable = [[0 for i in range(alternatives)] for y in range(alternatives)]
for i in range(alternatives):
    for j in range(alternatives):
        if i!=j and reliabilityTable[i][j] >= d:
            dominanceTable[i][j] = 1

print("Dominance Table")
print(dominanceTable)

#Section 6: Concordance, Disconcordance and Net Credibility Degrees

phiPlus = [round(sum(x),3) for x in reliabilityTable ]
phiMinus = [round(sum(x),3) for x in zip(*reliabilityTable)]
phi = [round((x1 - x2),3) for (x1, x2) in zip(phiPlus, phiMinus)]

print("Positive Flow")
print(phiPlus)
print("Negative Flow")
print(phiMinus)
print("Net Flow")
print(phi)

plt.bar(alternativeName, phi, color = 'b', edgecolor = 'black')
plt.xlabel('Alternatives')
plt.ylabel('Net Credibility Index')
plt.title('Electre III Method', fontsize=16)
ax = plt.gca()
ax.set_facecolor('red')
plt.grid()

result = [[0 for x in range(2)] for y in range(alternatives)]
for i in range(alternatives):
    result[i][0] = alternativeName[i]
    result[i][1] = phi[i]

result = sorted(result, key=lambda tup: tup[1], reverse=True)

print(result)
plt.show()