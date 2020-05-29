#Filename: electreIII.py
#Description: Implementation of MCDA method ELECTRE III
#Author: Elissaios Sarmas. November 3, 2019.
import csv
import numpy as np
import matplotlib.pyplot as plt
import math

inputname = "elec3.csv"

file =  open(inputname, "rt")
list1 = list(csv.reader(file))

#Section 1: Read Input Data such as Alternatives, Criteria, Weights and the Decision Matrix

criteria = int(list1[0][1])
alternatives = int(list1[1][1])
weights = [0 for y in range(criteria)]
optimizationType = [0 for y in range(criteria)]
preferenceThreshold = [0 for y in range(criteria)]
indifferenceThreshold = [0 for y in range(criteria)]
vetoThreshold = [0 for y in range(criteria)]
for i in range(criteria):
    optimizationType[i] = int(list1[2][i+1])
    weights[i] = float(list1[3][i+1])
    vetoThreshold[i] = float(list1[4][i+1])
    preferenceThreshold[i] = float(list1[5][i+1])
    indifferenceThreshold[i] = float(list1[6][i+1])
decisionMatrix = [[0 for y in range(criteria)] for x in range(alternatives)]
companyName = ["" for i in range(alternatives)]
for i in range(alternatives):
    companyName[i] = list1[i+16][0]
    for j in range(criteria):
        decisionMatrix[i][j] = float(list1[i+16][j+1])

#Section 2: Calculation of the Agreement Table for Electre ΙΙΙ Method

sumOfWeights = sum(weights)
agreementTable = [[0 for i in range(alternatives)] for y in range(alternatives)]
for k in range(criteria):
    if optimizationType[k] == 0:
        for i in range(alternatives):
            for j in range(alternatives):
                if i!=j:
                    if decisionMatrix[j][k] - decisionMatrix[i][k] <= indifferenceThreshold[k]:
                        agreementTable[i][j] = round(agreementTable[i][j] + 1.0 * weights[k],2)
                    elif decisionMatrix[j][k] - decisionMatrix[i][k] <= preferenceThreshold[k]:
                        agreementTable[i][j] = round(agreementTable[i][j] + ((decisionMatrix[i][k] - decisionMatrix [j][k] + preferenceThreshold[k])*1.0 / (preferenceThreshold[k]-indifferenceThreshold[k])) * weights[k],2)
                    else:
                        agreementTable[i][j] = round(agreementTable[i][j] + 0.0 * weights[k],2)
    elif optimizationType[k] == 1:
        for i in range(alternatives):
            for j in range(alternatives):
                if i!=j:
                    if decisionMatrix[i][k] - decisionMatrix[j][k] <= indifferenceThreshold[k]:
                        agreementTable[i][j] = round(agreementTable[i][j] + 1.0 * weights[k],2)
                    elif decisionMatrix[i][k] - decisionMatrix[j][k] <= preferenceThreshold[k]:
                        agreementTable[i][j] = round(agreementTable[i][j] + ((decisionMatrix[j][k] - decisionMatrix [i][k] + preferenceThreshold[k])*1.0 / (preferenceThreshold[k]-indifferenceThreshold[k])) * weights[k],2)
                    else:
                        agreementTable[i][j] = round(agreementTable[i][j] + 0.0 * weights[k],2)

#Section 3: Calculation of the Disagreement Tables for Electre III Method

sumOfWeights = sum(weights)
disagreementTable = [[[0 for k in range(criteria)] for i in range(alternatives)] for j in range(alternatives)]
for k in range(criteria):
    if optimizationType[k] == 0:
        for i in range(alternatives):
            for j in range(alternatives):
                if i!=j:
                    if decisionMatrix[j][k] - decisionMatrix[i][k] <= preferenceThreshold[k]:
                        disagreementTable[i][j][k] = 0
                    elif decisionMatrix[j][k] - decisionMatrix[i][k] <= vetoThreshold[k]:
                        disagreementTable[i][j][k] = round(((decisionMatrix[j][k] - decisionMatrix [i][k] - preferenceThreshold[k])*1.0 / (vetoThreshold[k]-preferenceThreshold[k])),2)
                    else:
                        disagreementTable[i][j][k] = 1
    elif optimizationType[k] == 1:
        for i in range(alternatives):
            for j in range(alternatives):
                if i!=j:
                    if decisionMatrix[i][k] - decisionMatrix[j][k] <= indifferenceThreshold[k]:
                        disagreementTable[i][j][k] = 0
                    elif decisionMatrix[i][k] - decisionMatrix[j][k] <= vetoThreshold[k]:
                        disagreementTable[i][j][k] = round(((decisionMatrix[j][k] - decisionMatrix [i][k] + preferenceThreshold[k])*1.0 / (vetoThreshold[k]-preferenceThreshold[k])),2)
                    else:
                        disagreementTable[i][j][k] = 1


#Section 4: Calculation of reliability indexes

reliabilityTable = [[0 for i in range(alternatives)] for y in range(alternatives)]
for i in range(alternatives):
    for j in range(alternatives):
        if i!=j:
            reliabilityTable[i][j] = agreementTable[i][j]
            for k in range(criteria):
                if agreementTable[i][j] < disagreementTable[i][j][k]:
                    reliabilityTable[i][j] = reliabilityTable[i][j] * ((1 - disagreementTable[i][j][k]) / (1 - agreementTable[i][j]))
        else:
            reliabilityTable[i][j] = 1

#Section 5: Calculation of Dominance Table

d = 0.8
dominanceTable = [[0 for i in range(alternatives)] for y in range(alternatives)]
for i in range(alternatives):
    for j in range(alternatives):
        if i!=j and reliabilityTable[i][j] >= d:
            dominanceTable[i][j] = 1

#Section 6: Concordance, Disconcordance and Net Credibility Degrees

phiPlus = [round(sum(x),3) for x in reliabilityTable ]
phiMinus = [round(sum(x),3) for x in zip(*reliabilityTable)]
phi = [round((x1 - x2),3) for (x1, x2) in zip(phiPlus, phiMinus)]

print(phi)

plt.bar(companyName, phi, color = 'b', edgecolor = 'black')
plt.xlabel('Alternatives')
plt.ylabel('Net Credibility Index')
plt.title('Electre III Method', fontsize=16)
ax = plt.gca()
ax.set_facecolor('red')
plt.grid()

result = [[0 for x in range(2)] for y in range(alternatives)]
for i in range(alternatives):
    result[i][0] = companyName[i]
    result[i][1] = phi[i]

result = sorted(result, key=lambda tup: tup[1], reverse=True)

print(result)
plt.show()