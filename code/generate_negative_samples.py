import numpy as np
np.random.seed(1337)
import csv
import random

# Read CSV file and store data in SaveList
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName, 'r'))
    for row in csv_reader:
        SaveList.append(row)
    return

# Save data to a CSV file
def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

# Read source file
OriginalData = []
ReadMyCsv(OriginalData, "interaction.csv")
print(len(OriginalData))

# Preprocessing
LncCircRNA_miRNA = []
for row in OriginalData:
    Pair = [row[0], row[1]]  # row[0] is circRNA, row[1] is miRNA
    LncCircRNA_miRNA.append(Pair)

print('Length of LncCircRNA_miRNA:', len(LncCircRNA_miRNA))
print('Length of OriginalData:', len(OriginalData))

# Build AllmiRNA
AllmiRNA = []
for row in OriginalData:
    if row[1] not in AllmiRNA:
        AllmiRNA.append(row[1])

print('Length of AllmiRNA:', len(AllmiRNA))

# Build AllCircRNA
AllCircRNA = []
for row in OriginalData:
    if row[0] not in AllCircRNA:
        AllCircRNA.append(row[0])

print('Length of AllCircRNA:', len(AllCircRNA))
storFile(AllCircRNA, 'AllCircRNA.csv')

# Select Positive and Negative Samples
PositiveSample = LncCircRNA_miRNA
print('Length of PositiveSample:', len(PositiveSample))

NegativeSample = []
while len(NegativeSample) < len(PositiveSample):
    counterM = random.randint(0, len(AllmiRNA)-1)
    counterC = random.randint(0, len(AllCircRNA)-1)
    CircRNA_miRNA_Pair = [AllCircRNA[counterC], AllmiRNA[counterM]]

    if CircRNA_miRNA_Pair in LncCircRNA_miRNA or CircRNA_miRNA_Pair in NegativeSample:
        continue

    NegativeSample.append(CircRNA_miRNA_Pair)

print('Length of NegativeSample:', len(NegativeSample))
storFile(NegativeSample, 'NegativeSample.csv')
