import warnings
warnings.filterwarnings('ignore')
import csv
import numpy as np

# Read CSV file and store each row in SaveList
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return

# Save data to a CSV file
def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

# Create an empty list data1 and read data from "interaction.csv"
data1 = []
ReadMyCsv(data1, "interaction.csv")
print(len(data1))

# Create an empty list data2 and read data from "NegativeSample.csv"
data2 = []
ReadMyCsv(data2, "NegativeSample.csv")
print(len(data2))

# Create an empty list data_final and combine data1 and data2 into a new 2D array
data_final = []
data_final = np.vstack((data1, data2))

# Print the shape and content of data_final
print(data_final.shape)
print(data_final)

# Save data_final as "combined_samples.csv"
storFile(data_final, 'positive_negative_samples.csv')
