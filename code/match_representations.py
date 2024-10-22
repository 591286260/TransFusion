import csv

# Read data from a CSV file and store it in a dictionary, using id as the key and feature representation as the value
def read_feature_file(file_name):
    feature_dict = {}
    with open(file_name, 'r', encoding='gbk') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            id_value = row[0]
            feature_value = row[1:]
            feature_dict[id_value] = feature_value
    return feature_dict

# Replace ids with feature representations and store them in a new CSV file
def replace_id_with_feature(input_file_name, feature_dict, output_file_name):
    with open(input_file_name, 'r', encoding='gbk') as csvfile:
        with open(output_file_name, 'w', newline='', encoding='gbk') as output_csvfile:
            csv_writer = csv.writer(output_csvfile)

            for row in csv.reader(csvfile):
                id1, id2 = row
                feature1 = feature_dict.get(id1, [])
                feature2 = feature_dict.get(id2, [])
                combined_feature = feature1 + feature2
                csv_writer.writerow(combined_feature)

if __name__ == "__main__":
    # Read feature representations from the second CSV file and store them in a dictionary
    feature_file_name = "connect.csv"
    feature_dict = read_feature_file(feature_file_name)

    # Replace ids in the first CSV file with feature representations and store the results in the third CSV file
    input_file_name = "positive_negative_samples.csv"
    output_file_name = "SampleFeature.csv"
    replace_id_with_feature(input_file_name, feature_dict, output_file_name)
