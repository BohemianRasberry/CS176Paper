import csv

column_name = 'diagnosisClass'  # Replace with your actual column name
filename = 'Mesothelioma_clean_data.csv'
count_of_ones = 0
count_of_twos = 0

with open(filename, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row[column_name] == '1':  # Assuming the ones are stored as strings
            count_of_ones += 1
        else:
            count_of_twos += 1

print(f"Count of 1s in column '{column_name}': {count_of_ones}")
print(f"Count of 2s in column '{column_name}': {count_of_twos}")
