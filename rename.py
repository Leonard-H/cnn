import os
import csv

# Define the path to the CSV file and the directory
csv_file = '19/train_subset.csv'
base_dir = '19/train_subset'

# Open the CSV file and process each row
with open(csv_file, 'r') as file:
  reader = csv.DictReader(file)
  for row in reader:
    old_file_path = os.path.join(base_dir, row['File'])
    new_file_path = os.path.join(base_dir, f"{row['subject_id']}.jpg")
    
    # Rename the file if it exists
    if os.path.exists(old_file_path):
      os.rename(old_file_path, new_file_path)
      print(f"Renamed: {old_file_path} -> {new_file_path}")
    else:
      print(f"File not found: {old_file_path}")