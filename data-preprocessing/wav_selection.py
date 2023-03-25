import csv
import os
import shutil
import re

# CSV file path and name
csv_file = './note_audio.csv'

# Input and output directories
# interior, exterior, help, robbery, sexual, theft, violence
target = 'violence'
# num = 5
# input_dir = f'/Users/yoohajun/Desktop/grad_audio/{target}_{num}'
input_dir = f'/Users/yoohajun/Desktop/grad_audio/source/{target}/train'
output_dir = f'/Users/yoohajun/Desktop/grad_audio/diffusion/train'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Regular expression to match the number
pattern = r'\d+'

# Read the CSV file
file_mapping = []
with open(csv_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        index, filename = row
        # Remove the label suffix from the filename
        # filename = p.sub('', filename)
        matches = re.findall(pattern, filename)
        number = '-'.join(matches)
        # file_mapping[number] = filename
        print(matches)
        print(number)
        file_mapping.append(f'{number}')
        # print(file_mapping[-1])

print('........'*20)

for file_name in os.listdir(input_dir):
    if file_name.endswith('.wav'):
        # Check if the file exists in the file mapping list

        matches_wav = re.findall(pattern, file_name)
        number_wav = '-'.join(matches_wav)

        if number_wav in file_mapping:
            print('yes = ', file_name)
            try:
                shutil.move(os.path.join(input_dir, file_name), os.path.join(output_dir, file_name))
                print(f'Moved {file_name} to {output_dir}')
            except Exception as e:
                print(e)
                print(f'Failed to move {file_name} to {output_dir}')

print('moved total:', len(os.listdir(output_dir)))


