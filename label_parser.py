import os
import csv


path_ = os.path.join('Data','Разметка','Labeling_ships_clear')
path_accuracy = os.path.join(path_,'labels')



labels_ = []
txt_files=[]

for file in os.listdir(path_accuracy):
    if file.endswith('.txt'):
        txt_files.append(file)
        with open(path_accuracy+'/'+file) as csvfile:
            reader=csv.reader(csvfile, delimiter = ' ')
            rows= []
            for row in reader:
                rows.append([float(value) for value in row])
            labels_.append(rows)

print(f'{labels_[0]} - first label')
print(f'{len(labels_)} labels found')
