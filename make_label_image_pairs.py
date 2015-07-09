import csv
import os

 
def load_label_data():

    with open('trainLabels.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')

        filenames = []
        bins = [[],[],[],[],[]]
        for row in reader:
            filenames.append(row[0])
            if row[1] is '0':
                bins[0].append(row[0])
            elif row[1] is '1':
                bins[1].append(row[0])
            elif row[1] is '2':
                bins[2].append(row[0])
            elif row[1] is '3':
                bins[3].append(row[0])
            elif row[1] is '4':
                bins[4].append(row[0])
    return bins


###################################
if __name__ == '__main__':
    bins = load_label_data()
    counts = [len(x) for x in bins]
    header_flag = True
    binned_names = [['' for i in range(5)] for j in range(max(counts))]

    N = max(counts)
    for i in range(N):
        for j in range(5):
            try:
                binned_names[i][j] = bins[j][i]
            except IndexError:
                continue

    # The format is labels; counts; filenames
    with open('label_image_pairs.csv', 'wb') as f:
        writer = csv.writer(f)
        for i in range(N):
            if header_flag is True:
                writer.writerows([['0', '1', '2', '3', '4'], counts])
                header_flag = False
            writer.writerows([binned_names[i]])
            # print("\rProgress: " + str(float(i)*100/N)),

    print("\nCSV file containing image-label pairs was created.")