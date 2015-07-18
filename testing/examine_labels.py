import csv
import os
from random import randint
import numpy as np
from PIL import Image
# import Image

 
def load_data():
    with open('./labels/trainLabels.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')

        filenames = []
        levels = []
        bins = [[], [], [], [], []]
        for row in reader:
            levels.append(row[1])
            filenames.append(row[0])
            if levels[-1] is '0':
                bins[0].append(row[0])
            elif levels[-1] is '1':
                bins[1].append(row[0])
            elif levels[-1] is '2':
                bins[2].append(row[0])
            elif levels[-1] is '3':
                bins[3].append(row[0])
            elif levels[-1] is '4':
                bins[4].append(row[0])

    occurences = [len(x) for x in bins]
    return bins, np.asarray(occurences, dtype=float)


def show_rand_image(bins, level):
    if level is None or level == []:
        level = randint(0, 4)
    num = randint(0, len(bins[level]))
    print(level, bins[level][num])
    # img = Image.open('./train/' + bins[level][num] + '.jpg')
    # os.path.abspath(os.path.join(os.path.dirname(__file__)
    print(os.path.join('/', 'home', 'dick', 'Documents', 'Kaggle', 'train', bins[level][num] + '_thumb.jpeg'))
    img = Image.open(os.path.join('/', 'home', 'dick', 'Documents', 'Kaggle', 'train', bins[level][num] + '_thumb.jpeg'))
    print(img.size)
    img2 = img.thumbnail((250, 250), Image.ANTIALIAS)
    img.show()

def make_cdf(pdf):
    cdf = []
    for i in range(len(pdf)):
        cdf.append(sum(pdf[:i]))
    return cdf


def sample_labels(p):
    rand_val = np.random.uniform(low=0.0, high=np.nextafter(1, 2), size=1)
    for n in range(len(p)):
        if rand_val <= p[n]:
            return n-1, rand_val
    return len(p)-1, rand_val


###################################
if __name__ == '__main__':
    bins, counts = load_data()

    count_pmf = counts/np.sum(counts)
    cdf = make_cdf(count_pmf)
    label_val, rand_val = sample_labels(cdf)
    print(label_val, rand_val)

    # show_rand_image(bins, 0)
    # show_rand_image(bins, 1)
    # show_rand_image(bins, 2)
    # show_rand_image(bins, 3)
    # show_rand_image(bins, 4)

    imageFile = Image.open("/home/dick/Documents/Kaggle/sample/10_left.jpeg")

    imageFile.thumbnail((512, 512), Image.ANTIALIAS)
    # imageFile.show()

    test_file_names = os.listdir('/media/dick/Storage/test')
    N = len(test_file_names)

    # Make completely random test set labels for first submission
    with open('./labels/firstSubmission.csv', 'wb') as f:
        writer = csv.writer(f)
        for i in range(N):
            label_val = sample_labels(cdf)
            if i == 0:
                writer.writerows([['image', 'level']])
            writer.writerows([[test_file_names[i][:-5], label_val[0]]])

    print("\nCSV file containing image-label pairs was created.")