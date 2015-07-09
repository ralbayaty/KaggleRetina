# We'll need the csv module to read the file
import csv
import os
from random import randint
from PIL import Image
import Image
import ImageTk
import Tkinter                                                                          


def new():
    wind = Tkinter.Toplevel()
    wind.geometry('600x600')               # This not work, why? 
    imageFile2 = Image.open("someimage2.jpg")
    image2 = ImageTk.PhotoImage(imageFile2)

    panel2 = Tkinter.Label(wind , image=image2)
    panel2.place(relx=0.0, rely=0.0)
    wind.mainloop()
 
def load_data():

    with open('trainLabels.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')

        filenames = []
        levels = []
        bins = [[],[],[],[],[]]
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
        
    occurences = []
    occurences.append(levels.count('0'))
    occurences.append(levels.count('1'))
    occurences.append(levels.count('2'))
    occurences.append(levels.count('3'))
    occurences.append(levels.count('4'))
    # print(zip(range(0,5), occurences))
    # print(min(occurences), max(occurences))
    # print(bins)
    # print(zip(filenames, levels))

    return bins


def show_rand_image(bins, level):
    if level is []:
        level = randint(0,4)
    num = randint(0,len(bins[level]))
    print(level, bins[level][num])
    # img = Image.open('./train/' + bins[level][num] + '.jpg')
    img = Image.open(os.path.abspath(os.path.join(os.path.dirname(__file__), 'train', bins[level][num] + '_thumb.jpg')))
    print(img.size)
    img2 = img.thumbnail((250,250),Image.ANTIALIAS)
    img.show()


###################################
if __name__ == '__main__':
    bins = load_data()
    show_rand_image(bins, 0)
    show_rand_image(bins, 1)
    show_rand_image(bins, 2)
    show_rand_image(bins, 3)
    show_rand_image(bins, 4)

    master = Tkinter.Tk()
    master.geometry('600x600')               # This work fine
    imageFile = Image.open("/home/dick/Documents/Kaggle/sample/10_left.jpeg")
    # imageFile = Image.open("/media/dick/External/KaggleRetina/train/17957_right.jpeg")
    
    image1 = ImageTk.PhotoImage(imageFile.thumbnail((250,250), Image.ANTIALIAS))

    panel1 = Tkinter.Label(master , image=image1)
    panel1.place(relx=0.0, rely=0.0)
    B = Tkinter.Button(master, text = 'New image', command = new).pack()
    master.mainloop()