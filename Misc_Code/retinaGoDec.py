# We'll need the csv module to read the file
import csv
import os
from random import randint
from PIL import Image
import numpy as np
from numpy.linalg import norm
from scipy.linalg import qr
from scipy import misc
from glob import glob
import Image, ImageOps                                                                        

 
def load_data():

    with open('./labels/trainLabels.csv', 'r') as f:
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
    img = Image.open(os.path.abspath(os.path.join(os.path.dirname(__file__), 'train', bins[level][num] + '.jpeg')))
    print(img.size)
    img.show()

def get_max_image_sizes():
    names = [x for x in glob('train/*')]
    sizes = [[],[]]
    for name in names:
        img = Image.open(os.path.abspath(os.path.join(os.path.dirname(__file__), name)))
        a, b = img.size
        sizes[0].append(a)
        sizes[1].append(b)
    print(sizes[0])
    print(sizes[1])
    return min(sizes[0]), min(sizes[1]), max(sizes[0]), max(sizes[1])


def wthresh(a, thresh):
    #Soft wavelet threshold
    res = np.abs(a) - thresh
    return np.sign(a) * ((res > 0) * res)

#Default threshold of .03 is assumed to be for input in the range 0-1...
#original matlab had 8 out of 255, which is about .03 scaled to 0-1 range
def go_dec(X, thresh=.03, rank=10, power=0, tol=1e-3,
           max_iter=100, random_seed=0, verbose=True):
    m, n = X.shape
    if m < n:
        X = X.T
    m, n = X.shape
    L = X
    S = np.zeros(L.shape)
    itr = 0
    random_state = np.random.RandomState(random_seed)    
    while True:
        Y2 = random_state.randn(n, rank)
        for i in range(power+1):
            Y1 = np.dot(L, Y2)
            Y2 = np.dot(L.T, Y1);
        Q, R = qr(Y2, mode='economic')
        L_new = np.dot(np.dot(L, Q), Q.T)
        T = L - L_new + S
        L = L_new
        S = wthresh(T, thresh)
        T -= S
        err = norm(T.ravel(), 2)
        if (err < tol) or (itr >= max_iter):
            break 
        L += T
        itr += 1
    #Is this even useful in soft GoDec? May be a display issue...
    G = X - L - S
    if m < n:
        L = L.T
        S = S.T
        G = G.T
    if verbose:
        print "Finished at iteration %d" % (itr)
    return L, S, G


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray / 255.


def get(img, y0, y1, x0, x1, mode="nearest"):
    xs, ys = np.mgrid[y0:y1, x0:x1]
    height, width, dim = img.shape

    if mode == "nearest":
        xs = np.clip(xs, 0, height-1)
        ys = np.clip(ys, 0, width-1)

    elif mode == "wrap":
        xs = xs % height
        ys = ys % width

    elif mode == "reflect":
        maxh = height-1
        maxw = width-1

        # An unobvious way of performing reflecting modulo
        # You should comment this
        xs = np.absolute((xs + maxh) % (2 * maxh) - maxh)
        ys = np.absolute((ys + maxw) % (2 * maxw) - maxw)

    elif mode == "constant":
        output = np.empty((y1-y0, x1-x0, 3))
        output.fill(0) # WHAT THE CONSTANT IS

        # LOADS of bounds checks and restrictions
        # You should comment this
        target_section = output[max(0, -y0):min(y1-y0, -y0+height), max(0, -x0):min(x1-x0, -x0+width), :]
        new_fill = img[max(0, y0):min(height, y1), max(0, x0):min(width, x1), :]

        # Crop both the sections, so that they're both the size of the smallest
        # Use more lines; I'm too lazy right now
        target_section[:new_fill.shape[0], :new_fill.shape[1], :] = new_fill[:target_section.shape[0], :target_section.shape[1], :]

        return output

    else:
        raise NotImplementedError("Unknown mode")

    return img[xs, ys]

###################################
if __name__ == '__main__':
    bins = load_data()
    # show_rand_image(bins, 0)
    names = [name for name in glob('train/*')]
    # num = len(names)
    num = 1000
    # Training min, min, max, max image dims: (400, 289, 5184, 3456)
    # m1, m2, d1, d2 = get_max_image_sizes()
    d1, d2 = 3456, 5184
    d1, d2 = 36, 36
    print(d1, d2, num)

    # X = np.zeros((d1, d2, num))
    X = np.zeros((36, 36, num))
    for n, i in enumerate(names):
        if n < num:
            img = misc.imread(i).astype(np.double)
            img = misc.imresize(img, (d1, d2), interp='bilinear')
            # img = get(misc.imread(i).astype(np.double), 0, 3456, 0, 5184, mode="constant")
            X[:, :, n] = rgb2gray(img) / 255.
    # X = X.reshape(d1*d2, num)
    sz = 100
    print(X.shape)
    
    L, S, G = go_dec(X[:, :sz])
    L = L.reshape(d1, d2, sz) * 255.
    S = S.reshape(d1, d2, sz) * 255.
    G = G.reshape(d1, d2, sz) * 255.

    savemat("./GoDec_background_subtraction.mat", {"1": L, "2": S, "3": G, })
    print("GoDec complete")
    L.show()
    S.show()
    G.show()

