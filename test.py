import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

pylab.rcParams['figure.figsize'] = 20, 12


def imshow(img, caption):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)


if __name__ == "__main__":
    caption = 'bobble heads on top of the shelf'
    img = np.load("result.npy")
    imshow(img, caption)
    plt.show()
