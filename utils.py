import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List
import math
import itertools
import matplotlib.axes as mplt_axes

def read_image(fpath:str)->np.ndarray:
    """Takes in a file path to an image (ex. jpeg, png) and returns a numpy array

    Args:
        fpath (str): path to image file
    """
    img = Image.open(fpath)
    return np.asarray(img)


def plot_images(images: List[np.ndarray], max_cols=4, fig_scale=10, title=""):
    """_summary_

    Args:
        images (List[np.ndarray]): _description_
        max_cols (int, optional): _description_. Defaults to 4.
        fig_scale (int, optional): _description_. Defaults to 10.
        title (str, optional): _description_. Defaults to "".
    """
    n = len(images)
    if n < 1: return

    if isinstance(images[0], list) or isinstance(images[0], tuple):
        temp = []
        for tup in images:
            if isinstance(tup, np.ndarray):
                temp.append(tup)
                continue
            for img in tup:
                temp.append(img)
        images, temp = temp, None

        n = len(images)

    rows, cols = math.ceil(n/max_cols), max_cols

    if n < max_cols:
        rows, cols = 1, n
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_scale*cols, fig_scale*(max(rows-1, 1))))
    
    if rows == 1:
        if isinstance(axes, mplt_axes.SubplotBase):
            axes.imshow(images[0])
        else:
            for i in range(len(axes)):
                if i > len(images)-1:
                    fig.delaxes(axes[i])
                    continue
                axes[i].imshow(images[i])
    else:
        i = 0
        for row in range(len(axes)):
            for col in range(len(axes[row])):
                if i > len(images) - 1:
                    fig.delaxes(axes[row, col])
                    continue
                axes[row, col].imshow(images[i])
                i+=1
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(pad= max(1.08, rows))
    plt.show()
    
    

def show_image_mask_pair(img:np.ndarray, mask:np.ndarray, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(16,6))
    for ax, im in zip(axes, [img, mask]):
        ax.imshow(im)
    fig.suptitle(title, fontsize=16)
    plt.show()


if __name__ =="__main__":
    pass