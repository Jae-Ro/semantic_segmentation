from PIL import Image
import matplotlib.pyplot as plt
from typing import List
import math
import matplotlib.axes as mplt_axes
import numpy as np
import pandas as pd
import torchvision

def read_image(fpath:str)->np.ndarray:
    """Takes in a file path to an image (ex. jpeg, png) and returns a numpy array

    Args:
        fpath (str): path to image file
    """
    img = Image.open(fpath)
    return np.asarray(img)

        

def show_image_mask_pair(img:np.ndarray, mask:np.ndarray, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(16,6))
    for ax, im in zip(axes, [img, mask]):
        ax.imshow(im)
    fig.suptitle(title, fontsize=16)
    plt.show()


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
    

    
def imshow(inp, size, title=None):
    '''
        Shows images
        Parameters:
            inp: images
            title: A title for image
    '''
    inp = inp.numpy().transpose((1, 2, 0))
    plt.figure(figsize=size)
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def show_databatch(inputs, size=(8,8), batch=16):
    out = torchvision.utils.make_grid(inputs[:batch])
    imshow(out, size)


def get_color_map(df):
    '''
    Returns the reversed String.
    Parameters:
        df: A Dataframe with rgb values with class maps.
    Returns:
        code2id: A dictionary with color as keys and class id as values.   
        id2code: A dictionary with class id as keys and color as values.
        name2id: A dictionary with class name as keys and class id as values.
        id2name: A dictionary with class id as keys and class name as values.
    '''
    cls = df
    color_code = [tuple(cls.drop("name",axis=1).loc[idx]) for idx in range(len(cls.name))]
    code2id = {v: k for k, v in enumerate(list(color_code))}
    id2code = {k: v for k, v in enumerate(list(color_code))}

    color_name = [cls['name'][idx] for idx in range(len(cls.name))]
    name2id = {v: k for k, v in enumerate(list(color_name))}
    id2name = {k: v for k, v in enumerate(list(color_name))}  
    return code2id, id2code, name2id, id2name

def rgb_to_mask(img, color_map):
    ''' 
        Converts a RGB image mask of shape to Binary Mask of shape [batch_size, classes, h, w]
        Parameters:
            img: A RGB img mask
            color_map: Dictionary representing color mappings
        returns:
            out: A Binary Mask of shape [batch_size, classes, h, w]
    '''
    num_classes = len(color_map)
    shape = img.shape[:2]+(num_classes,)
    out = np.zeros(shape, dtype=np.float64)
    for i, cls in enumerate(color_map):
        out[:,:,i] = np.all(np.array(img).reshape( (-1,3) ) == color_map[i], axis=1).reshape(shape[:2])
    return out.transpose(2,0,1)

def mask_to_rgb(mask, color_map):
    ''' 
        Converts a Binary Mask of shape to RGB image mask of shape [batch_size, h, w, 3]
        Parameters:
            img: A Binary mask
            color_map: Dictionary representing color mappings
        returns:
            out: A RGB mask of shape [batch_size, h, w, 3]
    '''
    single_layer = np.argmax(mask, axis=1)
    output = np.zeros((mask.shape[0],mask.shape[2],mask.shape[3],3))
    for k in color_map.keys():
        output[single_layer==k] = color_map[k]
    return np.uint8(output)

def plot_curves(df_stats):
    plt.figure(figsize=(8, 6))
    for c in ['train_loss', 'val_loss']:
        plt.plot(df_stats[c], label=c)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Training and Validation Loss Curve')
    plt.show()

def visualize(imgs, title='Original', cols=6, rows=1, plot_size=(16, 16), change_dim=False):
    fig=plt.figure(figsize=plot_size)
    columns = cols
    rows = rows
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.title(title+str(i))
        if change_dim: plt.imshow(imgs.transpose(0,2,3,1)[i])
        else: plt.imshow(imgs[i])
    plt.show()



if __name__ =="__main__":
    pass