from flask import (
        Flask, 
        render_template,
        send_from_directory,
        Response,
        url_for,
        request
    )
import json
import pandas as pd
import numpy as np
import os
import io
from PIL import Image
import base64
from torchvision import transforms
from datetime import date

app = Flask(__name__)


def get_all_images(data_dir:str):
    """_summary_

    Args:
        data_dir (str): _description_

    Returns:
        _type_: _description_
    """
    img_list = os.listdir(data_dir)
    img_list = [os.path.join(data_dir, img_path) for img_path in img_list]
    return sorted(img_list)


def get_img_mask_overlay(img_path, mask_path, size=(480), opacity=0.3):
    """_summary_

    Args:
        img_path (_type_): _description_
        mask_path (_type_): _description_
        size (tuple, optional): _description_. Defaults to (480).
        opacity (float, optional): _description_. Defaults to 0.3.

    Returns:
        _type_: _description_
    """
    trans = transforms.Compose([
        transforms.Resize(size)
    ])

    img, mask = Image.open(img_path), Image.open(mask_path)
    img, mask = trans(img), trans(mask)
    print(img.size, mask.size)
    
    data = io.BytesIO()
    img.save(data, "PNG")
    encoded_img_data = base64.b64encode(data.getvalue())
    
    data = io.BytesIO()
    mask.save(data, "PNG")
    encoded_mask_data = base64.b64encode(data.getvalue())

    overlay = Image.blend(img, mask, opacity)
    data = io.BytesIO()
    overlay.save(data, "PNG")
    encoded_overlay_data = base64.b64encode(data.getvalue())

    return encoded_img_data, encoded_mask_data, encoded_overlay_data


def alter_curr_index(n, idx):
    """_summary_

    Args:
        n (_type_): _description_
        idx (_type_): _description_

    Returns:
        _type_: _description_
    """
    if idx > n -1: return 0
    return idx


def adjust_curr_index(img_list, mask_list, img_id):
    """_summary_

    Args:
        img_list (_type_): _description_
        mask_list (_type_): _description_
        img_id (_type_): _description_

    Returns:
        _type_: _description_
    """
    curr_index = 0
    for i, img_path in enumerate(img_list):
        img = img_path.split("/")[-1].split(".png")[0]
        if img == img_id:
            curr_index = i
            break
    return curr_index


def auto_save(img_list, mask_list, keep_dict, remove_dict, fname=f'{date.today().strftime("%Y%m%d")}_datareview'):
    """_summary_

    Args:
        img_list (_type_): _description_
        mask_list (_type_): _description_
        keep_dict (_type_): _description_
        remove_dict (_type_): _description_
        fname (_type_, optional): _description_. Defaults to f'{date.today().strftime("%Y%m%d")}_datareview'.
    """
    fname= f'{date.today().strftime("%Y%m%d")}_datareview'
    keep_fname, remove_fname = f"{fname}_keep.csv", f"{fname}_remove.csv"
    keep_paths, remove_paths = [], []
    for i, img_path in enumerate(img_list):
        img_id = img_path.split("/")[-1].split(".png")[0]
        if img_id in keep_dict: keep_paths.append({ "img_id": img_id, "img_path": img_path, "mask_path": mask_list[i]})
        elif img_id in remove_dict: remove_paths.append({ "img_id": img_id, "img_path": img_path, "mask_path": mask_list[i]})
    
    df_keep = pd.DataFrame(keep_paths)
    df_remove = pd.DataFrame(remove_paths)

    print(df_keep.head())
    print(df_remove.head())
    df_keep.to_csv(keep_fname, index=False)
    df_remove.to_csv(remove_fname, index=False)
    
    return



# intialize global variables
img_dir, mask_dir = "/volumes/data_fast/camvid/train", "/volumes/data_fast/camvid/train_labels"
img_list, mask_list = get_all_images(img_dir), get_all_images(mask_dir)
curr_index = 0
remove_dict, keep_dict = {}, {}


@app.route('/cdn/<path:fpath>')
def get_file(fpath):
    """_summary_

    Args:
        fpath (_type_): _description_

    Returns:
        _type_: _description_
    """
    fpath = fpath.split("/")
    dir_path = f"/{'/'.join(fpath[:-1])}"
    print(dir_path)
    return send_from_directory(dir_path, fpath[-1])


@app.route("/", methods=["GET", "POST"])
def index():
    """_summary_

    Returns:
        _type_: _description_
    """
    global img_list
    global mask_list
    global curr_index
    global keep_dict
    global remove_dict

    curr_index = 0

    img, mask, overlay = get_img_mask_overlay(img_list[curr_index], mask_list[curr_index])
    img_name = img_list[curr_index].split("/")[-1]
    mask_name = mask_list[curr_index].split("/")[-1]

    prev_img_id = img_list[curr_index-1].split("/")[-1].split(".png")[0]
    next_img_id = img_list[curr_index+1].split("/")[-1].split(".png")[0]
    print(prev_img_id, next_img_id)

    return render_template(
                'index.html', 
                image=img.decode('utf-8'), 
                mask=mask.decode('utf-8'),
                overlay=overlay.decode('utf-8'),
                img_name = img_name,
                mask_name = mask_name,
                prev_img_id = prev_img_id,
                next_img_id = next_img_id,
                keep_list=keep_dict,
                remove_list=remove_dict
            )

@app.route('/<img_id>', methods=["GET", "POST"])
def get_img_mask_pair_by_id(img_id, methods=['GET', 'POST']):
    """_summary_

    Args:
        img_id (_type_): _description_
        direction (_type_): _description_

    Returns:
        _type_: _description_
    """

    print(img_id)
    global img_list
    global mask_list
    global curr_index
    global keep_dict
    global remove_dict

    n = len(img_list)

    # Adjust current index -- in case hopping directly to page
    curr_index = adjust_curr_index(img_list, mask_list, img_id)
    print(f"Adjusted Curr Index: {curr_index}")

    # figure out if iterating through direction
    direction = request.args.get('direction')
    prev_index = curr_index - 1

    # POST Body Request Form
    data = dict(request.form)
    if 'action' in data and 'img_id' in data:
        action = data['action']
        id = data['img_id']
        if action == "keep":
            keep_dict[id] = keep_dict.get(id, 0) + 1
            if id in remove_dict: del remove_dict[id]
        elif action == "remove":
            remove_dict[id] = remove_dict.get(id, 0) + 1
            if id in keep_dict: del keep_dict[id]

        print(keep_dict, remove_dict)
        print(f"{action} Image ID: {id}")

        # auto save progress
        auto_save(img_list, mask_list, keep_dict, remove_dict)


    # Handle navigation
    print(f"Prev: {prev_index}, Curr: {curr_index}")

    # Get Image & Mask Pair Data
    img, mask, overlay = get_img_mask_overlay(img_list[curr_index], mask_list[curr_index])
    img_name = img_list[curr_index].split("/")[-1]
    mask_name = mask_list[curr_index].split("/")[-1]

    prev_img_id = img_list[prev_index].split("/")[-1].split(".png")[0]
    next_img_id = img_list[alter_curr_index(n, curr_index+1)].split("/")[-1].split(".png")[0]
    print(prev_img_id, next_img_id)

    if direction == "next": curr_index = alter_curr_index(n, curr_index+1)
    elif direction == "prev": curr_index -=1


    return render_template(
                'index.html', 
                image=img.decode('utf-8'), 
                mask=mask.decode('utf-8'),
                overlay=overlay.decode('utf-8'),
                img_name = img_name,
                mask_name = mask_name,
                prev_img_id = prev_img_id,
                next_img_id = next_img_id,
                keep_list=keep_dict,
                remove_list=remove_dict
            )


if __name__=="__main__":
    app.run(host="0.0.0.0", port=5050)