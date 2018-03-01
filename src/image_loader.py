import numpy as np
from math import floor
from glob import glob
from PIL import Image
from os import path, makedirs

def load_images(paths, desired_size=224, scale_down=False, scale_up=False, pad_up=False, write_to=None, preserve_dir_tree_at=None, file_names=False):
    """Take a glob pattern for images and load into a 4-D numpy array, converted to grayscale"""
    if type(paths) == str:
        file_paths = glob(paths)
    elif type(paths) == list:
        file_paths = []
        for each_path in paths:
            for file_name in glob(each_path):
                file_paths.append(file_name)
    else:
        raise Exception("cannot load: %s" % str(paths))

    list_of_image_3d_arrays = []
    list_of_file_names = []
    for file_path in file_paths:
        im = Image.open(file_path).convert("RGB")
        w, h = im.size
        if scale_down and (w > desired_size or h > desired_size):
            im = resize_image(im, desired_size)
        elif (w, h) != (desired_size, desired_size):
            if scale_up:
                im = resize_image(im, desired_size)
            elif pad_up:
                im = pad_image(im, desired_size)
        if write_to:
            save_image(im, file_path, write_to, preserve_dir_tree_at)
        arr = np.array(im) / 255
        list_of_image_3d_arrays.append(arr)
        list_of_file_names.append(path.basename(file_path))

    if file_names:
        return list_of_image_3d_arrays, list_of_file_names

    return list_of_image_3d_arrays

def pad_image(im, desired_size):
    new_im = Image.new("RGB", size=(desired_size, desired_size), color=0)
    w_margin = (desired_size - im.size[0]) / 2
    h_margin = (desired_size - im.size[1]) / 2
    new_im.paste(im, (floor(w_margin), floor(h_margin), floor(desired_size - w_margin), floor(desired_size - h_margin)))
    return new_im

def resize_image(im, desired_size):
    w, h = im.size
    aspect_ratio = w / h
    if (aspect_ratio > 1):
        im = im.resize((desired_size, floor(desired_size / aspect_ratio)), Image.ANTIALIAS)
    else:
        im = im.resize((floor(desired_size * aspect_ratio), desired_size), Image.ANTIALIAS)
    if w != h:
        return pad_image(im, desired_size)
    else:
        return im

def save_image(im, file_path, write_to, preserve_dir_tree_at):
    if preserve_dir_tree_at:
        preserve_dir_tree_at = path.abspath(preserve_dir_tree_at)

    makedirs(write_to, exist_ok=True)
    if not preserve_dir_tree_at:
        im.save(path.join(write_to, path.basename(file_path)))
    else:
        abs_file_path = path.abspath(file_path)
        if not abs_file_path.startswith(preserve_dir_tree_at):
            raise Exception("%s must be inside %s" % (file_path, preserve_dir_tree_at))
        else:
            sub_dir = path.dirname(abs_file_path[len(preserve_dir_tree_at) + 1:])
            full_dir = path.join(write_to, sub_dir)
            makedirs(full_dir, exist_ok=True)
            im.save(path.join(full_dir, path.basename(file_path)))

