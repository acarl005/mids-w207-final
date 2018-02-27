import numpy as np
from math import floor
import glob
from PIL import Image
from os import path, makedirs


def load_images(paths, desired_size=224, scale_down=False, scale_up=False, pad_up=False, write_to=None, preserve_dir_tree_at=None):
    """Take a glob pattern for images and load into a 4-D numpy array, converted to grayscale"""
    if type(paths) == str:
        file_paths = glob.glob(paths)
    elif type(paths) == list:
        file_paths = []
        for each_path in paths:
            for file_name in glob.glob(each_path):
                file_paths.append(file_name)
    else:
        raise Exception("cannot load: %s" % str(paths))

    if preserve_dir_tree_at:
        preserve_dir_tree_at = path.abspath(preserve_dir_tree_at)

    list_of_image_3d_arrays = []
    for file_path in file_paths:
        im = Image.open(file_path).convert("RGB")
        w, h = im.size
        if scale_down and (w > desired_size or h > desired_size):
            im = resize_image(im, desired_size)
        if write_to:
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

        arr = np.array(im)
        list_of_image_3d_arrays.append(arr)
    to_4d_array_of_images = np.array(list_of_image_3d_arrays)
    return to_4d_array_of_images

def resize_image(im, desired_size):
    aspect_ratio = im.size[0] / im.size[1]
    newIm = Image.new("RGB", size=(desired_size, desired_size), color=0)
    if (aspect_ratio > 1):
        im = im.resize((desired_size, floor(desired_size / aspect_ratio)), Image.ANTIALIAS)
        margin = (desired_size - im.size[1]) / 2
        newIm.paste(im, (0, floor(margin), desired_size, floor(desired_size - margin)))
    else:
        im = im.resize((floor(desired_size * aspect_ratio), desired_size), Image.ANTIALIAS)
        margin = (desired_size - im.size[0]) / 2
        newIm.paste(im, (floor(margin), 0, floor(desired_size - margin), desired_size))
    return newIm


