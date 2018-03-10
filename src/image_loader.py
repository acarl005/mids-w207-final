import numpy as np
from multiprocessing import Pool
from math import floor
from glob import glob
from PIL import Image
from os import path, makedirs

class Worker:
    def __init__(self, desired_size, scale_down=False, scale_up=False, pad_up=False, write_to=None, preserve_dir_tree_at=None, rotate=0):
        self.desired_size = desired_size
        self.scale_down = scale_down
        self.scale_up = scale_up
        self.pad_up = pad_up
        self.write_to = write_to
        self.preserve_dir_tree_at = preserve_dir_tree_at
        # convert to absolute path for consistency
        if self.preserve_dir_tree_at:
            self.preserve_dir_tree_at = path.abspath(self.preserve_dir_tree_at)
        self.rotate = rotate

    def __call__(self, file_path):
        im = Image.open(file_path).convert("RGB")
        w, h = im.size
        if self.scale_down and (w > self.desired_size or h > self.desired_size):
            im = self.resize_image(im)
        elif (w, h) != (self.desired_size, self.desired_size):
            if self.scale_up:
                im = self.resize_image(im)
            elif self.pad_up:
                im = self.pad_image(im)
        if self.rotate:
            im = im.rotate(self.rotate)
        if self.write_to:
            self.save_image(im, file_path)
        return np.array(im) / 255

    def pad_image(self, im):
        new_im = Image.new("RGB", size=(self.desired_size, self.desired_size), color=0)
        w_margin = (self.desired_size - im.size[0]) / 2
        h_margin = (self.desired_size - im.size[1]) / 2
        new_im.paste(im, (floor(w_margin), floor(h_margin), floor(self.desired_size - w_margin), floor(self.desired_size - h_margin)))
        return new_im

    def resize_image(self, im):
        w, h = im.size
        aspect_ratio = w / h
        if (aspect_ratio > 1):
            im = im.resize((self.desired_size, floor(self.desired_size / aspect_ratio)), Image.ANTIALIAS)
        else:
            im = im.resize((floor(self.desired_size * aspect_ratio), self.desired_size), Image.ANTIALIAS)
        if w != h:
            return self.pad_image(im)
        else:
            return im

    def save_image(self, im, file_path):
        makedirs(self.write_to, exist_ok=True)
        if not self.preserve_dir_tree_at:
            im.save(path.join(self.write_to, path.basename(file_path)))
        else:
            abs_file_path = path.abspath(file_path)
            if not abs_file_path.startswith(self.preserve_dir_tree_at):
                raise Exception("%s must be inside %s" % (file_path, self.preserve_dir_tree_at))
            else:
                sub_dir = path.dirname(abs_file_path[len(self.preserve_dir_tree_at) + 1:])
                full_dir = path.join(self.write_to, sub_dir)
                makedirs(full_dir, exist_ok=True)
                im.save(path.join(full_dir, path.basename(file_path)))

pool = Pool()

def load_images(paths, desired_size=224, scale_down=False, scale_up=False, pad_up=False, write_to=None, preserve_dir_tree_at=None, return_file_paths=False, rotate=0):
    """Take a glob pattern for images and load into a 4-D numpy array, converted to grayscale"""
    if type(paths) == str:
        file_paths = glob(paths)
    elif type(paths) == list or type(paths) == np.ndarray or type(paths) == tuple:
        file_paths = []
        for each_path in paths:
            for file_name in glob(each_path):
                file_paths.append(file_name)
    else:
        raise Exception("cannot load: %s" % str(paths))

    if len(file_paths) == 0:
        raise Exception("no images found")

    worker = Worker(desired_size, scale_down, scale_up, pad_up, write_to, preserve_dir_tree_at, rotate)

    list_of_image_3d_arrays = pool.map(worker, file_paths)

    if return_file_paths:
        return list_of_image_3d_arrays, file_paths

    return list_of_image_3d_arrays

