#!/usr/bin/python3

from typing import Iterable, List, Tuple
import h5py
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal_array, gdal
from torch.utils.data import Dataset
from glob import glob
import torch
from torch import Tensor
import torchvision.transforms.functional as F
from torchvision import transforms
import os


def to_tiles(array, tile_size=512):
    if array.ndim == 3:

        array = np.pad(array, ((0, 0),
                               (0, tile_size - array.shape[1] % tile_size),
                               (0, tile_size - array.shape[2] % tile_size)),
                       constant_values=0)

        channel_axis = array.shape.index(min(array.shape))
        assert channel_axis == 0, "channel axis is not first"
        array = np.moveaxis(array, channel_axis, -1)
        arr_height, arr_width, channels = array.shape
        array = array.reshape(arr_height // tile_size,
                              tile_size,
                              arr_width // tile_size,
                              tile_size, channels)
        array = array.swapaxes(1, 2).reshape(-1,
                                             tile_size,
                                             tile_size,
                                             channels)
        array = np.moveaxis(array, -1, 1) / 2048

    elif array.ndim == 2:
        
        array = np.pad(array, ((0, tile_size - array.shape[0] % tile_size),
                               (0, tile_size - array.shape[1] % tile_size)),
                       constant_values=0)

        arr_height, arr_width = array.shape
        array = array.reshape(arr_height // tile_size,
                              tile_size,
                              arr_width // tile_size,
                              tile_size)
        array = array.swapaxes(1, 2).reshape(-1,
                                             tile_size,
                                             tile_size)
    return array


def clean_tiles(X, Y):
    """
        Remove empty tiles
    """
    idx = X.mean((-3, -2, -1)) >= 0.02
    X = X[idx]
    Y = Y[idx]
    return X, Y


def make_training_hdf5(tile_size=512):
    """
        Iterate over the training x, y tifs,
        compress them in tiles in hdf5 format
        and get 1/3 per image for validation.
    """
    for x, y in zip(glob('Training/x*.tif'), glob('Training/y*.tif')):

        X = gdal_array.LoadFile(x)
        Y = gdal_array.LoadFile(y)

        X_tiles = to_tiles(X, tile_size=tile_size)
        Y_tiles = to_tiles(Y, tile_size=tile_size)
        X_tiles, Y_tiles = clean_tiles(X_tiles, Y_tiles)

        # Get file names minus the extension
        x_name = x.split("/")[-1].split('.')[-2]
        y_name = y.split("/")[-1].split('.')[-2]

        with h5py.File(name='Training/training_data.hdf5', mode='a') as f:
            training_x = f.require_group('training/X')
            training_y = f.require_group('training/Y')

            idx = np.random.permutation(len(X_tiles))
            val_samples = len(idx) // 3

            X_train = X_tiles[idx[val_samples:]]
            Y_train = Y_tiles[idx[val_samples:]]
            
            if val_samples:
                X_valid = X_tiles[idx[:val_samples]]
                Y_valid = Y_tiles[idx[:val_samples]]
            else:
                X_valid = 0
                Y_valid = 0
                
            training_x.require_dataset(x_name,
                                       shape=X_train.shape,
                                       compression='gzip',
                                       compression_opts=8,
                                       track_order=True,
                                       chunks=(1,
                                               4,
                                               tile_size,
                                               tile_size),
                                       dtype='f',
                                       data=X_train)
            training_y.require_dataset(y_name,
                                       shape=Y_train.shape,
                                       compression='gzip',
                                       compression_opts=8,
                                       track_order=True,
                                       chunks=(1,
                                               tile_size,
                                               tile_size),
                                       dtype='i1',
                                       data=Y_train)

            if val_samples:
                validation_x = f.require_group("validation/X")
                validation_y = f.require_group("validation/Y")
                validation_x.require_dataset(x_name,
                                             shape=X_valid.shape,
                                             compression='gzip',
                                             track_order=True,
                                             chunks=(1,
                                                     4,
                                                     tile_size,
                                                     tile_size),
                                             dtype='f',
                                             data=X_valid)
                validation_y.require_dataset(y_name,
                                             shape=Y_valid.shape,
                                             compression='gzip',
                                             track_order=True,
                                             chunks=(1,
                                                     tile_size,
                                                     tile_size),
                                             dtype='i1',
                                             data=Y_valid)
                
                
class Buildings(Dataset):
    def __init__(self, validation: bool=False,
                 aug_split=.5):
        self.aug_split = aug_split
        self.validation = validation
        self.file = h5py.File('Training/training_data.hdf5')
        self.X = self.file['training/X']
        self.Y = self.file['training/Y']
        self.Xv = self.file['validation/X']
        self.Yv = self.file['validation/Y']
        self.train_paths = [
            ('training/X/'+keyx, 'training/Y/'+keyy)
            for keyx, keyy in zip(self.X.keys(), self.Y.keys())
        ]
        self.validation_paths = [
            ('validation/X/'+keyx, 'validation/Y/'+keyy)
            for keyx, keyy in zip(self.Xv.keys(), self.Yv.keys())
        ]
        self._train_lengths = [
            self.file[path[0]].len() for path in self.train_paths
        ]
        self._validation_lengths = [
            self.file[path[0]].len() for path in self.validation_paths
        ]
        self._cum_train_len = [
            # Subtract 1 to transform into indexes
            sum(self._train_lengths[:i]) - 1
            for i in range(1, len(self._train_lengths)+1)
        ]
        self._cum_validation_len = [
            sum(self._validation_lengths[:i]) - 1
            for i in range(1, len(self._validation_lengths)+1)
        ]
        self._p = {
            'rotation': (0, 180),
            'brightness': (.5, 1.5),
            'contrast': (.5, 1.5),
            'saturation': (.5, 1.5),
            'hue': (-.01, .01)
        }
        self.transforms = {
                "rotate": transforms.RandomRotation(self._p['rotation']),
                "color": transforms.ColorJitter()
            }
        self.color_functions = [
            F.adjust_brightness,
            F.adjust_contrast,
            F.adjust_saturation,
            F.adjust_hue
        ]
    
    # @staticmethod
    # def _adjust_contrast_(img: Tensor, factor: float):
    #     """
    #         Custom contrast adjustment method supporting
    #         more than 3 channels, based on torchvision.transforms.F
    #         _blend() implementation.
            
    #         Added multiplication by (img > 0) which keeps nodata
    #         values unaffected.
    #     """
    #     mean = img.mean((-3, -2, -1), keepdim=True)
    #     return factor * img + (1 - factor) * mean * (img > 0)
    
    # @staticmethod
    # def _adjust_brightness_(img: Tensor, factor: float):
    #     return img*factor
    
    def _random_color_jitter_(self, img: Tensor, *args: List[float], **kwargs):
        """
            img: 4 dimensional Tensor of 4 channels
            args: ColorJitter min-maxes
            
            ColorJitter for first 3 channels, last 3 channels
            and average results. Apply operations sequentially
            the in right order.
            
            p: Returns (0: Tensor[int]: Sequence of operations,
                        1: float: factor for brightness rescaling,
                        2: float: factor for contrast rescaling,
                        3: float: factor for saturation rescaling,
                        4: float: factor for hue shift)
        """
        if self.validation or kwargs['R'] < (1-self.aug_split):
            return img
        
        p = self.transforms['color'].get_params(*args)
        assert img.dim() == 3, "Tensor is not 3 dimensional"
        # Break to 2 three-channel tensors
        # To be elligible for pytorch's functions
        img1 = img[:3]
        img2 = img[1:]
        for f_idx in p[0]:
            # p[1:][f_idx]: corresponding parameters for function f.
            img1 = self.color_functions[f_idx](img[:3, ...], p[1:][f_idx])
            img2 = self.color_functions[f_idx](img[1:, ...], p[1:][f_idx])
        img1 = torch.cat([img1, torch.zeros(1, img.size(-2), img.size(-1))], 0)
        img2 = torch.cat([torch.zeros(1, img.size(-2), img.size(-1)), img2], 0)
        # Rejoin and average overlap
        img = img1 + img2
        img[[1, 2]] = img[[1, 2]] / 2
        return img
            
    def _random_rotation_(self, img: List[Tensor], R):
        if self.validation or R < (1-self.aug_split):
            return img[0], img[1]
        p = self.transforms['rotate'].get_params(self._p['rotation'])
        img[0], img[1] = F.rotate(img[0], p), F.rotate(img[1], p)
        return img[0], img[1]
        
    def __len__(self):
        return (sum(self._train_lengths) if not self.validation
                else sum(self._validation_lengths))

    def __getitem__(self, index):
        """
            Look into accumulated indexes of hdf5 Datasets
            and figure out which Dataset it falls in, by mapping
            the cum_len list with min(index, cum_len). The Dataset
            index will then be the first occurence of given <index>.
            
            To get the item's index in the current Dataset, get the
            modulo of the previous Dataset and subtract 1.
            
            grp_idx: Dataset index in self.train_paths
                     If it is the first Dataset, simply
                     return <index>.
        """
        R = np.random.random()
        if not self.validation:
            grp_idx = list(map(lambda x: min(x, index),
                           self._cum_train_len)).index(index)
            path = self.train_paths[grp_idx]
            # Figure out index within Dataset.
            # Get the modulo of previous elements
            # with <index> and subtract one to
            # find how many steps further you need
            # to go.
            # If it falls in the first Dataset
            # (grp_idx == 0) simply return <index>.
            idx = (index % self._cum_train_len[grp_idx-1] - 1
                   if grp_idx else index)
        elif self.validation:
            grp_idx = list(map(lambda x: min(x, index),
                           self._cum_validation_len)).index(index)
            path = self.validation_paths[grp_idx]
            idx = (index % self._cum_validation_len[grp_idx-1] - 1
                   if grp_idx else index)
                   
        image, label = (torch.from_numpy(self.file[path[0]][idx]),
                        torch.from_numpy(self.file[path[1]][idx]))
        image = self._random_color_jitter_(image,
                                           self._p['brightness'],
                                           self._p['contrast'],
                                           self._p['saturation'],
                                           self._p['hue'],
                                           R=R)
        image, label = self._random_rotation_([image, label.unsqueeze(0)],
                                              R)
        return image, label.to(torch.long).squeeze(0)

    def _get_group_(self, index):
        grp_idx = list(map(lambda x: min(x, index),
                           self._cum_train_len)).index(index)
        path = self.train_paths[grp_idx]
        idx = (index % self._cum_train_len[grp_idx-1] - 1
               if grp_idx else index)
        return path, idx

def plot_samples(samples: List[Tuple[Tensor]]):
    # TODO make this better.
    fig, axes = plt.subplots(min(len(samples), 4), 2)
    for i, ax_row in enumerate(axes):
        sample = samples[i]
        sample = (sample[0].numpy(), sample[1].numpy())
        ax_row[0].imshow(
            np.moveaxis(
                sample[0], 0, -1
                )[:, :, [2, 1, 0]])
        ax_row[1].imshow(sample[1])
        ax_row[0].set_axis_off()
        ax_row[1].set_axis_off()
        ax_row[0].set_title("x%s" % i)
        ax_row[1].set_title("y%s" % i)
    plt.tight_layout()
    plt.show()


if not os.path.exists("Training/training_data.hdf5"):
    make_training_hdf5()
