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
        array = np.moveaxis(array, -1, 1)

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


def make_training_hdf5(tile_size=512):
    """
        Iterate over the training x, y tifs,
        compress them in tiles in hdf5 format
        and get 1/3 per image for validation.
    """
    for x, y in zip(glob('training/x*.tif'), glob('training/y*.tif')):

        X = gdal_array.LoadFile(x)
        Y = gdal_array.LoadFile(y)

        X_tiles = to_tiles(X, tile_size=tile_size) / 2048
        Y_tiles = to_tiles(Y, tile_size=tile_size)

        x_name = x.split("/")[-1].split('.')[-2]
        y_name = y.split("/")[-1].split('.')[-2]

        with h5py.File(name='training/training_data.hdf5', mode='a') as f:
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
    def __init__(self):
        self.file = h5py.File('training/training_data.hdf5')
        self.X = self.file['training/X']
        self.Y = self.file['training/Y']
        self.train_paths = [
            ('training/X/'+keyx, 'training/Y/'+keyy)
            for keyx, keyy in zip(self.X.keys(), self.Y.keys())
        ]
        self.__lengths = [
            self.file[path[0]].len() for path in self.train_paths
        ]
        self.__cum_len = [
            sum(self.__lengths[:i]) for i in range(1, len(self.__lengths)+1)
        ]
        self._p = {
            'rotation': (0, 180),
            'brightness': (0.5, 1.5),
            'contrast': (.7, 1.3)
        }
        self.transforms = transforms.Compose(
            [
                transforms.RandomRotation((0, 180)),
                transforms.ColorJitter(.5, .5, .5, .5),
                F.adjust_contrast()
            ]
        )
    
    @staticmethod
    def _adjust_contrast_(img: Tensor, factor: float):
        """
            Custom contrast adjustment method supporting
            more than 3 channels, based on torchvision.transforms.F
            _blend() implementation.
            
            Added multiplication by (img > 0) which keeps nodata
            values unaffected.
        """
        mean = img.mean((-3, -2, -1), keepdim=True)
        return factor * img + (1 - factor) * mean * (img > 0)
    
    @staticmethod
    def _adjust_brightness_(img: Tensor, factor: float):
        return img*factor
    
    @staticmethod
    def _random_color_jitter_(img: Tensor, *args: List[float]):
        p = transforms.ColorJitter().get_params(*args)
        """
            ColorJitter for 2 3/4 and average results
            Apply operations sequentially in right order.
        """
        ...
        
    def __len__(self):
        return sum(self.__lengths)

    def __getitem__(self, index):
        # Figure out Dataset index
        grp_idx = list(map(lambda x: min(x, index),
                           self.__cum_len)).index(index)
        path = self.train_paths[grp_idx]
        # Figure out index within Dataset
        idx = self.__cum_len[grp_idx-1] % index - 1
        return self.file[path][idx]


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


if not os.path.exists("training/training_data.hdf5"):
    make_training_hdf5()
