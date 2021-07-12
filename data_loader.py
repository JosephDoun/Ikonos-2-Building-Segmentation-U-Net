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
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.DEBUG,
    datefmt='%b %d %H:%M:%S'
)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def to_tiles(X, Y, tile_size=512):
    log.info("Splitting tif image to tiles...")
    if X.ndim == 3 and tile_size == 512:
        # Pad X to be divisible by tile size
        X = np.pad(X, ((0, 0),
                       (0, tile_size - X.shape[1] % tile_size),
                       (0, tile_size - X.shape[2] % tile_size)),
                   constant_values=0)

        channel_axis = X.shape.index(min(X.shape))
        assert channel_axis == 0, "channel axis is not -3"

        # Move channels axis to the end for convenience
        X = np.moveaxis(X, channel_axis, -1)
        arr_height, arr_width, channels = X.shape
        X = X.reshape(arr_height // tile_size,
                      tile_size,
                      arr_width // tile_size,
                      tile_size, channels)
        X = X.swapaxes(1, 2).reshape(-1,
                                     tile_size,
                                     tile_size,
                                     channels)
        # Return channels to axis -3 as expected by PyTorch
        X = np.moveaxis(X, -1, 1)

        # Pad Y to be divisible by tile size
        Y = np.pad(Y, ((0, tile_size - Y.shape[0] % tile_size),
                       (0, tile_size - Y.shape[1] % tile_size)),
                   constant_values=0)

        arr_height, arr_width = Y.shape
        Y = Y.reshape(arr_height // tile_size,
                      tile_size,
                      arr_width // tile_size,
                      tile_size)
        Y = Y.swapaxes(1, 2).reshape(-1,
                                     tile_size,
                                     tile_size)

    elif X.ndim == 4:
        X = np.moveaxis(X, 1, -1)
        samples, height, width, channels = X.shape
        X = X.reshape(samples,
                      height//tile_size,
                      tile_size,
                      width//tile_size,
                      tile_size,
                      channels)
        X = X.swapaxes(2, 3).reshape(-1,
                                     tile_size,
                                     tile_size,
                                     channels)
        X = np.moveaxis(X, -1, -3)

        samples, height, width = Y.shape
        Y = Y.reshape(samples,
                      height // tile_size,
                      tile_size,
                      width // tile_size,
                      tile_size)
        Y = Y.swapaxes(2, 3).reshape(-1,
                                     tile_size,
                                     tile_size)
    return X, Y


def clean_tiles(X, Y):
    """
        Remove mostly no-data tiles
    """
    log.info(
        "Cleaning up no-data tiles"
    )
    
    idx = X.mean((-3, -2, -1)) >= 0.02
    X = X[idx]
    Y = Y[idx]
    return X, Y


def separate_labels(X, Y):
    log.info(
        "Separating batch labels to positive and negative"
    )
    idx_pos = Y.mean((-1, -2)) >= 0.1
    X_pos, Y_pos = X[idx_pos], Y[idx_pos]
    X_neg, Y_neg = X[~idx_pos], Y[~idx_pos]
    return X_pos, Y_pos, X_neg, Y_neg


def make_training_hdf5(train_tiles=512, val_tiles=512):
    """
        Iterate over the training x, y tifs,
        compress them in tiles in hdf5 format
        and get 1/3 per image for validation.
    """
    log.info(
        "Creating HDF5 dataset for training and validation"
    )
    for x, y in zip(glob('Training/x*.tif'), glob('Training/y*.tif')):
        
        log.info(
            "Processing sub area %s with labels %s " % (x, y)
        )
        # Load tifs as arrays
        X = gdal_array.LoadFile(x) / (2**11)
        Y = gdal_array.LoadFile(y)

        X_tiles, Y_tiles = to_tiles(X, Y, tile_size=val_tiles)
        X_tiles, Y_tiles = clean_tiles(X_tiles, Y_tiles)

        with h5py.File(name='Training/training_data.hdf5', mode='a') as f:
            training_x = f.require_group('training/X')
            training_y = f.require_group('training/Y')

            idx = np.random.permutation(len(X_tiles))
            val_samples = len(idx) // 3

            X_train = X_tiles[idx[val_samples:]]
            Y_train = Y_tiles[idx[val_samples:]]
            X_train, Y_train = to_tiles(
                X_train, Y_train, tile_size=train_tiles
                )
            X_train, Y_train = clean_tiles(X_train, Y_train)
            X_train_pos, Y_train_pos, X_train_neg, Y_train_neg = \
                separate_labels(X_train, Y_train)

            X_train_pos, Y_train_pos = clean_tiles(X_train_pos, Y_train_pos)
            X_train_neg, Y_train_neg = clean_tiles(X_train_neg, Y_train_neg)
            
            if val_samples:
                X_valid = X_tiles[idx[:val_samples]]
                Y_valid = Y_tiles[idx[:val_samples]]
            else:
                X_valid = 0
                Y_valid = 0


            def expand(dataset, data, label: bool):
                """
                dataset: h5py dataset instance for expansion
                data: data to fit in dataset
                label: flag to assume dataset & data dimensions
                """
                log.info("Expanding HDF5 file...")
                if not label:
                    dataset_shape = dataset.shape
                    dataset.resize(
                        (
                            dataset_shape[0] + data.shape[0],
                            dataset_shape[1],
                            dataset_shape[2],
                            dataset_shape[3]
                        )
                    )
                    dataset[dataset_shape[0]:] = data
                elif label:
                    dataset_shape = dataset.shape
                    dataset.resize(
                        (
                            dataset_shape[0] + data.shape[0],
                            dataset_shape[1],
                            dataset_shape[2]
                        )
                    )
                    dataset[dataset_shape[0]:] = data
                    
            assert X_train_pos.shape[0] == Y_train_pos.shape[0]
            assert X_train_neg.shape[0] == Y_train_neg.shape[0]
            
            try:
                positive_x = training_x['pos']
                positive_y = training_y['pos']
                negative_x = training_x['neg']
                negative_y = training_y['neg']
                
                expand(positive_x, X_train_pos, False)
                expand(positive_y, Y_train_pos, True)
                expand(negative_x, X_train_neg, False)
                expand(negative_y, Y_train_neg, True)

            except KeyError:
                positive_x = training_x.create_dataset("pos",
                                                       shape=X_train_pos.shape,
                                                       dtype='f',
                                                       compression='gzip',
                                                       compression_opts=8,
                                                       maxshape=(None,
                                                                 4,
                                                                 train_tiles,
                                                                 train_tiles),
                                                       data=X_train_pos)
                positive_y = training_y.create_dataset("pos",
                                                       shape=Y_train_pos.shape,
                                                       dtype='i1',
                                                       compression='gzip',
                                                       compression_opts=8,
                                                       maxshape=(None,
                                                                 train_tiles,
                                                                 train_tiles),
                                                       data=Y_train_pos)
                negative_x = training_x.create_dataset('neg',
                                                       shape=X_train_neg.shape,
                                                       compression='gzip',
                                                       compression_opts=8,
                                                       maxshape=(None,
                                                                 4,
                                                                 train_tiles,
                                                                 train_tiles),
                                                       dtype='f',
                                                       data=X_train_neg)
                negative_y = training_y.create_dataset('neg',
                                                       shape=Y_train_neg.shape,
                                                       dtype='i1',
                                                       compression='gzip',
                                                       compression_opts=8,
                                                       maxshape=(None,
                                                                 train_tiles,
                                                                 train_tiles),
                                                       data=Y_train_neg)

            if val_samples:

                validation = f.require_group("validation")

                try:
                    _X = validation['X']
                    _Y = validation['Y']

                    expand(_X, X_valid, False)
                    expand(_Y, Y_valid, True)
                    
                except KeyError:
                    _X = validation.create_dataset('X',
                                                   shape=X_valid.shape,
                                                   compression='gzip',
                                                   compression_opts=8,
                                                   maxshape=(None,
                                                               4,
                                                               val_tiles,
                                                               val_tiles),
                                                   dtype='f',
                                                   data=X_valid)
                    _Y = validation.create_dataset('Y',
                                                   shape=Y_valid.shape,
                                                   dtype='i1',
                                                   compression='gzip',
                                                   compression_opts=8,
                                                   maxshape=(None,
                                                             val_tiles,
                                                             val_tiles),
                                                   data=Y_valid)


class Buildings(Dataset):
    def __init__(self, validation: bool = False,
                 aug_split=.66):
        
        self.aug_split = aug_split
        self.validation = validation
        
        # h5py dataset objects
        self.file = h5py.File('Training/training_data.hdf5')
        self.X_train_pos = self.file['training/X/pos']
        self.Y_train_pos = self.file['training/Y/pos']
        self.X_train_neg = self.file['training/X/neg']
        self.Y_train_neg = self.file['training/Y/neg']
        self.X_val = self.file['validation/X']
        self.Y_val = self.file['validation/Y']

        # Augmentation ranges
        self._p = {
            'rotation': (0, 180),
            'brightness': (.5, 1.5),
            'contrast': (.5, 1.5),
            'saturation': (.5, 1.5),
            'hue': (-.01, .01),
            'affine': (
                # Rotation
                (-180, 180),
                # Translation
                (0, 0),
                # Scaling
                (0.5, 1.5),
                # Shear
                (-22, 23, -22, 23),
                # Size
                (4, 512, 512)
            )
        }
        # Transformation classes
        self.transforms = {
            "rotate": transforms.RandomRotation(self._p['rotation']),
            "color": transforms.ColorJitter(),
            "affine": transforms.RandomAffine(10)
        }
        # Color manipulation functions
        # Not used // Deprecated
        # self.color_functions = [
        #     F.adjust_brightness,
        #     F.adjust_contrast,
        #     F.adjust_saturation,
        #     F.adjust_hue
        # ]

    def _adjust_contrast_(self, img: Tensor, factor: float, R: float):
        """
        Contrast adjustment function based on
        torchvision.transforms.functional_Tensor._blend()
        modified for multiple channels 
        """
        if self.validation or R < (1 - self.aug_split):
            return img
        mean = img.mean((-3, -2, -1), keepdim=True)
        return (factor * img + (1 - factor) * mean)  # .clamp(0, 1)

    def _adjust_brightness_(self, img: Tensor, factor: float, R: float):
        """
        Brightness adjustment function with added noise
        """
        if self.validation or R < (1 - self.aug_split):
            return img
        return ((img + torch.randn_like(img)*0.01)*factor)  # .clamp(0, 1)

    def _random_affine_trans_(self, img: List[Tensor], R):
        if self.validation or R < (1-self.aug_split):
            return img[0], img[1]
        p = self.transforms['affine'].get_params(*self._p['affine'])
        img[0], img[1] = F.affine(img[0], *p), F.affine(img[1], *p)
        return img[0], img[1]

    def _random_flip_(self, img: List[Tensor]):
        if self.validation:
            return img
        if torch.rand(1) < .5:
            img[0] = F.hflip(img[0])
            img[1] = F.hflip(img[1])
        if torch.rand(1) < .5:
            img[0] = F.vflip(img[0])
            img[1] = F.vflip(img[1])
        return img[0], img[1]

    def __len__(self):
        if self.validation:
            length = self.X_val.len()
        else:
            length = self.X_train_pos.len() + self.X_train_neg.len()
        return length

    def __getitem__(self, index):
        """
        Retrieve sample
        
        Training class:
            Even index:
                Return positive sample
            Odd index:
                Return negative sample
        
        Validation class:
            Iterate over samples as normal
            
        """
        if not self.validation:
            if not index % 2:
                features = self.X_train_pos
                labels = self.Y_train_pos
                idx = index // 2
                idx %= len(features)
            elif index % 2:
                features = self.X_train_neg
                labels = self.Y_train_neg
                idx = torch.randint(features.len(), (1,))
        elif self.validation:
            idx = index
            features = self.X_val
            labels = self.Y_val

        image, label = (torch.from_numpy(features[idx]),
                        torch.from_numpy(labels[idx]))

        R_color = torch.rand(1)
        R_affine = torch.rand(1)

        image = self._adjust_brightness_(image, torch.rand(1)*.99+0.5,
                                         R=R_color)
        image = self._adjust_contrast_(image, torch.rand(1)*.99+0.5,
                                       R=R_color)
        image, label = self._random_affine_trans_([image, label.unsqueeze(0)],
                                                  R=R_affine)
        image, label = self._random_flip_([image, label])
        return image, label.to(torch.long).squeeze(0)


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
    os.makedirs("Training", exist_ok=True)
    make_training_hdf5(train_tiles=64, val_tiles=512)
