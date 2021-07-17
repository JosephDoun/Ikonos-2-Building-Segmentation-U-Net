#!/usr/bin/python3

from typing import Iterable, List, Tuple
import h5py
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
from numpy.lib.type_check import imag
from osgeo import gdal_array, gdal
from torch.utils.data import Dataset
from glob import glob
import torch
from torch import Tensor
import torchvision.transforms.functional as F
import torch.nn.functional as nn_F
from torchvision import transforms
import os
import logging
import argparse
from sys import argv

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.DEBUG,
    datefmt='%H:%M:%S %b %d'
)

log = logging.getLogger(argv[0])
log.setLevel(logging.DEBUG)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Data Creation',
                                     parents=[])
    argparse.ArgumentParser(prog="Data Loader")
    parser.add_argument(
        "--training-tile-size", '-t',
        help="2D dimensions of training samples",
        type=int,
        default=32
    )
    parser.add_argument(
        "--validation-tile-size", '-v',
        help='2D dimensions of validation samples',
        type=int,
        default=512
    )
    parser.add_argument(
        "--validation-split", "-s",
        help="Percentage of samples to keep for validation - Float: 0 to 1",
        type=float,
        default=0.166
    )
    args = parser.parse_args(argv[1:])


def to_tiles(X, Y, tile_size=512):
    log.info(f"Splitting tif image to tiles... {tile_size} x {tile_size}")
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
        X = np.pad(X, ((0, 0),
                       (0, 0),
                       (0, tile_size - X.shape[-2] % tile_size),
                       (0, tile_size - X.shape[-1] % tile_size)),
                   constant_values=0)
        Y = np.pad(Y, ((0, 0),
                       (0, tile_size - Y.shape[-2] % tile_size),
                       (0, tile_size - Y.shape[-1] % tile_size)),
                   constant_values=0)

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

    # idx = X.mean((-3, -2, -1)) >= 0.05
    idx = (X != 0).mean((-3, -2, -1)) > 0.20
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
            val_samples = int(len(idx) * args.validation_split) or 1

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
    with h5py.File("Training/training_data.hdf5", mode='r') as f:
        log.info(
            f"""
            
            Positive samples: {f['training/X/pos'].len()} of size {train_tiles} x {train_tiles}
            Negative samples: {f['training/X/neg'].len()} of size {train_tiles} x {train_tiles}
            
            Validation samples: {f['validation/X'].len()} of size {val_tiles} x {val_tiles}
            """
        )


class Buildings(Dataset):
    
    def log_augmentation(f):
        def wrapper(self, *args):
            if f.__name__ not in self.augmentations.keys():
                self.augmentations[f.__name__] = args[1:]
            r = f(self, *args)
            return r
        return wrapper
    
    def __init__(self, validation: bool = False,
                 aug_split=.66, ratio=10):

        self.aug_split = aug_split
        self.validation = validation
        self.ratio = ratio

        # h5py dataset objects
        self.file = h5py.File('Training/training_data.hdf5')
        self.X_train_pos = self.file['training/X/pos']
        self.Y_train_pos = self.file['training/Y/pos']
        self.X_train_neg = self.file['training/X/neg']
        self.Y_train_neg = self.file['training/Y/neg']
        self.X_val = self.file['validation/X']
        self.Y_val = self.file['validation/Y']

        self.augmentations = {}

    @log_augmentation
    def _adjust_contrast_(self, img: Tensor, factor: float):
        """
        Contrast adjustment function based on
        torchvision.transforms.functional_Tensor._blend()
        modified for multiple channels 
        """
        if self.validation:
            return img
        mean = img.mean((-3, -2, -1), keepdim=True)
        return (factor * img + (1 - factor) * mean)  # .clamp(0, 1)

    @log_augmentation
    def _adjust_brightness_(self, img: Tensor, factor: float):
        """
        Brightness adjustment
        """
        if self.validation:
            return img
        return img*factor  # .clamp(0, 1)

    @log_augmentation
    def _affine_(self, img: List[Tensor]):
        if self.validation:
            return img[0], img[1]
        
        degrees = torch.rand(1).item() * 360 - 180
        translations = [int(img[0].shape[-2] * .15 * torch.rand(1)),
                        int(img[0].shape[-1] * .15 * torch.rand(1))]
        scale = torch.rand(1) * 0.1 + 1
        shear = torch.rand(1).item() * 120 - 60
        
        img[0] = F.affine(img[0], angle=degrees,
                          translate=translations,
                          scale=scale,
                          shear=shear)
        img[1] = F.affine(img[1], angle=degrees,
                          translate=translations,
                          scale=scale,
                          shear=shear)
        return img[0], img[1]

    @log_augmentation
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

    @log_augmentation
    def _noise_(self, img: Tensor, factor: float):
        if self.validation:
            return img
        img = img + torch.randn_like(img) * factor
        return img.clamp(0, 1)

    @log_augmentation
    def _elastic_deformation_(self, img: List[Tensor],
                              k: int,
                              sigma: float,
                              alpha: int):
        if self.validation:
            return img
        """
        Elastic deformation based on Simard et al. 2003
        *Best Practices for Convolutional Neural Networks Appled to
        Visual Document Analysis*
        
        Implemented as to the recommendation of Ronneberger et al. 2015
        in the original U-Net paper.
        """
        dy = F.gaussian_blur(
            2 * torch.rand(1, *img[1].shape) - 1, k, sigma=sigma
        ) * alpha
        dx = F.gaussian_blur(
            2 * torch.rand(1, *img[1].shape) - 1, k, sigma=sigma
        ) * alpha

        idxy = torch.arange(img[1].shape[-2])
        idxx = torch.arange(img[1].shape[-1])

        y, x = torch.meshgrid(idxy, idxx)

        grid = torch.cat([(y+dy).unsqueeze(-1),
                          (x+dx).unsqueeze(-1)], -1).squeeze(0)

        img[0] = nn_F.grid_sample(img[0].unsqueeze(0),
                                  grid/(img[0].size(-1) - 1),
                                  align_corners=True,
                                  mode='bilinear').squeeze(0).clamp(0, 1)

        img[1] = nn_F.grid_sample(img[1].float().reshape(1, *img[1].shape),
                                  grid/(img[1].size(-1) - 1),
                                  align_corners=True,
                                  mode='nearest').reshape(img[1].shape)
        img[1][img[1] > 0] = 1
        return img

    @log_augmentation
    def _random_crop_(self, img: List[Tensor], size: Tuple[int]):
        if self.validation:
            return img
        p = transforms.RandomCrop.get_params(img[0], size)
        img[0] = F.crop(img[0], *p)
        img[1] = F.crop(img[1], *p)
        return img

    def __len__(self):
        if self.validation:
            length = self.X_val.len()
        else:
            length = self.ratio*self.X_train_neg.len()
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
            Iterate over samples normally

        """
        if not self.validation:
            ratio = self.ratio
            if index % ratio:
                features = self.X_train_pos
                labels = self.Y_train_pos
                idx = index - 1 - index // ratio
                idx %= len(features)
            elif not index % ratio:
                features = self.X_train_neg
                labels = self.Y_train_neg
                idx = index // ratio
        elif self.validation:
            idx = index
            features = self.X_val
            labels = self.Y_val

        # if not self.validation:
        #     # Default to pos
        #     features, labels = self.X_train_pos, self.Y_train_pos
        #     idx = index % self.X_train_pos.len()

        image, label = (torch.from_numpy(features[idx]),
                        torch.from_numpy(labels[idx]))

        # image, label = self._random_crop_([image, label], (64, 64))
        image, label = self._affine_([image, label.unsqueeze(0)])
        # image, label = self._random_flip_([image, label])
        # image, label = self._elastic_deformation_([image, label],
        #                                           k=21,
        #                                           sigma=10,
        #                                           alpha=30)
        image = self._noise_(image, 0.01)
        image = self._adjust_brightness_(image, torch.rand(1)*1. + .5)
        image = self._adjust_contrast_(image, torch.rand(1)*1. + .5)
        return image, label.to(torch.long).squeeze(0)
    
    def _augment_(self, img: List[Tensor]):
        """
        Collect and apply defined augmentations here
        
        :param img: List of tensors in [feature, label] format
        """
        pass
    

if not os.path.exists("Training/training_data.hdf5"):
    os.makedirs("Training", exist_ok=True)
    make_training_hdf5(train_tiles=args.training_tile_size,
                       val_tiles=args.validation_tile_size)
