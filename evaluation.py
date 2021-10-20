#!/usr/bin/python3

from osgeo import gdal_array, gdal
from model_architecture import BuildingsModel
from data_loader import to_tiles
from glob import glob
from tqdm import tqdm
from sys import argv

import matplotlib.pyplot as plt
import torch
import h5py
import os
import argparse


plt.rcParams.update({'font.size': 18})

parser = argparse.ArgumentParser("Model Evaluation on Test set")
parser.add_argument(
    "--model",
    type=str,
    default="Models/ikonos-2-buildings-net-spring.pt"
)


def write_hdf5(tiles: int):
    for x, y in zip(sorted(glob("Evaluation/x*.tif")), sorted(glob("Evaluation/y*.tif"))):
        X = gdal_array.LoadFile(x) / (2**11)
        Y = gdal_array.LoadFile(y)
        X, Y = to_tiles(X, Y, 512)

        x, y = x.split('/')[-1].split('.')[-2], y.split('/')[-1].split('.')[-2]

        with h5py.File("Evaluation/test_data.hdf5", 'a') as f:
            test_group_x = f.require_group('Test/X')
            test_group_y = f.require_group('Test/Y')

            test_group_x.create_dataset(
                x,
                shape=X.shape,
                dtype='f',
                data=X,
                compression='gzip',
                compression_opts=8,
                maxshape=(
                    None,
                    4,
                    tiles,
                    tiles
                )
            )
            test_group_y.create_dataset(
                y,
                shape=Y.shape,
                dtype='i1',
                data=Y,
                compression='gzip',
                compression_opts=8,
                maxshape=(
                    None,
                    tiles,
                    tiles
                )
            )


# def expand(dataset, data, label: bool):
#     """
#     dataset: h5py dataset instance for expansion
#     data: data to fit in dataset
#     label: flag to assume dataset & data dimensions
#             <false> NOT a label: <X>
#             <true> a label: <Y>
#     """
#     if not label:
#         dataset_shape = dataset.shape
#         dataset.resize(
#             (
#                 dataset_shape[0] + data.shape[0],
#                 dataset_shape[1],
#                 dataset_shape[2],
#                 dataset_shape[3]
#             )
#         )
#         dataset[dataset_shape[0]:] = data
#     elif label:
#         dataset_shape = dataset.shape
#         dataset.resize(
#             (
#                 dataset_shape[0] + data.shape[0],
#                 dataset_shape[1],
#                 dataset_shape[2]
#             )
#         )
#         dataset[dataset_shape[0]:] = data
#     return


class Evaluate:

    AREAS = ['URBAN', 'INDUSTRIAL', 'BACKGROUND']
    C_MAT = {
        "TP": (0, 0),
        "FP": (0, 1),
        "FN": (1, 0),
        "TN": (1, 1)
    }

    def __init__(self) -> None:

        f =  h5py.File("Evaluation/test_data.hdf5", "r")
        self.X = {
            "URBAN": f['Test/X/x1'],
            "INDUSTRIAL": f['Test/X/x2'],
            "BACKGROUND": f['Test/X/x3']
        }
        self.Y = {
            "URBAN": f['Test/Y/y1'],
            "INDUSTRIAL": f['Test/Y/y2'],
            "BACKGROUND": f['Test/Y/y3']
        }
        self.cli_args = parser.parse_args(argv[1:])
        self.model = BuildingsModel(4, 16)
        self.model.load_state_dict(
            torch.load(self.cli_args.model)
        )
        self.model.eval()

        self.sample_fig, self.sample_axes = plt.subplots(1, 3, figsize=(15, 10))
        self.c_fig, self.c_ax = plt.subplots(figsize=(15, 10))
        self.c_ax.xaxis.tick_top()

    def main(self):
        for AREA in tqdm(self.AREAS):
            self.evaluate(AREA)
    
    def evaluate(self, AREA):
        c_matrix = torch.zeros(2, 2)
        for i, (image, label) in tqdm(enumerate(zip(self.X[AREA], self.Y[AREA]))):
            image, label, y_hat = self.predict(image, label)
            self.add_metrics(label, y_hat, c_matrix)
            self.write_prediction(image, label, y_hat, AREA, i)
        for i in range(1, 3):
            self.write_matrix(c_matrix, AREA, i)

    def predict(self, image, label):
        image = torch.from_numpy(image).unsqueeze_(0)
        label = torch.from_numpy(label)
        _, y_hat = self.model(image)[-1].max(-3)
        return image, label, y_hat

    def add_metrics(self, y, y_hat, c_matrix):
        c_matrix[self.C_MAT['TP']] += ((y_hat == 1) & (y == 1)).sum()
        c_matrix[self.C_MAT['FP']] += ((y_hat == 1) & (y == 0)).sum()
        c_matrix[self.C_MAT['FN']] += ((y_hat == 0) & (y == 1)).sum()
        c_matrix[self.C_MAT['TN']] += ((y_hat == 0) & (y == 0)).sum()

    def write_matrix(self, c_matrix: torch.Tensor, AREA, i):
        titles = {
            1: 'Precision',
            2: 'Recall'
        }
        self.c_ax.imshow(c_matrix / c_matrix.sum(-i, keepdim=True))
        self.c_ax.set_title(titles[i])
        self.c_ax.set_xticks([0, 1]), self.c_ax.set_xticklabels(['Buildings', 'Background'])
        self.c_ax.set_xlabel('Actual')
        self.c_ax.set_yticks([0, 1]), self.c_ax.set_yticklabels(['Buildings', 'Background'])
        self.c_ax.set_ylabel('Predictions')
        self.c_ax.set_label("Producer's Accuracy: % Correct Actual")
        self.annotate(c_matrix, self.c_ax, i)
        self.c_fig.tight_layout()
        self.c_fig.savefig(f"Evaluation/Results/{AREA}_{titles[i]}.png")
        self.c_ax.clear()
        
    def annotate(self, c_matrix, ax, i):
        pa = c_matrix / c_matrix.sum(-i, keepdim=True)
        for key in self.C_MAT:
            ax.annotate(text=f"{round(pa[self.C_MAT[key]].item()*100, 2)}%",
                        xy=(self.C_MAT[key][1]-0.1, self.C_MAT[key][0]),)

    def write_prediction(self, img, label, y_hat, AREA, i):
        self.sample_axes[0].imshow(img.squeeze(0).moveaxis(0, -1)[:, :, [2, 1, 0]])
        self.sample_axes[0].set_label("X")
        self.sample_axes[1].imshow(label.squeeze(0))
        self.sample_axes[1].set_label('Y')
        self.sample_axes[2].imshow(y_hat.squeeze(0))
        self.sample_axes[2].set_label('y_hat')
        self.sample_fig.tight_layout()
        self.sample_fig.savefig(f"Evaluation/Predictions/{AREA}{i}.png")
        for ax in self.sample_axes:
            ax.clear()


if __name__ == '__main__':
    if not os.path.exists("Evaluation/test_data.hdf5"):
        write_hdf5(512)
    Evaluate().main()