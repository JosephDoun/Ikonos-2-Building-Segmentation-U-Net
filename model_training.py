#!/usr/bin/python3

import logging
import argparse
import sys
import os
from typing import List
import matplotlib.pyplot as plt

from torch.tensor import Tensor

from buildings_unet import BuildingsModel
from torch.utils.data import DataLoader
from data_load import Buildings
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import torch

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class Training:
    def __init__(self, argv=sys.argv[1:]) -> None:
        log.name = type(self).__name__
        parser = argparse.ArgumentParser(description=type(self).__name__)
        parser.add_argument("--epochs",
                            help='Number of epochs for training',
                            default=100,
                            type=int)
        parser.add_argument("--batch-size",
                            help="Batch size for training",
                            default=2,
                            type=int)
        parser.add_argument("--num-workers",
                            help="Number of background proceses"
                                 "for data loading",
                            default=4,
                            type=int)
        parser.add_argument("--lr",
                            help='Learning rate',
                            default=0.03,
                            type=float)
        parser.add_argument("--report", "-r",
                            help="Produce final graph report",
                            default=0,
                            const=1,
                            action='store_const')
        parser.add_argument("--debug", "-d",
                            help="Print per batch losses",
                            default=0,
                            const=1,
                            action='store_const')
        parser.add_argument("--monitor", "-m",
                            help="Plot and monitor a random validation sample",
                            default=0,
                            const=1,
                            action='store_const')
        parser.add_argument("--l2",
                            help='L2 Regularization parameter',
                            default=0,
                            type=float)
        parser.add_argument("--dropout",
                            help='L2 Regularization parameter',
                            default=0,
                            type=float)
        self.argv = parser.parse_args(argv)
        self.model = self.__init_model__()
        self.loss_fn = CrossEntropyLoss(reduction='none')
        self.optimizer = self.__init_optimizer__()
        self.training_loader, self.validation_loader = self.__init_loaders__()
        if self.argv.report:
            self.report = {}
        if self.argv.monitor:
            self.t_monitor_idx = torch.randint(
                self.training_batches, (1,)
            )
            self.v_monitor_idx = torch.randint(
                self.validation_batches, (1,)
            )
            self.fig, self.axes = plt.subplots(2, 3)
            plt.ion()
            plt.show()

    def start(self):
        log.info(
            "Initiating Buildings U-Net training "
            "with parameters %s" % self.argv
        )
        for epoch in range(1, self.argv.epochs+1):
            training_metrics = self.__train_epoch__(epoch,
                                                    self.training_loader)
            validation_metrics = self.__validate_epoch__(epoch,
                                                         self.validation_loader)
            self.__log__(epoch,
                         Training=training_metrics,
                         Validation=validation_metrics)

    def __train_epoch__(self, epoch, training_loader):
        self.model.train()
        metrics = torch.zeros(
            3,
            len(training_loader.dataset),
            512, 512,
            device='cuda'
        )
        for i, (X, Y) in enumerate(training_loader):
            self.optimizer.zero_grad()
            X = X.to('cuda', non_blocking=True)
            Y = Y.to('cuda', non_blocking=True)
            z, a = self.model(X)
            loss, _loss = self.__compute_loss__(z, Y)
            loss.backward()
            self.optimizer.step()
            self._compute_metrics_(i, a, Y, _loss, metrics)
            if self.argv.monitor and i == self.t_monitor_idx:
                self.__monitor_sample__(epoch=epoch,
                                        X=X.cpu().detach().numpy(),
                                        Y=Y.cpu().detach().numpy(),
                                        a=a.cpu().detach().numpy(),
                                        mode=0)
        return metrics.to('cpu')

    def __validate_epoch__(self, epoch, validation_loader):
        with torch.no_grad():
            self.model.eval()
            metrics = torch.zeros(
                3,
                len(validation_loader.dataset),
                512, 512,
                device='cuda'
            )
            for i, (X, Y) in enumerate(validation_loader):
                X = X.to('cuda')
                Y = Y.to('cuda')
                z, a = self.model(X)
                loss, _loss = self.__compute_loss__(z, Y)
                self._compute_metrics_(i, a, Y, _loss, metrics)
                if self.argv.monitor and i == self.v_monitor_idx:
                    self.__monitor_sample__(epoch=epoch,
                                            X=X.cpu().detach().numpy(),
                                            Y=Y.cpu().detach().numpy(),
                                            a=a.cpu().detach().numpy(),
                                            mode=1)
            return metrics.to('cpu')

    def __compute_loss__(self, z, Y):
        """
            :param z: non-activated output
            :param Y: targets
        """
        loss = self.loss_fn(z, Y)
        return loss.mean(), loss

    def __init_model__(self):
        model = BuildingsModel(4, 8)
        if torch.cuda.is_available():
            model = model.to('cuda')
        return model

    def __init_optimizer__(self, l=0):
        return optim.AdamW(self.model.parameters(),
                           lr=self.argv.lr,
                           weight_decay=self.argv.l2)

    def __init_loaders__(self):
        training_loader = DataLoader(Buildings(validation=False),
                                     batch_size=self.argv.batch_size,
                                     num_workers=self.argv.num_workers,
                                     pin_memory=True)
        validation_loader = DataLoader(Buildings(validation=True),
                                       batch_size=self.argv.batch_size,
                                       num_workers=self.argv.num_workers,
                                       pin_memory=True)
        self.training_batches = -(-len(training_loader.dataset) //
                                  self.argv.batch_size)
        self.validation_batches = -(-len(validation_loader.dataset) //
                                    self.argv.batch_size)
        return training_loader, validation_loader

    def _compute_metrics_(self, i, a, Y, loss, metrics: Tensor):
        _, predictions = a.max(-3)
        idx = i * self.argv.batch_size
        _ = slice(idx, idx+Y.size(0))
        metrics[0, _] = predictions
        metrics[1, _] = Y
        metrics[2, _] = loss

    def __checkpoint__(self, epoch):
        torch.save(
            {
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict()
            }, 'checkpoints/checkpoint.pt'
        )

    def __load_checkpoint__(self):
        pass

    def __log__(self, epoch, **metrics):
        _ = {}
        for mode, m in metrics.items():
            TP = ((m[0] == 1) & (m[1] == 1)).sum()
            FP = ((m[0] == 1) & (m[1] == 0)).sum()
            TN = ((m[0] == 0) & (m[1] == 0)).sum()
            FN = ((m[0] == 0) & (m[1] == 1)).sum()
            P = TP / (TP + FP)
            R = TP / (TP + FN)
            F_score = 2 * (P * R) / (P + R)
            _[mode+'F'] = F_score
            _[mode+'L'] = m[2].mean()

        log.info(
            " [ Epoch %4d of %4d :: %s Loss %2.3f - F Measure: %2.3f ::"
            " %s Loss %2.3f - F Measure: %2.3f ]"
            % (
                epoch,
                self.argv.epochs,
                'Training',
                _['TrainingL'],
                _['TrainingF'],
                'Validation',
                _['ValidationL'],
                _['ValidationF']
            )
        )

    def __monitor_sample__(self, **parameters):
        if not parameters['epoch'] % 50:
            X = parameters['X'][0, :3]
            Y = parameters['Y'][0]
            p = parameters['a'][0].max(-3)
            self.axes[parameters['mode'], 0].clear()
            self.axes[parameters['mode'], 0].imshow(X.moveaxis(0, -1))
            self.axes[parameters['mode'], 1].clear()
            self.axes[parameters['mode'], 1].imshow(Y)
            self.axes[parameters['mode'], 2].clear()
            self.axes[parameters['mode'], 2].imshow(p)


if __name__ == "__main__":
    Training().start()
