#!/usr/bin/python3

import logging
import argparse
import sys
import os
from typing import List

from torch.tensor import Tensor

from buildings_unet import BuildingsModel
from torch.utils.data import DataLoader
from data_load import Buildings
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class TrainingEnvironment:
    def __init__(self, argv=sys.argv[1:]) -> None:
        parser = argparse.ArgumentParser(description=type(self).__name__)
        parser.add_argument("--epochs",
                            help='Number of epochs for training',
                            default=100,
                            type=int)
        parser.add_argument("--batch-size",
                            help="Batch size for training",
                            default=32,
                            type=int)
        parser.add_argument("--num-workers",
                            help="Number of background proceses"
                                 "for data loading",
                            default=8,
                            type=int)
        parser.add_argument("--lr",
                            help='Learning rate',
                            default=0.003,
                            type=float)
        parser.add_argument("--report",
                            help="Produce final graph report",
                            default=0,
                            const=1,
                            action='store_const')
        self.argv = parser.parse_args(argv)
        self.model = self.__init_model__()
        self.loss_fn = CrossEntropyLoss(reduction='none')
        self.optimizer = self.__init_optimizer__()
        if self.argv.report:
            self.report = {}
        
    def start(self):
        log.info(
            "Initiating Buildings U-Net training "
            "with parameters %s" % self.argv
        )
        training_loader, validation_loader = self.__init_loaders__()
        for epoch in range(1, self.argv.epochs+1):
            log.info("Epoch %5d of %5d:" % (epoch, self.argv.epochs))
            training_metrics = self.__train_epoch__(training_loader)
            validation_metrics = self.__validate_epoch__(validation_loader)
            self.__log__(Training=training_metrics,
                         Validation=validation_metrics)
        
    def __train_epoch__(self, training_loader):
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
        return metrics.to('cpu')
        
    def __validate_epoch__(self, validation_loader):
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
            return metrics.to('cpu')
        
    def __compute_loss__(self, z, Y):
        """
            :param z: non-activated output
            :param Y: targets
        """
        loss = self.loss_fn(z, Y)
        return loss.mean(), loss
        
    def __init_model__(self):
        model = BuildingsModel(4, 16)
        if torch.cuda.is_available():
            model = model.to('cuda')
        return model        
        
    def __init_optimizer__(self):
        return Adam(self.model.parameters(), lr=self.argv.lr)
    
    def __init_loaders__(self):
        training_loader = DataLoader(Buildings(validation=False),
                                     batch_size=self.argv.batch_size,
                                     num_workers=self.argv.num_workers,
                                     pin_memory=True)
        validation_loader = DataLoader(Buildings(validation=True),
                                       batch_size=self.argv.batch_size,
                                       num_workers=self.argv.num_workers,
                                       pin_memory=True)
        return training_loader, validation_loader
    
    def _compute_metrics_(self, i, a, Y, loss, metrics: Tensor):
        _, predictions = a.max(-3)
        idx = i * self.argv.batch_size
        _ = slice(idx,idx+Y.size(0))
        metrics[0, _] = predictions
        metrics[1, _] = Y
        metrics[2, _] = loss
    
    def _compute_accuracies(self, training_metrics, validation_metrics=None):
        pass
    
    def __log__(self, **metrics):
        for mode, m in metrics.items():
            TP = ((m[0] == 1) & (m[1] == 1)).sum()
            FP = ((m[0] == 1) & (m[1] == 0)).sum()
            TN = ((m[0] == 0) & (m[1] == 0)).sum()
            FN = ((m[0] == 0) & (m[1] == 1)).sum()
            P = TP / (TP + FP)
            R = TP / (TP + FN)
            F_score = 2 * P * R / (P + R)
            log.info(
                "%s Loss %f - F1 Score: %f" % (
                    mode,
                    metrics[2].mean(),
                    F_score
                )
            )
        
        
if __name__ == "__main__":
    TrainingEnvironment().start()
    