#!/usr/bin/python3

import logging
import argparse
import sys
import os

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
                            default=100,
                            type=int)
        parser.add_argument("--num-workers",
                            help="Number of background proceses"
                                 "for data loading",
                            default=8,
                            type=int)
        self.argv = parser.parse_args(argv)
        self.model = self.init_model()
        self.optimizer = self.init_optimizer()
        
    def start(self):
        log.info(
            "Initiating Buildings U-Net training "
            "with parameters %s" % self.argv
        )
        training_loader, validation_loader = self._init_loaders()
        for epoch in range(1, self.argv.epochs+1):
            log.info("message %s" % epoch)
            training_metrics = self.__train_epoch__(training_loader)
            validation_metrics = self.__validate_epoch__(validation_loader)
        
    def __train_epoch__(self, training_loader) -> Tensor:
        self.model.train()
        
        for X, Y in training_loader:
            X = X.to('cuda', non_blocking=True)
            Y = Y.to('cuda', non_blocking=True)
            z, a = self.model(X)
            loss = self.__compute_loss__(z, Y)
        
    def __validate_epoch__(self, validation_loader) -> Tensor:
        ...
        
    def __compute_loss__(self):
        pass
        
    def __init_model__(self):
        model = BuildingsModel(4, 16)
        if torch.cuda.is_available():
            model = model.to('cuda')
        return model        
        
    def __init_optimizer__(self):
        return Adam(self.model.parameters(), lr=0.002)
    
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
    
    def _compute_accuracy_(self):
        pass
    
    def __log__(self):
        pass
        
        
if __name__ == "__main__":
    TrainingEnvironment().start()