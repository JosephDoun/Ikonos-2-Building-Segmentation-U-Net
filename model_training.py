#!/usr/bin/python3

import logging
import re
import sys
import os
from typing import List, Tuple
from cffi import model
import numpy as np
import matplotlib.pyplot as plt
from CLI_parser import parser

from torch.tensor import Tensor

from model_architecture import BuildingsModel
from torch.utils.data import DataLoader
from data_load import Buildings
from torch.nn import CrossEntropyLoss, parameter
import torch.optim as optim
import torch

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S'
)

log = logging.getLogger(__name__)
mpl = logging.getLogger('matplotlib')
log.setLevel(logging.DEBUG)
mpl.setLevel(logging.WARNING)


class Training:
    def __init__(self, argv=sys.argv[1:]) -> None:
        log.name = type(self).__name__
        parser.description = type(self).__name__
        self.epoch = 1
        self.argv = parser.parse_args(argv)
        if self.argv.reload:
            self.checkpoint = self.__load_checkpoint__()
            self.epoch = self.checkpoint['epoch']
        self.model = self.__init_model__()
        self.loss_fn = CrossEntropyLoss(weight=torch.Tensor([1., 1.]).cuda(),
                                        reduction='none')
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
            for ax in self.axes.flat:
                ax.set_axis_off()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        self.__init_scheduler__()

    def start(self):
        log.info(
            "[ Initiating Buildings U-Net Training "
            "with parameters %s ]" % self.argv
        )
        for epoch in range(self.epoch, self.argv.epochs+1):
            training_metrics = self.__train_epoch__(epoch,
                                                    self.training_loader)
            
            if epoch == 1 or not epoch % 100:
                self.model._register_hooks_()
                
            validation_metrics = self.__validate_epoch__(epoch,
                                                         self.validation_loader)
            self.__log__(epoch,
                         Training=training_metrics,
                         Validation=validation_metrics)
            if not epoch % 100:
                self.__checkpoint__(epoch)
            if not epoch % 100 or epoch == 1:
                log.info("  -- Monitoring Active: Saving sample image --")
                self.fig.savefig('Monitoring/Predictions/results_epoch_%d.png'
                                 % epoch)
                self.__monitor_layers__(epoch)
                # self.__adjust_learning_rates__(layer_abs_means)
            # Feed validation loss to the scheduler
            self.scheduler.step(validation_metrics[-1].mean())

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
            if all([self.argv.monitor and i == self.t_monitor_idx,
                    epoch == 1 or not epoch % 100]):
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
                if all([self.argv.monitor and i == self.v_monitor_idx,
                        epoch == 1 or not epoch % 100]):
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
        model = BuildingsModel(4, self.argv.init_scale)
        if torch.cuda.is_available():
            model = model.to('cuda')
        if self.argv.reload:
            model.load_state_dict(self.checkpoint['model_state'])
        return model

    def __init_optimizer__(self):
        opt = optim.Adam([
            {'params': self.model.down_1.parameters(),
             'weight_decay': 0.00000},
            {'params': self.model.down_2.parameters(),
             'weight_decay': 0.00000},
            {'params': self.model.down_3.parameters(),
             'weight_decay': 0.00001},
            {'params': self.model.down_4.parameters(),
             'weight_decay': 0.00004},
            {'params': self.model.down_5.parameters(),
             'weight_decay': 0.00004},
            {'params': self.model.up_1.parameters(),
             'weight_decay': 0.00004},
            {'params': self.model.up_2.parameters(),
             'weight_decay': 0.00001},
            {'params': self.model.up_3.parameters()},
            {'params': self.model.up_4.parameters()},
            {'params': [*self.model.z.parameters(),
                        *self.model.prob.parameters()]}
            ],
                         lr=self.argv.lr,
                         weight_decay=self.argv.l2,
                         betas=(.99, .99))
        # for i, group in enumerate(opt.param_groups):
        #         group['weight_decay'] = self.argv.l2 / (i+1)
        if self.argv.reload:
            opt.load_state_dict(self.checkpoint['optimizer_state'])
        return opt

    def __init_scheduler__(self):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              'min',
                                                              0.1,
                                                              patience=100,
                                                              verbose=True,
                                                              min_lr=5e-7)
        if self.argv.reload:
            self.scheduler.load_state_dict(self.checkpoint['scheduler_state'])

    def __init_loaders__(self):
        training_loader = DataLoader(Buildings(validation=False,
                                               aug_split=self.argv.augmentation),
                                     batch_size=self.argv.batch_size,
                                     num_workers=self.argv.num_workers,
                                     pin_memory=True,
                                     shuffle=True)
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
        log.info("  -- Writing Checkpoint --")
        torch.save(
            {
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict()
            },
            self.argv.checkpoint
        )

    def __load_checkpoint__(self) -> dict:
        checkpoint = torch.load(self.argv.checkpoint)
        return checkpoint

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
            IOU = TP / (TP + FP + FN)
            _[mode+'F'] = F_score
            _[mode+'L'] = m[2].mean()
            _[mode+'IOU'] = IOU

        log.info(
            "[ Epoch %4d of %4d :: %s Loss %2.5f - IoU: %2.5f ::"
            " %s Loss %2.5f - IoU: %2.5f / F: %2.5f ]"
            % (
                epoch,
                self.argv.epochs,
                'Training',
                _['TrainingL'],
                _['TrainingIOU'],
                'Validation',
                _['ValidationL'],
                _['ValidationIOU'],
                _['ValidationF']
            )
        )

    def __monitor_layers__(self, epoch):
        
        fig, axes = plt.subplots(3, 6, figsize=(15, 10))
        axes = axes.flatten()
        fig.suptitle("Epoch %d" % epoch)
        
        for i, (name, activation) in enumerate(self.model.activations.items()):
            axes[i].set_title(name)
            hist = np.array(
                list(
                    map(
                        lambda x: np.histogram(x, bins=20)[0],
                        # Reshape the batch sample to features * values
                        # Get histogram for each feature
                        activation[0].reshape(activation[0].size(0), -1)
                        )
                    )
                )
            
            # Normalize each row for full brightness
            hist = hist / np.expand_dims(hist.max(-1), 1)
            axes[i].imshow(hist, aspect='auto', cmap='Reds')
            
            # bins = np.array(
            #     list(
            #         map(
            #             lambda x: np.histogram(x, bins=20)[1],
            #             activation[0].reshape(activation.size(1), -1)
            #         )
            #     )
            # )
        
        fig.tight_layout()
        fig.savefig("Monitoring/Weights/%d.png" % epoch)
        self.model.activations = {}

    def __save_report__(self):
        pass

    def __monitor_sample__(self, **parameters):
        X = parameters['X'][0, [2, 1, 0]]
        Y = parameters['Y'][0]
        p = parameters['a'][0].argmax(-3)
        self.fig.suptitle('Epoch %d' % parameters['epoch'])
        self.axes[parameters['mode'], 0].clear()
        self.axes[parameters['mode'], 0].imshow(np.moveaxis(X, 0, -1))
        self.axes[parameters['mode'], 1].clear()
        self.axes[parameters['mode'], 1].imshow(Y)
        self.axes[parameters['mode'], 2].clear()
        self.axes[parameters['mode'], 2].imshow(p)
        self.axes[0, 0].set_title('X')
        self.axes[0, 0].set_xlabel('Training', rotation='vertical')
        self.axes[0, 1].set_title('Y')
        self.axes[0, 2].set_title('y_hat')
        self.axes[1, 0].set_title('X')
        self.axes[1, 0].set_ylabel('Validation', rotation='vertical')
        self.axes[1, 1].set_title('Y')
        self.axes[1, 2].set_title('y_hat')


if __name__ == "__main__":
    Training().start()
