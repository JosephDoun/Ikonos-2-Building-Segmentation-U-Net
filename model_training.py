#!/usr/bin/python3

import logging
import re
import sys
import os
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from CLI_parser import parser
from datetime import datetime

from torch import Tensor

from model_architecture import BuildingsModel
from torch.utils.data import DataLoader
from data_loader import Buildings
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import torch

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.DEBUG,
    datefmt='%H:%M:%S %b%d'
)

log = logging.getLogger(__name__)
mpl = logging.getLogger('matplotlib')
log.setLevel(logging.DEBUG)
mpl.setLevel(logging.WARNING)

run_time = datetime.now().strftime("%Y_%b%d_%H%M")


class Training:
    def __init__(self, argv=sys.argv[1:]) -> None:
        log.name = type(self).__name__
        parser.description = type(self).__name__
        self.epoch = 1
        self.argv = parser.parse_args(argv)
        self.report_rate = self.argv.report_rate or self.argv.epochs // 10
        self.check_rate = self.argv.check_rate or self.argv.epochs // 10

        if self.argv.reload:
            self.checkpoint = self.__load_checkpoint__()
            self.epoch = self.checkpoint['epoch']

        self.model = self.__init_model__()
        self.loss_fn = CrossEntropyLoss(reduction='none')
        self.optimizer = self.__init_optimizer__()
        self.training_loader, self.validation_loader = self.__init_loaders__()

        if self.argv.report:
            """
            # TODO
            Reporting should be eventually switched to
            Tensorboard for efficiency and simplicity 
            """
            self._init_report_figure_()

        if self.argv.monitor:
            self._init_monitor_figures()

        self.__init_scheduler__()

    def _init_report_figure_(self):
        self.report = {
            'total_training_loss': [],
            'pos_training_loss': [],
            'neg_training_loss': [],
            'total_validation_loss': [],
            'pos_validation_loss': [],
            'neg_validation_loss': []
        }
        self.r_fig, self.r_axes = plt.subplots(1, 2, figsize=(15, 10))
        self.r_fig.suptitle(
            f"""
            Training Report - Balance: {self.argv.balance_ratio} 
            {self.training_loader.dataset.augmentations}"""
        )
        for ax in self.r_axes:
            ax.set_ylabel("Cross Entropy Loss")
            ax.set_xlabel("Relevant Epochs")
            ax.set_facecolor((.9, .9, .9))

    def _init_monitor_figures(self):
        self.t_monitor_idx = torch.randint(
            self.training_batches, (1,)
        )
        self.v_monitor_idx = torch.randint(
            self.validation_batches, (1,)
        )
        self.pred_fig, self.pred_axes = plt.subplots(2, 3, figsize=(15, 10))
        self.act_fig, self.act_axes = plt.subplots(3, 6, figsize=(15, 10))
        self.act_axes = self.act_axes.flatten()

        for ax in self.pred_axes.flat:
            ax.set_axis_off()
            ax.get_xaxis().set_visible(False)

    def start(self):
        log.info(
            "[ Initiating Buildings U-Net Training "
            "with parameters %s ]" % self.argv
        )
        for epoch in range(self.epoch, self.argv.epochs+1):
            training_metrics = self.__train_epoch__(epoch,
                                                    self.training_loader)

            if epoch == 1 or not epoch % self.check_rate:
                # Register hooks to capture validation
                # Hooks are removed on execution to preserve memory.
                # Repeat at next checkpoint
                self.model._register_hooks_()

            validation_metrics = self.__validate_epoch__(epoch,
                                                         self.validation_loader)
            self.__log__(epoch,
                         T=training_metrics,
                         V=validation_metrics)

            if not epoch % self.check_rate:

                self.__checkpoint__(epoch)

            if (not epoch % self.check_rate or epoch == 1) and self.argv.monitor:

                log.info("  -- Monitoring Active: Saving sample image --")
                self.pred_fig.savefig('Monitoring/Predictions/results_epoch_%d.png'
                                      % epoch)
                self.__monitor_layers__(epoch)
                # self.__adjust_learning_rates__(layer_abs_means)
            # Feed loss to scheduler
            # self.scheduler.step(training_metrics[-1].mean())

    def __train_epoch__(self, epoch, training_loader):
        self.model.train()
        metrics = torch.zeros(
            3,
            len(training_loader.dataset),
            training_loader.dataset[0][0].size(-2),
            training_loader.dataset[0][0].size(-1),
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
                    epoch == 1 or not epoch % self.report_rate]):
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
                512,
                512,
                device='cuda'
            )
            for i, (X, Y) in enumerate(validation_loader):
                X = X.to('cuda')
                Y = Y.to('cuda')
                z, a = self.model(X)
                loss, _loss = self.__compute_loss__(z, Y)
                self._compute_metrics_(i, a, Y, _loss, metrics)

                if all([self.argv.monitor and i == self.v_monitor_idx,
                        epoch == 1 or not epoch % self.check_rate]):

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

    def _compute_metrics_(self, i, a, Y, loss, metrics: Tensor):
        _, predictions = a.max(-3)
        idx = i * self.argv.batch_size
        _ = slice(idx, idx+Y.size(0))
        metrics[0, _] = predictions
        metrics[1, _] = Y
        metrics[2, _] = loss

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
             'weight_decay': 0.00000},
            {'params': self.model.down_4.parameters(),
             'weight_decay': 0.00000},
            {'params': self.model.down_5.parameters(),
             'weight_decay': 0.00000},
            {'params': self.model.up_1.parameters(),
             'weight_decay': 0.00000},
            {'params': self.model.up_2.parameters(),
             'weight_decay': 0.00000},
            {'params': self.model.up_3.parameters()},
            {'params': self.model.up_4.parameters()},
            {'params': [*self.model.z.parameters(),
                        *self.model.prob.parameters()]}
        ],
            lr=self.argv.lr,
            weight_decay=self.argv.l2,
            betas=(.99, .99))

        if self.argv.reload:
            opt.load_state_dict(self.checkpoint['optimizer_state'])
            for p in opt.param_groups:
                p['lr'] = self.argv.lr
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
                                               aug_split=self.argv.augmentation,
                                               ratio=self.argv.balance_ratio),
                                     batch_size=self.argv.batch_size,
                                     num_workers=self.argv.num_workers,
                                     pin_memory=True,
                                     shuffle=True)
        validation_loader = DataLoader(Buildings(validation=True,
                                                 ratio=self.argv.balance_ratio),
                                       batch_size=self.argv.batch_size,
                                       num_workers=self.argv.num_workers,
                                       pin_memory=True)
        self.training_batches = -(-len(training_loader.dataset) //
                                  self.argv.batch_size)
        self.validation_batches = -(-len(validation_loader.dataset) //
                                    self.argv.batch_size)

        if self.report_rate:
            # Index a sample to register active augmentations
            # To use for report figure title
            training_loader.dataset[0]

        return training_loader, validation_loader

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
        if self.report:
            self._report_()

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
            _[mode+'Lpos'] = m[2][m[1] == 1].mean()
            _[mode+'Lneg'] = m[2][m[1] == 0].mean()

        log.info(
            "[ E%4d /%4d :: %s L %2.3f Lpos %2.3f Lneg %2.3f - IoU %2.3f ::"
            " %s L %2.3f Lpos %2.3f Lneg %2.3f - IoU %2.3f / F %2.3f ]"
            % (
                epoch,
                self.argv.epochs,
                '(T)',
                _['TL'],
                _['TLpos'],
                _['TLneg'],
                _['TIOU'],
                '(V)',
                _['VL'],
                _['VLpos'],
                _['VLneg'],
                _['VIOU'],
                _['VF']
            )
        )

        if self.argv.report and (not epoch % self.report_rate
                                 or epoch == 1):
            self.report['total_training_loss'].append(
                _['TL']
            )
            self.report['total_validation_loss'].append(
                _['VL']
            )
            self.report['pos_training_loss'].append(
                _['TLpos']
            )
            self.report['neg_training_loss'].append(
                _['TLneg']
            )
            self.report['pos_validation_loss'].append(
                _['VLpos']
            )
            self.report['neg_validation_loss'].append(
                _['VLneg']
            )

    def __monitor_layers__(self, epoch):

        self.act_axes = self.act_axes.flatten()
        self.act_fig.suptitle("Epoch %d" % epoch)

        for i, (name, activation) in enumerate(self.model.activations.items()):
            self.act_axes[i].set_title(name)
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
            self.act_axes[i].imshow(hist, aspect='auto', cmap='Reds')

            # bins = np.array(
            #     list(
            #         map(
            #             lambda x: np.histogram(x, bins=20)[1],
            #             activation[0].reshape(activation.size(1), -1)
            #         )
            #     )
            # )

        self.act_fig.tight_layout()
        self.act_fig.savefig("Monitoring/Activations/%d.png" % epoch)

        self.model.activations = {}

        for ax in self.act_axes:
            ax.clear()

    def __monitor_sample__(self, **parameters):
        X = parameters['X'][0, [2, 1, 0]]
        Y = parameters['Y'][0]
        p = parameters['a'][0].argmax(-3)
        self.pred_fig.suptitle('Epoch %d' % parameters['epoch'])
        self.pred_axes[parameters['mode'], 0].clear()
        self.pred_axes[parameters['mode'], 0].imshow(np.moveaxis(X, 0, -1))
        self.pred_axes[parameters['mode'], 1].clear()
        self.pred_axes[parameters['mode'], 1].imshow(Y)
        self.pred_axes[parameters['mode'], 2].clear()
        self.pred_axes[parameters['mode'], 2].imshow(p)
        self.pred_axes[0, 0].set_title('X')
        self.pred_axes[0, 0].set_ylabel('Training', rotation='vertical')
        self.pred_axes[0, 1].set_title('Y')
        self.pred_axes[0, 2].set_title('y_hat')
        self.pred_axes[1, 0].set_title('X')
        self.pred_axes[1, 0].set_ylabel('Validation', rotation='vertical')
        self.pred_axes[1, 1].set_title('Y')
        self.pred_axes[1, 2].set_title('y_hat')
        for ax in self.pred_axes.flatten():
            ax.set_yticks([])

    def _report_(self):
        self.r_axes[0].plot(self.report['total_training_loss'],
                            'k--', label='total_training loss')
        self.r_axes[0].plot(self.report['total_validation_loss'],
                            'k-', label='total_validation loss')
        self.r_axes[0].legend()
        self.r_axes[1].plot(self.report['pos_training_loss'],
                            'k--', label='pos_train_loss')
        self.r_axes[1].plot(self.report['pos_validation_loss'],
                            'k-', label='pos_val_loss')
        self.r_axes[1].plot(self.report['neg_training_loss'],
                            label='neg_train_loss', color=(.5, .5, .5),
                            ls='--')
        self.r_axes[1].plot(self.report['neg_validation_loss'],
                            label='neg_val_loss', color=(.5, .5, .5))
        self.r_axes[1].legend()
        os.makedirs("Reports", exist_ok=True)
        log.info(
            "  -- Reporting Active: Saving Report -- "
        )
        self.r_fig.savefig(
            f"Reports/report_b{self.argv.balance_ratio}_{run_time}.png")

        for ax in self.r_axes:
            for line in ax.lines:
                line.remove()


if __name__ == "__main__":
    Training().start()
