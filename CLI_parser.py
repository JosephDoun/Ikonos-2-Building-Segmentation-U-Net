import argparse


parser = argparse.ArgumentParser(description='Training',
                                 prog='Model Training')

parser.add_argument("--epochs",
                    help='Number of epochs for training',
                    default=100,
                    type=int)
parser.add_argument("--batch-size",
                    help="Batch size for training",
                    default=1,
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
                    help="No use. Reserved.",
                    default=0,
                    const=1,
                    action='store_const')
parser.add_argument("--monitor", "-m",
                    help="Observe activations and predictions of a sample",
                    default=0,
                    const=1,
                    action='store_const')
parser.add_argument("--l2",
                    help='L2 Regularization parameter',
                    default=0,
                    type=float)
parser.add_argument("--reload",
                    help='Load checkpoint and continue training',
                    default=0,
                    const=1,
                    action='store_const')
parser.add_argument("--init-scale",
                    help="The factor to multiply input channels with: "
                         "in_channels*init_scale = out_channels"
                         " - Controls overall U-net feature length",
                    default=8,
                    type=int)
parser.add_argument("--checkpoint", "-c",
                    help="Path to saved checkpoint",
                    default="Checkpoints/checkpoint.pt",
                    type=str)
parser.add_argument("--augmentation", "-a",
                    help="Float within [0, 1], training samples percentage"
                    "to augment",
                    default=.66,
                    type=float)
parser.add_argument("--balance-ratio", '-b',
                    type=int,
                    default=2,
                    help="Every n-th sample is negative, the rest are positive")