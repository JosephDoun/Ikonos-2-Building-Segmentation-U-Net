import argparse


parser = argparse.ArgumentParser(description='Training')

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
parser.add_argument("--reload",
                    help='Load checkpoint and continue training',
                    default=0,
                    const=1,
                    action='store_const')
parser.add_argument("--init-scale",
                    help="The factor to multiply input channels with: "
                         "in_channels*init_scale = out_channels",
                    default=8,
                    type=int)
parser.add_argument("--checkpoint",
                    help="Path to saved checkpoint",
                    default="Checkpoints/checkpoint.pt",
                    type=str)