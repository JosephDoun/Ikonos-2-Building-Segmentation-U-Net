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
                    help="Number of background processes "
                            "for data loading",
                    default=4,
                    type=int)
parser.add_argument("--lr",
                    help='Learning rate',
                    default=0.001,
                    type=float)
parser.add_argument("--report", "-r",
                    help="""
                    Store losses on memory and produce a report graph -- Contrained by memory size.
                    Control with REPORT_RATE to minimize logs accordingly
                    """,
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
parser.add_argument("--init-scale", "-i",
                    help="""
                    The factor to initially multiply input channels with:
                    in_channels*INIT_SCALE = out_channels
                    -- Controls overall U-net feature length
                    """,
                    default=8,
                    type=int)
parser.add_argument("--checkpoint", "-c",
                    help="Path to saved checkpoint",
                    default="Checkpoints/checkpoint.pt",
                    type=str)
parser.add_argument("--augmentation", "-a",
                    help="Float within [0, 1], training samples percentage "
                    "to augment. Applies to distribution changing augmentations.",
                    default=1.0,
                    type=float)
parser.add_argument("--balance-ratio", '-b',
                    type=int,
                    default=2,
                    help="Every n-th sample is negative, the rest are positive")
parser.add_argument("--report-rate",
                    type=int,
                    default=0,
                    help="""
                    Epoch frequency to log losses for reporting.
                    Default: EPOCHS // 10
                    """)
parser.add_argument("--check-rate",
                    type=int,
                    default=0,
                    help="""
                    Make checkpoint every n epochs - For Monitor/Checkpoint.
                    Default: EPOCHS // 10
                    """)