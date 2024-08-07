from argparse import ArgumentParser
from .model.xgmodel import XGModel
from .model.safe import Safe
from .preprocess import Preprocess

def memphis_safe():
    parser = ArgumentParser(description="Memphis Security Anomaly Forecasting Engine")
    subparsers = parser.add_subparsers(dest="option")

    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess dataset")
    preprocess_parser.add_argument("DATASET", help="Dataset to preprocess")
    preprocess_parser.add_argument("-t", "--threshold", help="Latency threshold to consider an anomaly", default=0.05, type=float)
    preprocess_parser.add_argument("-r", "--rate",      help="Train/test split rate",                    default=0.75, type=float)

    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("TRAIN",             help="Train dataset to train model"                    )
    train_parser.add_argument("-c", "--cross-val", help="Cross-validation subsets",    default=5, type=int)
    train_parser.add_argument("-e", "--export",    help="Export partial datasets",     action="store_true")

    test_parser = subparsers.add_parser("test", help="Test model")
    test_parser.add_argument("MODEL",   help="Model to test")
    test_parser.add_argument("DATASET", help="Dataset to test")
    test_parser.add_argument("-t", "--threshold", help="Latency threshold to consider an anomaly", default=0.05, type=float)
    test_parser.add_argument("-e", "--export",    help="Export partial datasets",                  action="store_true"     )

    args = parser.parse_args()
    if args.option == "preprocess":
        data = Preprocess(args.DATASET)
        data.preprocess(args.threshold, args.rate)
    elif args.option == "train":
        model = XGModel(args.TRAIN)
        model.train(args.cross_val, args.export)
    elif args.option == "test":
        model = Safe(args.MODEL, args.DATASET)
        model.test(args.threshold)
