from argparse import ArgumentParser
from .preprocess.tag import Tag
from .preprocess.split import Split
from .preprocess.clean import Clean
from .model.xgmodel import XGModel
from .model.safe import Safe

def memphis_safe():
    parser = ArgumentParser(description="Memphis Security Anomaly Forecasting Engine")
    subparsers = parser.add_subparsers(dest="option")

    tag_parser = subparsers.add_parser("tag", help="Tag dataset with anomalies")
    tag_parser.add_argument("DATASET", help="Dataset to analyze")
    tag_parser.add_argument("-t", "--threshold", help="Latency threshold to consider an anomaly", default=0.05, type=float)
    tag_parser.add_argument("-e", "--export",    help="Export partial datasets",                  action="store_true"     )

    split_parser = subparsers.add_parser("split", help="Split tagged dataset")
    split_parser.add_argument("DATASET", help="Tagged dataset to analyze")
    split_parser.add_argument("-r", "--rate",      help="Train/test split rate",   default=0.75, type=float)
    split_parser.add_argument("-e", "--export",    help="Export partial datasets", action="store_true"     )

    clean_parser = subparsers.add_parser("clean", help="Clean split datasets")
    clean_parser.add_argument("TRAIN",          help="Train dataset to clean"                       )
    clean_parser.add_argument("TEST",           help="Test dataset to clean"                        )
    clean_parser.add_argument("-e", "--export", help="Export partial datasets", action="store_true" )

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
    if args.option == "tag":
        data = Tag(args.DATASET)
        data.tag(args.threshold, args.export)
    elif args.option == "split":
        data = Split(args.DATASET)
        data.split(args.rate, args.export)
    elif args.option == "clean":
        data = Clean(args.TRAIN, args.TEST)
        data.clean(args.export)
    elif args.option == "train":
        model = XGModel(args.TRAIN)
        model.train(args.cross_val, args.export)
    elif args.option == "test":
        model = Safe(args.MODEL, args.DATASET)
        model.test(args.threshold)
