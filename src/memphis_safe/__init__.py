from argparse import ArgumentParser
from .model.xgmodel import XGModel
from .model.safe import Safe
from .eval import Eval

def memphis_safe():
    parser = ArgumentParser(description="Memphis Security Anomaly Forecasting Engine")
    subparsers = parser.add_subparsers(dest="option")

    train_parser = subparsers.add_parser("train",     help="Train model")
    train_parser.add_argument("TRAIN",                help="Train dataset to train model"                       )

    linear_parser = subparsers.add_parser("linear",     help="Train model (linear)")
    linear_parser.add_argument("TRAIN",                help="Train dataset to train model"                       )

    avg_parser = subparsers.add_parser("avg",     help="Train model (avg)")
    avg_parser.add_argument("TRAIN",                help="Train dataset to train model"                       )

    test_parser = subparsers.add_parser("test", help="Test model")
    test_parser.add_argument("MODEL",   help="Model to test")
    test_parser.add_argument("DATASET", help="Dataset to test")
    test_parser.add_argument("-t", "--threshold", help="Latency threshold to consider an anomaly", default=24, type=float)

    eval_parser = subparsers.add_parser("eval", help="Show real-time detection metrics")
    eval_parser.add_argument("TESTCASE", help="Path to testcase to extract application time")
    eval_parser.add_argument("DATASET", help="Dataset to eval")

    args = parser.parse_args()
    if args.option == "train":
        model = XGModel(args.TRAIN)
        model.train()
    elif args.option == "linear":
        model = XGModel(args.TRAIN)
        model.linear()
    elif args.option == "avg":
        model = XGModel(args.TRAIN)
        model.avg()
    elif args.option == "test":
        model = Safe(args.MODEL, args.DATASET)
        model.test(args.threshold)
    elif args.option == "eval":
        rtd = Eval(args.TESTCASE, args.DATASET)
        rtd.eval()
    else:
        parser.print_usage()
