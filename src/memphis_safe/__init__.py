from argparse import ArgumentParser
from .model.xgmodel import XGModel
from .model.safe import Safe
from .eval import Eval
from .joiner import Joiner
from .safe_avg import SafeAvg

def memphis_safe():
    parser = ArgumentParser(description="Memphis Security Anomaly Forecasting Engine")
    subparsers = parser.add_subparsers(dest="option")

    train_parser = subparsers.add_parser("train",     help="Train model")
    train_parser.add_argument("TRAIN",                help="Train dataset to train model"                       )

    linear_parser = subparsers.add_parser("linear",     help="Train model (linear)")
    linear_parser.add_argument("TRAIN",                help="Train dataset to train model"                       )

    avg_parser = subparsers.add_parser("avg",     help="Train model (avg)")
    avg_parser.add_argument("TRAIN",              help="Train dataset to train model"                       )

    test_parser = subparsers.add_parser("test", help="Test model")
    test_parser.add_argument("MODEL",   help="Model to test")
    test_parser.add_argument("DATASET", help="Dataset to test")
    test_parser.add_argument("-t", "--threshold", help="Latency threshold to consider an anomaly", default=50, type=int)

    test_avg_parser = subparsers.add_parser("test-avg", help="Test model")
    test_avg_parser.add_argument("TRAIN",   help="Train dataset to extract averages")
    test_avg_parser.add_argument("DATASET", help="Dataset to test")
    test_avg_parser.add_argument("-t", "--threshold", help="Latency threshold to consider an anomaly", default=50, type=int)

    eval_parser = subparsers.add_parser("eval", help="Show real-time detection metrics")
    eval_parser.add_argument("DATASET", help="Dataset to eval")
    eval_parser.add_argument("TESTCASE", nargs='?', default=None, help="Path to testcase to extract application time")

    join_parser = subparsers.add_parser("join", help="Join Mapp datasets (test+rtd)")
    join_parser.add_argument("TEST", help="Test dataset (w/o threat)")
    join_parser.add_argument("RTD", help="RTD dataset (w/ threat)")
    join_parser.add_argument("-t", "--threshold", help="Latency threshold to consider an anomaly", default=10, type=int)

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
    elif args.option == "join":
        joiner = Joiner(args.TEST, args.RTD, args.threshold)
        joiner.join()
    elif args.option == "test-avg":
        model = SafeAvg(args.TRAIN, args.DATASET, args.threshold)
    else:
        parser.print_usage()
