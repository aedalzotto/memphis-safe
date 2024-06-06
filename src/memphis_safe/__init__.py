from argparse import ArgumentParser
from .safe import Safe

def memphis_safe():
    parser = ArgumentParser(description="Memphis Security Anomaly Forecasting Engine")
    parser.add_argument("DATASET", help="Dataset to analyze")
    parser.add_argument("-t", "--threshold", help="Latency threshold to consider an anomaly", default=0.05, type=float)
    parser.add_argument("-r", "--rate",      help="Train/test split rate",                    default=0.75, type=float)
    parser.add_argument("-e", "--export",    help="Export partial datasets",                  action="store_true"     )
    args = parser.parse_args()

    safe = Safe(args.DATASET)
    safe.tag(args.threshold, args.export)
    safe.split(args.rate, args.export)
