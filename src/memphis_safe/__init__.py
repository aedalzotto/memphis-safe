from argparse import ArgumentParser
from .preprocess.data import Data
from .model.xgmodel import XGModel

def memphis_safe():
    parser = ArgumentParser(description="Memphis Security Anomaly Forecasting Engine")
    parser.add_argument("DATASET", help="Dataset to analyze")
    parser.add_argument("-t", "--threshold", help="Latency threshold to consider an anomaly", default=0.05, type=float)
    parser.add_argument("-r", "--rate",      help="Train/test split rate",                    default=0.75, type=float)
    parser.add_argument("-e", "--export",    help="Export partial datasets",                  action="store_true"     )
    parser.add_argument("-c", "--cross-val", help="Cross-validation subsets",                 default=5,    type=int  )
    args = parser.parse_args()

    data = Data(args.DATASET)
    train = data.wrangle(args.threshold, args.rate, args.export)

    model = XGModel(train)
    model.train(args.cross_val, args.export)