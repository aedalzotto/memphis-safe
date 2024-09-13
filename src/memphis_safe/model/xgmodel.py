from numpy import sqrt
from pandas import read_csv
from yaspin import yaspin
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from pickle import dump

class XGModel:
    def __init__(self, name, estimators, depth):
        with yaspin(text="Loading train dataset...") as spinner:
            self.name = name
            self.estimators = estimators
            self.depth = depth
            self.X    = read_csv(name)
            spinner.ok()

        self.model = XGBRegressor(objective="reg:squarederror", n_estimators=estimators, max_depth=depth)

    def train(self, cv_k):
        print("\n", end="")
        with yaspin(text="Training model...") as spinner:
            self.y    = self.X[["total_time"]]
            self.X.drop(columns=["total_time"], inplace=True)
            scores = cross_val_score(self.model, self.X, self.y, scoring='neg_mean_squared_error', cv=cv_k)
            rmse = sqrt(-scores)
            self.model.fit(self.X, self.y)
            spinner.ok()

        print("Cross-validation mean RMSE: {}".format(round(rmse.mean(), 3)))
        print("Cross-validation RMSE std dev: {}".format(round(rmse.std(), 3)))
        print("{}-fold cross-validation scores:".format(cv_k))
        print(rmse)

        print("", end="\n")
        with yaspin(text="Exporting model...") as spinner:
            tokens   = self.name.split("/")
            name     = tokens[-1].split(".")[-2]
            name_tks = name.split("_")
            name     = "_".join(name_tks[:-1])
            path   = "."
            if len(tokens) > 1:
                path   = "/".join(tokens[0:-1])
            full_name = "{}/{}_e{}_d{}".format(path, name, self.estimators, self.depth)
            self.model.save_model("{}_model.json".format(full_name))
            dump(self.model, open("{}_model.pkl".format(full_name), "wb"))
            spinner.ok()

        print("Model exported to {}_model.{{json,pkl}}".format(full_name))
