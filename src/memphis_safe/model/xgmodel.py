from numpy import sqrt
from pandas import read_csv, get_dummies
from yaspin import yaspin
from xgboost import XGBRegressor, to_graphviz,plot_tree
from sklearn.model_selection import cross_val_score
from .neg_mape import neg_mean_percentage_error

class XGModel:
    def __init__(self, name):
        self.name       = name

        with yaspin(text="Loading train dataset...") as spinner:
            self.X         = read_csv(name)
            self.y         = self.X[["latency"]]
            self.X         = self.X[["rel_time", "prod", "cons", "hops", "size"]]
            self.X["prod"] = self.X["prod"].astype("category")
            self.X["cons"] = self.X["cons"].astype("category")
            self.X         = get_dummies(self.X, columns=["prod", "cons"])
            spinner.ok()

    def train(self, cv_k):
        n_estimators = 100
        max_depth = 6
        print("\n", end="")
        with yaspin(text="Training base model...") as spinner:
            model = XGBRegressor(base_score=50, n_estimators=n_estimators, max_depth=max_depth)
            scores = cross_val_score(model, self.X, self.y, scoring=neg_mean_percentage_error, cv=cv_k)
            mape = -scores
            spinner.ok()

        print("Base cross-validation mean MAPE: {}".format(round(mape.mean(), 3)))
        print("Base cross-validation MAPE std dev: {}".format(round(mape.std(), 3)))
        print("{}-fold base cross-validation scores:".format(cv_k))
        print(mape)

        with yaspin(text="Finding smallest model...") as spinner:
            last_score            = mape.mean()
            max_score             = last_score * 1.1
            last_bad_n_estimators = 0
            while True:
                if last_score < max_score:
                    last_good_n_estimators = n_estimators
                else:
                    last_bad_n_estimators = n_estimators
                n_estimators = round((last_good_n_estimators+last_bad_n_estimators) / 2)
                if n_estimators in [last_good_n_estimators, last_bad_n_estimators]:
                    break
                print("Trying {} estimators".format(n_estimators))
                model = XGBRegressor(base_score=50, n_estimators=n_estimators, max_depth=max_depth)
                scores = cross_val_score(model, self.X, self.y, scoring=neg_mean_percentage_error, cv=cv_k)
                mape = -scores
                last_score = mape.mean()

            model = XGBRegressor(base_score=50, n_estimators=last_good_n_estimators, max_depth=max_depth)
            scores = cross_val_score(model, self.X, self.y, scoring=neg_mean_percentage_error, cv=cv_k)
            mape = -scores
            last_score = mape.mean()
            last_bad_max_depth = 0

            while True:
                if last_score < max_score:
                    last_good_max_depth = max_depth
                else:
                    last_bad_max_depth = max_depth
                max_depth = round((last_good_max_depth+last_bad_max_depth) / 2)
                if max_depth in [last_good_max_depth, last_bad_max_depth]:
                    break
                print("Trying depth {}".format(max_depth))
                model = XGBRegressor(base_score=50, n_estimators=n_estimators, max_depth=max_depth)
                scores = cross_val_score(model, self.X, self.y, scoring=neg_mean_percentage_error, cv=cv_k)
                mape = -scores
                last_score = mape.mean()

            model = XGBRegressor(base_score=50, n_estimators=last_good_n_estimators, max_depth=last_good_max_depth)
            scores = cross_val_score(model, self.X, self.y, scoring=neg_mean_percentage_error, cv=cv_k)
            mape = -scores
            model.fit(self.X, self.y)
            spinner.ok()

        print("n_estimators={}; max_depth={}".format(last_good_n_estimators, last_good_max_depth))
        print("Base cross-validation mean MAPE: {}".format(round(mape.mean(), 3)))
        print("Base cross-validation MAPE std dev: {}".format(round(mape.std(), 3)))
        print("{}-fold base cross-validation scores:".format(cv_k))
        print(mape)

        print("", end="\n")
        with yaspin(text="Exporting model...") as spinner:
            tokens   = self.name.split("/")
            name     = tokens[-1].split(".")[-2]
            name_tks = name.split("_")
            name     = "_".join(name_tks[:-1])
            path   = "."
            if len(tokens) > 1:
                path   = "/".join(tokens[0:-1])
            full_name = "{}/{}_e{}_d{}".format(path, name, last_good_n_estimators, last_good_max_depth)
            model.save_model("{}_model.json".format(full_name))
            spinner.ok()

        print("Model exported to {}_model.json".format(full_name))
