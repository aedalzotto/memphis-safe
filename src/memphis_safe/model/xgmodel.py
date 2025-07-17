from numpy import sqrt
from pandas import read_csv, get_dummies
from yaspin import yaspin
from math import ceil
from xgboost import XGBRegressor, to_graphviz,plot_tree
from sklearn.model_selection import cross_val_score
from .neg_mape import neg_mean_percentage_error
from numpy import ones

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

    def __get_score(self, cv_k, n_estimators, max_depth):
        # model = XGBRegressor(base_score=50, n_estimators=n_estimators, max_depth=max_depth, min_child_weight=5, gamma=1, reg_lambda=1, subsample=0.8, colsample_bytree=0.8, eta=0.2)
        model = XGBRegressor(base_score=50, n_estimators=n_estimators, max_depth=max_depth, objective='reg:squarederror')
        scores = cross_val_score(model, self.X, self.y, scoring='neg_root_mean_squared_error', cv=cv_k)
        score = -scores
        return score, model

    def train(self, cv_k, n_estimators=None, max_depth=None):
        last_good_n_estimators = 100 if n_estimators is None else n_estimators
        last_good_max_depth = 6 if max_depth is None else max_depth
        print("\n", end="")
        with yaspin(text="Training base model...") as spinner:
            last_good_score, last_good_model = self.__get_score(cv_k, last_good_n_estimators, last_good_max_depth)
            spinner.ok()

        print("Base cross-validation mean RMSE: {}".format(round(last_good_score.mean(), 3)))
        print("Base cross-validation RMSE std dev: {}".format(round(last_good_score.std(), 3)))
        print("{}-fold base cross-validation scores:".format(cv_k))
        print(last_good_score)

        if n_estimators is None and max_depth is None:
            with yaspin(text="Finding smallest model...") as spinner:
                max_score = last_good_score.mean() * 1.1

                last_bad_n_estimators = 0
                while True:
                    n_estimators = round((last_good_n_estimators+last_bad_n_estimators) / 2)
                    if n_estimators in [last_good_n_estimators, last_bad_n_estimators]:
                        break
                    print("Trying {} estimators".format(n_estimators))
                    score, model = self.__get_score(cv_k, n_estimators, last_good_max_depth)
                    last_score = score.mean()
                    if last_score < max_score:
                        last_good_model = model
                        last_good_score = score
                        last_good_n_estimators = n_estimators
                    else:
                        last_bad_n_estimators = n_estimators

                last_bad_max_depth = 3
                while True:
                    max_depth = round((last_good_max_depth+last_bad_max_depth) / 2)
                    if max_depth in [last_good_max_depth, last_bad_max_depth]:
                        break
                    print("Trying depth {}".format(max_depth))
                    score, model = self.__get_score(cv_k, last_good_n_estimators, max_depth)
                    last_score = score.mean()
                    if last_score < max_score:
                        last_good_model = model
                        last_good_score = score
                        last_good_max_depth = max_depth
                    else:
                        last_bad_max_depth = max_depth
                    
                spinner.ok()

        # last_good_model.fit(self.X, self.y, sample_weight=self.weights)
        last_good_model.fit(self.X, self.y)

        print("n_estimators={}; max_depth={}".format(last_good_n_estimators, last_good_max_depth))
        print("Final cross-validation RMSE mean:    {}".format(round(last_good_score.mean(), 3)))
        print("Final cross-validation RMSE std dev: {}".format(round(last_good_score.std(), 3)))
        print("Final cross-validation RMSE min:     {}".format(round(last_good_score.min(), 3)))
        print("Final cross-validation RMSE max:     {}".format(round(last_good_score.max(), 3)))
        print("{}-fold base cross-validation scores:".format(cv_k))
        print(last_good_score)

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
            last_good_model.save_model("{}_model.json".format(full_name))
            spinner.ok()

        print("Model exported to {}_model.json".format(full_name))
