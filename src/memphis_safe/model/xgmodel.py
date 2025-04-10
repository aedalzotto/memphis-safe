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

            self.X.sort_values(by='rel_time', inplace = True)
            total = self.X.shape[0]
            self.weights = ones(total)
            n_warmup = self.X[self.X["rel_time"] < 1000].shape[0]
            weight = int(total/n_warmup/2)
            # weight = 1
            self.weights[:n_warmup] = weight
            self.y         = self.X[["latency"]]
            self.X         = self.X[["rel_time", "prod", "cons", "hops", "size"]]
            self.X["prod"] = self.X["prod"].astype("category")
            self.X["cons"] = self.X["cons"].astype("category")
            self.X         = get_dummies(self.X, columns=["prod", "cons"])
            spinner.ok()

        print("\n{} total".format(total))
        print("{} with more weight".format(n_warmup))
        print("{} weight".format(weight))

    def __get_mape(self, cv_k, n_estimators, max_depth):
        # model = XGBRegressor(base_score=50, n_estimators=n_estimators, max_depth=max_depth, min_child_weight=5, gamma=1, reg_lambda=1, subsample=0.8, colsample_bytree=0.8, eta=0.2)
        model = XGBRegressor(base_score=50, n_estimators=n_estimators, max_depth=max_depth)
        scores = cross_val_score(model, self.X, self.y, scoring=neg_mean_percentage_error, cv=cv_k, fit_params={'sample_weight': self.weights})
        mape = -scores
        return mape, model

    def train(self, cv_k):
        last_good_n_estimators = 100
        last_good_max_depth = 6
        print("\n", end="")
        with yaspin(text="Training base model...") as spinner:
            last_good_score, last_good_model = self.__get_mape(cv_k, last_good_n_estimators, last_good_max_depth)
            spinner.ok()

        print("Base cross-validation mean MAPE: {}".format(round(last_good_score.mean(), 3)))
        print("Base cross-validation MAPE std dev: {}".format(round(last_good_score.std(), 3)))
        print("{}-fold base cross-validation scores:".format(cv_k))
        print(last_good_score)

        with yaspin(text="Finding smallest model...") as spinner:
            max_score = last_good_score.mean() * 1.1

            last_bad_n_estimators = 0
            while True:
                n_estimators = round((last_good_n_estimators+last_bad_n_estimators) / 2)
                if n_estimators in [last_good_n_estimators, last_bad_n_estimators]:
                    break
                print("Trying {} estimators".format(n_estimators))
                mape, model = self.__get_mape(cv_k, n_estimators, last_good_max_depth)
                last_score = mape.mean()
                if last_score < max_score:
                    last_good_model = model
                    last_good_score = mape
                    last_good_n_estimators = n_estimators
                else:
                    last_bad_n_estimators = n_estimators

            last_bad_max_depth = 3
            while True:
                max_depth = round((last_good_max_depth+last_bad_max_depth) / 2)
                if max_depth in [last_good_max_depth, last_bad_max_depth]:
                    break
                print("Trying depth {}".format(max_depth))
                mape, model = self.__get_mape(cv_k, last_good_n_estimators, max_depth)
                last_score = mape.mean()
                if last_score < max_score:
                    last_good_model = model
                    last_good_score = mape
                    last_good_max_depth = max_depth
                else:
                    last_bad_max_depth = max_depth
                
            last_good_model.fit(self.X, self.y, sample_weight=self.weights)
            spinner.ok()

        print("n_estimators={}; max_depth={}".format(last_good_n_estimators, last_good_max_depth))
        print("Final cross-validation MAPE mean:    {}".format(round(last_good_score.mean(), 3)))
        print("Final cross-validation MAPE std dev: {}".format(round(last_good_score.std(), 3)))
        print("Final cross-validation MAPE min:     {}".format(round(last_good_score.min(), 3)))
        print("Final cross-validation MAPE max:     {}".format(round(last_good_score.max(), 3)))
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
