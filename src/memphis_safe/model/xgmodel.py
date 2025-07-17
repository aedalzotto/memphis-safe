from pandas import read_csv, get_dummies
from yaspin import yaspin
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

class XGModel:
    def __init__(self, name):
        self.name       = name

        with yaspin(text="Loading train dataset...") as spinner:
            X         = read_csv(name)
            y         = X[["latency"]]
            X         = X[["rel_time", "prod", "cons", "hops", "size"]]
            X["prod"] = X["prod"].astype("category")
            X["cons"] = X["cons"].astype("category")
            X         = get_dummies(X, columns=["prod", "cons"])

            self.X_train_full, self.X_test, self.y_train_full, self.y_test = train_test_split(
                X, 
                y, 
                test_size=0.2,
                random_state=7
            )

            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                self.X_train_full, 
                self.y_train_full, 
                test_size=0.2,
                random_state=7
            )

            spinner.ok()

    def train(self):
        print("\n", end="")
        with yaspin(text="Evaluating model...") as spinner:
            # model = XGBRegressor(base_score=50, n_estimators=n_estimators, max_depth=max_depth, min_child_weight=5, gamma=1, reg_lambda=1, subsample=0.8, colsample_bytree=0.8, eta=0.2)
            eval = XGBRegressor(base_score=50, objective='reg:squarederror', early_stopping_rounds=5)
            eval.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], verbose=False)
            spinner.ok()

        print("Stopped at iteration {}".format(eval.best_iteration))
        print("Model selection/tuning lines: {}".format(self.X_train.shape[0]))
        print("Model selection/tuning RMSE: {} -- do not report this data".format(round(eval.evals_result()["validation_0"]["rmse"][eval.best_iteration], 3)))

        print("\n", end="") 
        with yaspin(text="Training and testing final model...") as spinner:
            model = XGBRegressor(base_score=50, objective='reg:squarederror', n_estimators=eval.best_iteration)
            model.fit(self.X_train_full, self.y_train_full)
            y_pred = model.predict(self.X_test)
            rmse   = root_mean_squared_error(self.y_test, y_pred)
            spinner.ok()

        print("Model training lines: {}".format(self.X_train_full.shape[0]))
        print("Test RMSE: {}".format(round(rmse, 3)))

        print("", end="\n")
        with yaspin(text="Exporting model...") as spinner:
            tokens   = self.name.split("/")
            name     = tokens[-1].split(".")[-2]
            name_tks = name.split("_")
            name     = "_".join(name_tks[:-1])
            path   = "."
            if len(tokens) > 1:
                path   = "/".join(tokens[0:-1])
            full_name = "{}/{}_e{}".format(path, name, model.n_estimators)
            model.save_model("{}_model.json".format(full_name))
            spinner.ok()

        print("Model exported to {}_model.json".format(full_name))
