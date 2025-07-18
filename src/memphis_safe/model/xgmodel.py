from pandas import read_csv, get_dummies
from yaspin import yaspin
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

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

        self.param_grid = {
            'eta': [0.3, 0.4, 0.5], # default 0.3
            'gamma': [0.5, 0.7, 0.9], # default 0
            # 'max_depth': [3, 4, 5], # default 6           
            'min_child_weight': [3, 4, 5], # default 1
            # 'subsample': [0.7, 0.8, 0.9], # default 1.0
            'alpha': [0.3, 0.5, 0.7], # default 0
        }

    def train(self):
        eval = XGBRegressor(early_stopping_rounds=5)
        grid_search = GridSearchCV(
            estimator=eval,
            param_grid=self.param_grid,
            scoring='neg_root_mean_squared_error',
            cv=3,
            verbose=2, # Shows progress
            n_jobs=-1
        )
        grid_search.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], verbose=False)

        print("\n", end="")
        print("Model selection/tuning lines: {}".format(self.X_train.shape[0]))
        print("Model selection/tuning RMSE: {} -- do not report this data".format(round(-grid_search.best_score_, 3)))
        print("Stopped at iteration {}".format(grid_search.best_estimator_.best_iteration))
        print("Best parameters found: ")
        print(grid_search.best_params_)

        print("\n", end="") 
        with yaspin(text="Training and testing final model...") as spinner:
            model = XGBRegressor(n_estimators=grid_search.best_estimator_.best_iteration, **grid_search.best_params_)
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
            with open("{}_info.txt".format(full_name), "w") as f:
                f.write(str(grid_search.best_params_)+"\n")
            spinner.ok()

        print("Model exported to {}_model.json".format(full_name))
