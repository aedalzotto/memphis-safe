from random import sample, seed
from pandas import read_csv, get_dummies
from yaspin import yaspin
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from .score import TargetRMSECallback, neg_estimators
from sklearn.linear_model import LinearRegression

class XGModel:
    def __init__(self, name):
        self.name       = name

        with yaspin(text="Loading train dataset...") as spinner:
            self.dataset = read_csv(name)

            scenarios = list(self.dataset["scenario"].unique())
            seed(7)
            train_scenarios = sample(scenarios, int(len(scenarios)*0.75))
            val_scenarios  = list(set(scenarios) - set(train_scenarios))

            train = self.dataset[self.dataset["scenario"].isin(train_scenarios)]
            # self.weights_train = train["weight"].values
            val   = self.dataset[self.dataset["scenario"].isin(val_scenarios)]

            self.y_full         = self.dataset[["latency"]]
            self.X_full         = self.dataset[["rel_time", "prod", "cons", "hops", "size"]]
            self.X_full["prod"] = self.X_full["prod"].astype("category")
            self.X_full["cons"] = self.X_full["cons"].astype("category")
            self.X_full         = get_dummies(self.X_full, columns=["prod", "cons"])

            self.y_train         = train[["latency"]]
            self.X_train         = train[["rel_time", "prod", "cons", "hops", "size"]]
            self.X_train["prod"] = self.X_train["prod"].astype("category")
            self.X_train["cons"] = self.X_train["cons"].astype("category")
            self.X_train         = get_dummies(self.X_train, columns=["prod", "cons"])

            self.y_val         = val[["latency"]]
            self.X_val         = val[["rel_time", "prod", "cons", "hops", "size"]]
            self.X_val["prod"] = self.X_val["prod"].astype("category")
            self.X_val["cons"] = self.X_val["cons"].astype("category")
            self.X_val         = get_dummies(self.X_val, columns=["prod", "cons"])

            spinner.ok()

        self.param_grid = {
            'eta': [0.3, 0.4, 0.5], # default 0.3
            'gamma': [0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3], # default 0
            'min_child_weight': [3, 4, 5], # default 1
            # 'max_depth': [3, 4, 5], # default 6
            # 'subsample': [0.7, 0.8, 0.9], # default 1.0
            # 'alpha': [0.3, 0.5, 0.7], # default 0
        }

    def linear(self):
        reg = LinearRegression().fit(self.X_full, self.y_full)
        print(self.X_full.columns)

        coefficients = reg.coef_
        print(f"Coefficients: {coefficients}")

        intercept = reg.intercept_
        print(f"Intercept: {intercept}")

    def avg(self):
        for prod in self.dataset["prod"].unique():
            for cons in self.dataset[self.dataset["prod"] == prod]["cons"].unique():
                avg = self.dataset[(self.dataset["prod"] == prod) & (self.dataset["cons"] == cons)]["latency"].mean()
                print("{}-{} = {}".format(prod, cons, avg))
        
        print("Default = {}".format(self.dataset["latency"].mean()))

    def train(self):
        eval = XGBRegressor(early_stopping_rounds=5, n_estimators=33)
        grid_search = GridSearchCV(
            estimator=eval,
            param_grid=self.param_grid,
            scoring='neg_root_mean_squared_error',
            cv=5,
            verbose=2,
            n_jobs=-1
        )
        grid_search.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], verbose=False)
        estimators = grid_search.best_estimator_.best_iteration+1

        print("\n", end="")
        print("Model selection/tuning lines: {}".format(self.X_train.shape[0]))
        print("Stopped at iteration {}".format(estimators))
        print("Best parameters found: ")
        print(grid_search.best_params_)

        lines = self.X_train.shape[0]
        print("Validation training lines: {}".format(lines))
        print("Validation mean RMSE: {}".format(-round(grid_search.best_score_, 3)))

        with yaspin(text="Fitting final model...") as spinner:
            model = XGBRegressor(n_estimators=estimators, **grid_search.best_params_)
            model.fit(self.X_full, self.y_full)
            spinner.ok()

        lines = self.X_full.shape[0]
        print("Full training lines: {}".format(lines))

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
                f.write(str(lines)+"\n")
                f.write(str(grid_search.best_score_)+"\n")
            spinner.ok()

        print("Model exported to {}_model.json".format(full_name))
