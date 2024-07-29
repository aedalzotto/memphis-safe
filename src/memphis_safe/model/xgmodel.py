from numpy import sqrt
from pandas import read_csv
from yaspin import yaspin
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

class XGModel:
    def __init__(self, data):
        with yaspin(text="Loading train dataset...") as spinner:
            self.data = data
            self.X    = read_csv(data)
            self.y    = self.X[["total_time"]]
            self.X.drop(columns=["total_time"], inplace=True)
            spinner.ok()

        self.model = XGBRegressor(objective="reg:squarederror")

    def train(self, cv_k, export=False):
        print("\nTraining...")

        scores = cross_val_score(self.model, self.X, self.y, scoring='neg_mean_squared_error', cv=cv_k)
        rmse = sqrt(-scores)
        self.model.fit(self.X, self.y)

        if export:
            tokens = self.data.split("/")
            name   = tokens[-1].split(".")[-2]
            path   = "."
            if len(tokens) > 1:
                path   = "/".join(tokens[0:-1])
            full_name = "{}/{}".format(path, name)
            self.model.save_model("{}_model.json".format(full_name))

        print("Cross-validation mean RMSE: {}".format(round(rmse.mean(), 3)))
        print("Cross-validation RMSE std dev: {}".format(round(rmse.std(), 3)))
        print("{}-fold cross-validation scores:".format(cv_k))
        print(rmse)
        return self.model
