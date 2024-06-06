from numpy import sqrt
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

class XGModel:
    def __init__(self, data):
        self.X = data[0]
        self.y = data[1]
        self.model = XGBRegressor(objective="reg:squarederror")

    def train(self, k, export=False):
        print("\nTraining...")

        scores = cross_val_score(self.model, self.X, self.y, scoring='neg_mean_squared_error', cv=k)
        rmse = sqrt(-scores)
        self.model.fit(self.X, self.y)

        if export:
            self.model.save_model("model.json")

        print("Cross-validation mean RMSE: {}".format(round(rmse.mean(), 3)))
        print("{}-fold cross-validation scores:".format(k))
        print(rmse)

