from xgboost import XGBRegressor
from pandas import read_csv, DataFrame, concat
from yaspin import yaspin
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

class Safe:
    def __init__(self, model, test):
        self.model_name = model
        self.test_name  = test
        with yaspin(text="Loading model...") as spinner:
            self.model = XGBRegressor()
            self.model.load_model(model)
            spinner.ok()

        print("\n", end="")
        with yaspin(text="Loading test dataset...") as spinner:
            self.X = read_csv(test)
            self.X.drop("scenario", axis=1, inplace=True)
            spinner.ok()
    
    def test(self, threshold):
        print("\n", end="")
        y_pred = DataFrame()
        with yaspin(text="Testing model...") as spinner:
            self.y = self.X[["total_time", "anomaly"]]
            self.X.drop(columns=["total_time", "anomaly"], inplace=True)

            y_pred["time_pred"] = self.model.predict(self.X)
            y_pred["anomaly_pred"] = (self.y["total_time"] - y_pred["time_pred"]) / y_pred["time_pred"] > threshold
            spinner.ok()

        print("\nTest recall:    {}".format(   recall_score(self.y["anomaly"], y_pred["anomaly_pred"], zero_division=1.0)))
        print(  "Test precision: {}".format(precision_score(self.y["anomaly"], y_pred["anomaly_pred"])))
        print(  "Test F1:        {}".format(       f1_score(self.y["anomaly"], y_pred["anomaly_pred"])))

        print(confusion_matrix(self.y["anomaly"], y_pred["anomaly_pred"]))

        with yaspin(text="Exporting report...") as spinner:
            test_path = "_".join(self.test_name.split(".")[-2].split("_")[:-1])
            model_params = "_".join(self.model_name.split("_")[-3:-1])
            name = "{}_{}_tested.csv".format(test_path, model_params)
            df = concat([self.X, self.y, y_pred], axis=1)
            df.to_csv(name, index=False)
            spinner.ok()

        print("Report exported to {}".format(name))
