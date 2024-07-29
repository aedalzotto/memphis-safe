from xgboost import XGBRegressor
from pandas import read_csv
from yaspin import yaspin
from sklearn.metrics import recall_score, precision_score, f1_score

class Safe:
    def __init__(self, model, test):
        with yaspin(text="Loading model...") as spinner:
            self.model = XGBRegressor()
            self.model.load_model(model)
            spinner.ok()

        with yaspin(text="Loading test dataset...") as spinner:
            self.X = read_csv(test)
            self.y = self.X[["anomaly", "total_time"]]
            self.X.drop(columns=["anomaly", "total_time"], inplace=True)
            spinner.ok()
    
    def test(self, threshold):
        y_pred = self.model.predict(self.X)
        tag = self.y["anomaly"].astype('bool')
        tag_pred = abs(y_pred - self.y["total_time"]) / self.y["total_time"] > threshold

        print("\nTest recall:    {}".format(   recall_score(tag, tag_pred)))
        print(  "Test precision: {}".format(precision_score(tag, tag_pred)))
        print(  "Test F1:        {}".format(       f1_score(tag, tag_pred)))
