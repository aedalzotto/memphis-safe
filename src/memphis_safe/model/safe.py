from xgboost import XGBRegressor
from pandas import read_csv, DataFrame, concat, get_dummies
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
            self.df              = read_csv(test)
            self.X               = self.df[["rel_time", "prod", "cons", "hops", "size"]]
            self.X.loc[:,"prod"] = self.X["prod"].astype("category")
            self.X.loc[:,"cons"] = self.X["cons"].astype("category")
            self.X               = get_dummies(self.X, columns=["prod", "cons"])
            self.X               = self.X[self.model.feature_names_in_]
            spinner.ok()

        print("\nFeature importances:")
        print(self.model.feature_importances_)
    
    def test(self, threshold):
        print("\n", end="")
        y_pred = DataFrame()
        with yaspin(text="Testing model...") as spinner:
            y_pred["lat_pred"] = self.model.predict(self.X)
            y_pred["lat_diff"] = (self.df["latency"] - y_pred["lat_pred"]) / y_pred["lat_pred"]
            y_pred["mal_pred"] = y_pred["lat_diff"] > threshold
            spinner.ok()

        print("\nTest recall:    {}".format(   recall_score(self.df["malicious"], y_pred["mal_pred"])))
        print(  "Test precision: {}".format(precision_score(self.df["malicious"], y_pred["mal_pred"])))
        print(  "Test F1:        {}".format(       f1_score(self.df["malicious"], y_pred["mal_pred"])))

        print(confusion_matrix(self.df["malicious"], y_pred["mal_pred"], labels=[True, False]))

        self.df = concat([self.df, y_pred], axis=1)
        print("\nMin. diff TP:   {}".format(self.df[(self.df["mal_pred"] == True) & (self.df["malicious"] == True)]["lat_diff"].min()))
        print(  "Max. diff TP:   {}".format(self.df[(self.df["mal_pred"] == True) & (self.df["malicious"] == True)]["lat_diff"].max()))
        print(  "Avg. diff TP:   {}".format(self.df[(self.df["mal_pred"] == True) & (self.df["malicious"] == True)]["lat_diff"].mean()))

        print("\nMin. diff FP:   {}".format(self.df[(self.df["mal_pred"] == True) & (self.df["malicious"] == False)]["lat_diff"].min()))
        print(  "Max. diff FP:   {}".format(self.df[(self.df["mal_pred"] == True) & (self.df["malicious"] == False)]["lat_diff"].max()))
        print(  "Avg. diff FP:   {}".format(self.df[(self.df["mal_pred"] == True) & (self.df["malicious"] == False)]["lat_diff"].mean()))

        print("\nMin. HT FN:     {}".format(self.df[(self.df["mal_pred"] == False) & (self.df["malicious"] == True)]["mal_cycles"].min()))
        print(  "Max. HT FN:     {}".format(self.df[(self.df["mal_pred"] == False) & (self.df["malicious"] == True)]["mal_cycles"].max()))
        print(  "Avg. HT FN:     {}".format(self.df[(self.df["mal_pred"] == False) & (self.df["malicious"] == True)]["mal_cycles"].mean()))


        with yaspin(text="Exporting report...") as spinner:
            test_path = "_".join(self.test_name.split(".")[-2].split("_")[:-1])
            model_params = "_".join(self.model_name.split("_")[-3:-1])
            name = "{}_{}_tested.csv".format(test_path, model_params)
            self.df.to_csv(name, index=False)
            spinner.ok()

        print("Report exported to {}".format(name))
