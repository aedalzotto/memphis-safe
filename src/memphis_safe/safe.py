from pandas import read_csv
from random import sample

class Safe:
    def __init__(self, dataset):
        self.name = dataset
        self.df   = read_csv(dataset)
        # self.df["anomaly"] = False
        self.scenarios = list(set(self.df['scenario'].tolist()))

    def tag(self, threshold, export=False):
        # No point in parallelizing
        for scenario in self.scenarios:
            lat_n = self.df[(self.df["scenario"] == scenario) & (self.df["malicious"] ==  True)]["total_time"].values
            lat_m = self.df[(self.df["scenario"] == scenario) & (self.df["malicious"] == False)]["total_time"].values
            anomalies = (lat_m - lat_n) / lat_n > threshold
            self.df.loc[self.df[(self.df["scenario"] == scenario) & (self.df["malicious"] == True)].index, "anomaly"] = anomalies

        if export:
            tokens = self.name.split("/")
            name   = tokens[-1].split(".")[-2]
            path   = "/".join(tokens[0:-1])
            self.df.to_csv("{}/{}_tagged.csv".format(path, name), index=False)

        print("Dataset has {} anomalies".format(self.df["anomaly"].sum()))

    def split(self, rate, export=False):
        anomalies = {}
        for scenario in self.scenarios:
            anomalies[scenario] = self.df[(self.df["scenario"] == scenario) & (self.df["malicious"] == True)]["anomaly"].sum()
        srt_anom = {k: v for k, v in sorted(anomalies.items(), key=lambda item: item[1], reverse=True)}

        test_len = min(len(srt_anom), int(len([srt_anom[df] for df in srt_anom if srt_anom[df] > 0])*(1.0 - rate)))
        test_idx = list(srt_anom)[0:test_len]

        train_len = int(test_len / (1 - rate) * rate)
        train_idx = sample(list(srt_anom)[test_len:], train_len)

        self.train = self.df[(self.df["scenario"].isin(train_idx)) & (self.df["malicious"] == False)].reset_index(drop=True)
        self.test  = self.df[(self.df["scenario"].isin(test_idx))  & (self.df["malicious"] ==  True)].reset_index(drop=True)
        
        if export:
            tokens = self.name.split("/")
            name   = tokens[-1].split(".")[-2]
            path   = "/".join(tokens[0:-1])
            self.train.to_csv("{}/{}_train.csv".format(path, name), index=False)
            self.test .to_csv("{}/{}_test.csv" .format(path, name), index=False)

        print("\nTrain scenarios: {}".format(len(train_idx)))
        print("Test scenarios: {}".format(len(test_idx)))

        print("\nTest scenario distribution:")
        print(self.test["anomaly"].value_counts())

        print("\nScenario with most anomalies: {}".format(test_idx[0]))
        print("Scenario distribution:")
        print(self.test[(self.test["scenario"] == test_idx[0])]["anomaly"].value_counts())
