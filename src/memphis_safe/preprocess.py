from yaspin import yaspin
from pandas import read_csv
from joblib import Parallel, delayed
from tqdm import tqdm
from random import sample

class Preprocess:
    def __init__(self, dataset):
        with yaspin(text="Loading dataset...") as spinner:
            self.name = dataset
            self.df   = read_csv(dataset)
            spinner.ok()

    def preprocess(self, threshold, rate, notrain, notag):
        if notag:
            self.df["anomaly"] = False
        else:
            anomalies = self.__tag(threshold)

        if notrain:
            test = self.df[self.df["malicious"] == True]
            test = self.__clean_test(test)
            real_rate = 0.0
        else:
            train, test, real_rate = self.__split(rate, anomalies)
            train = self.__clean_train(train)
            test  = self.__clean_test(test)

        print("\n", end="")
        with yaspin(text="Exporting datasets...") as spinner:
            tokens = self.name.split("/")
            name   = tokens[-1].split(".")[-2]
            path   = "."
            if len(tokens) > 1:
                path   = "/".join(tokens[0:-1])
            full_name = "{}/{}_t{}_r{}".format(path, name, int(threshold*100), int(real_rate*100))
            test.to_csv("{}_test.csv" .format(full_name), index=False)
            if not notrain:
                train.to_csv("{}_train.csv".format(full_name), index=False)
            spinner.ok()
        
        print("Datasets exported to {}_{{train,test}}.csv".format(full_name))

    def __tag(self, threshold):
        print("\nComputing tags...")
        scenarios = self.df["scenario"].drop_duplicates().sort_values().to_list()
        anomalies = Parallel(n_jobs=-1, require="sharedmem")(delayed(self.__tag_scenario)(scenario, threshold) for scenario in tqdm(scenarios))

        print("Tagging dataset...")
        for anomaly in tqdm(anomalies):
            self.df.loc[self.df[(self.df["scenario"] == anomaly[0]) & (self.df["malicious"] == True)].index, "anomaly"] = anomaly[1]

        self.df.loc[self.df[self.df["malicious"] == False].index, "anomaly"] = False

        anom_scenarios = [sum(i[1]) for i in anomalies]
        anomaly_cnt = sum(anom_scenarios)
        print("Dataset has {} anomal{}".format(anomaly_cnt, 'y' if anomaly_cnt == 1 else 'ies'))

        return anom_scenarios

    def __split(self, rate, anomalies):
        print("\n", end="")
        with yaspin(text="Splitting dataset...") as spinner:
            anom_indices = sorted(range(len(anomalies)), key=anomalies.__getitem__, reverse=True)

            test_len = min(len(anomalies), int(sum(a > 0 for a in anomalies)*(1.0 - rate)))
            test_idx = anom_indices[0:test_len]

            train_len = int(test_len / (1 - rate) * rate)
            train_idx = sample(anom_indices[test_len:], train_len)

            real_rate = train_len / (train_len + test_len)

            train = self.df[(self.df["scenario"].isin(train_idx)) & (self.df["malicious"] == False)].reset_index(drop=True)
            test  = self.df[(self.df["scenario"].isin(test_idx))  & (self.df["malicious"] ==  True)].reset_index(drop=True)
            spinner.ok()

        print("Train scenarios: {}".format(len(train_idx)))
        print("Test scenarios: {}".format(len(test_idx)))
        print("Rate: {}".format(real_rate))

        print("\nTest scenario distribution:")
        print(test["anomaly"].value_counts())

        print("\nScenario with most anomalies: {}".format(test_idx[0]))
        print("Scenario distribution:")
        print(test[(test["scenario"] == test_idx[0])]["anomaly"].value_counts())

        return train, test, real_rate

    def __clean_train(self, train):
        print("\n", end="")
        with yaspin(text="Cleaning train dataset...") as spinner:
            train = train[["rel_timestamp", "prod", "cons", "hops", "size", "total_time"]]

            spinner.ok()

        return train
    
    def __clean_test(self, test):
        print("\n", end="")
        with yaspin(text="Cleaning test dataset...") as spinner:
            test = test[["scenario", "rel_timestamp", "prod", "cons", "hops", "size", "total_time", "anomaly"]]

            spinner.ok()

        return test

    def __tag_scenario(self, scenario, threshold):
        lat_n = self.df[(self.df["scenario"] == scenario) & (self.df["malicious"] == False)]["total_time"].values
        lat_m = self.df[(self.df["scenario"] == scenario) & (self.df["malicious"] ==  True)]["total_time"].values
        anomalies = (lat_m - lat_n) / lat_n > threshold
        return (scenario, anomalies)
    