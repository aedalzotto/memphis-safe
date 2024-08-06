from pandas import read_csv
from random import sample
from yaspin import yaspin
from tqdm import tqdm
from joblib import Parallel, delayed

class Split:
    def __init__(self, dataset):
        with yaspin(text="Loading dataset...") as spinner:
            self.name = dataset
            self.df   = read_csv(dataset)
            self.scenarios = list(set(self.df['scenario'].tolist()))
            spinner.ok()

    def split(self, rate, export=False):
        print("Computing anomalies...")

        anomalies = Parallel(n_jobs=-1, require='sharedmem')(delayed(self.__get_anomalies)(scenario) for scenario in tqdm(self.scenarios))
        anom_indices = sorted(range(len(anomalies)), key=anomalies.__getitem__, reverse=True)

        test_len = min(len(anomalies), int(sum(a > 0 for a in anomalies)*(1.0 - rate)))
        test_idx = anom_indices[0:test_len]

        train_len = int(test_len / (1 - rate) * rate)
        train_idx = sample(anom_indices[test_len:], train_len)

        print("\nTrain scenarios: {}".format(len(train_idx)))
        print("Test scenarios: {}".format(len(test_idx)))

        self.train = self.df[(self.df["scenario"].isin(train_idx)) & (self.df["malicious"] == False)].reset_index(drop=True)
        self.test  = self.df[(self.df["scenario"].isin(test_idx))  & (self.df["malicious"] ==  True)].reset_index(drop=True)
        
        if export:
            tokens = self.name.split("/")
            name   = tokens[-1].split(".")[-2]
            path   = "."
            if len(tokens) > 1:
                path   = "/".join(tokens[0:-1])
            full_name = "{}/{}".format(path, name)
            self.train.to_csv("{}_train.csv".format(full_name), index=False)
            self.test .to_csv("{}_test.csv" .format(full_name), index=False)

        print("\nTest scenario distribution:")
        print(self.test["anomaly"].value_counts())

        print("\nScenario with most anomalies: {}".format(test_idx[0]))
        print("Scenario distribution:")
        print(self.test[(self.test["scenario"] == test_idx[0])]["anomaly"].value_counts())

        return self.train, self.test
    
    def __get_anomalies(self, scenario):
        return self.df[(self.df["scenario"] == scenario) & (self.df["malicious"] == True)]["anomaly"].sum()
