from pandas import read_csv
from yaspin import yaspin
from tqdm import tqdm
from joblib import Parallel, delayed

class Tag:
    def __init__(self, dataset):
        with yaspin(text="Loading dataset...") as spinner:
            self.name = dataset
            self.df   = read_csv(dataset)
            self.scenarios = list(set(self.df['scenario'].tolist()))
            spinner.ok()

    def __tag_scenario(df, scenario, threshold):
        lat_n = df[(df["scenario"] == scenario) & (df["malicious"] == False)]["total_time"].values
        lat_m = df[(df["scenario"] == scenario) & (df["malicious"] ==  True)]["total_time"].values
        anomalies = (lat_m - lat_n) / lat_n > threshold
        return (scenario, anomalies)

    def tag(self, threshold, export=False):
        print("Computing tags...")
        anomalies = Parallel(n_jobs=-1)(delayed(Tag.__tag_scenario)(self.df, scenario, threshold) for scenario in tqdm(self.scenarios))

        print("Tagging dataset...")
        for anomaly in tqdm(anomalies):
            self.df.loc[self.df[(self.df["scenario"] == anomaly[0]) & (self.df["malicious"] == True)].index, "anomaly"] = anomaly[1]

        anomaly_cnt = self.df["anomaly"].sum()
        print("Dataset has {} anomal{}".format(anomaly_cnt, 'y' if anomaly_cnt == 1 else 'ies'))

        if export:
            tokens = self.name.split("/")
            name   = tokens[-1].split(".")[-2]
            path   = "."
            if (len(tokens) > 1):
                path   = "/".join(tokens[0:-1])
            full_name = "{}/{}_tagged.csv".format(path, name)
            self.df.to_csv(full_name, index=False)
            print("Dataset exported to {}".format(full_name))

        return self.df
