from pandas import read_csv, DataFrame, concat
from yaspin import yaspin
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from os import listdir
from joblib import Parallel, delayed
from tqdm import tqdm
from yaml import safe_load

class Eval:
    def __init__(self, testcase, dataset):
        with yaspin(text="Loading RTD dataset...") as spinner:
            self.df = read_csv(dataset)
            spinner.ok()

        self.apps = None
        self.base_scenario = ["{}/{}".format(testcase, scenario) for scenario in listdir(testcase) if scenario.startswith("sc_") and scenario.endswith("_m")]
        self.rtd_scenario  = ["{}/{}".format(testcase, scenario) for scenario in listdir(testcase) if scenario.startswith("sc_") and scenario.endswith("_rtd")]

    def __get_duration(scenario, apps):
            scen_name = scenario.split("/")[-1]
            scen_file = "{}/{}.yaml".format(scenario, scen_name)    
            with open(scen_file, "r") as f:
                yaml = safe_load(f)
                for task in yaml["management"]:
                    if task["task"] == "mapper_task":
                        mapper = task["static_mapping"]
                        break            

            with open("{}/log/log{}x{}.txt".format(scenario, mapper[0], mapper[1]), "r") as f:
                beggining = {}
                end = {}
                for line in f:
                    if line[0] == "$":
                        line = line.split("_")[-1]
                        tokens = line.split(" ")
                        if tokens[0] == "RELEASE" and int(tokens[6]) == 1:
                            beggining[int(tokens[6])] = int(tokens[3])
                        elif tokens[0] == "App" and int(tokens[1]) == 1:
                            end[int(tokens[1])] = int(tokens[5])

            lines = []
            for app in apps:
                line = {}
                line["scenario"] = scen_name.split("_")[1]
                line["app"] = app
                line["duration"] = int(end[app] - beggining[app])
                lines.append(line)

            return DataFrame(lines)


    def eval(self):
        apps = self.df["app"].unique()
        print("Extracting mapper logs from baseline scenario...")
        base_duration = Parallel(n_jobs=-1)(delayed(Eval.__get_duration)(scenario, apps) for scenario in tqdm(self.base_scenario))
        base_df = concat(base_duration, ignore_index=True)

        print("Extracting mapper logs from RTD scenario...")
        rtd_duration  = Parallel(n_jobs=-1)(delayed(Eval.__get_duration)(scenario, apps) for scenario in tqdm(self.rtd_scenario))
        rtd_df = concat(rtd_duration, ignore_index=True)

        true_pos = self.df[(self.df["malicious"] == True) & (self.df["mal_pred"] == True)]

        print("\nTest recall:    {}"  .format(round(   recall_score(self.df["malicious"], self.df["mal_pred"]),           3)))
        print(  "Test precision: {}"  .format(round(precision_score(self.df["malicious"], self.df["mal_pred"]),           3)))
        print(  "Test F1:        {}"  .format(round(       f1_score(self.df["malicious"], self.df["mal_pred"]),           3)))
        print(  "Avg. inf. lat.: {}"  .format(round(true_pos["inf_lat"].mean()/100.0,                                     3)))
        print(  "Avg. det. lat.: {}"  .format(round(true_pos["det_lat"].mean()/100.0,                                     3)))
        print(  "App time inc.:  {} %".format(round(((rtd_df["duration"].mean() / base_df["duration"].mean())-1.0)*100.0, 2)))

        print(confusion_matrix(self.df["malicious"], self.df["mal_pred"]))
