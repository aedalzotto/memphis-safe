from pandas import read_csv

class Metrics:
    def __init__(self, latency, inference, detection):
        self.lat = read_csv(latency)
        self.inf = read_csv(inference)
        self.det = read_csv(detection)
    
    def metrics(self):
        avg_lat = (self.lat["avg_latency"] * self.lat["n_inf"]).sum() / self.lat["n_inf"].sum() / 100.0
        print("Average inference latency = {} us".format(round(avg_lat, 1)))

        avg_det = self.inf["det_latency"].mean() / 100.0
        print("Average detection latency = {} us".format(round(avg_det, 1)))

        self.det["duration"] = self.det["end"] - self.det["beggining"]
        
        for scenario in self.det["scenario"]:
            base  = self.det.loc[self.det[(self.det["scenario"] == scenario) & (self.det["rtd"] == False)].index, "duration"].item()
            worst = self.det.loc[self.det[(self.det["scenario"] == scenario) & (self.det["rtd"] == True )].index, "duration"].item()
            rate = (worst - base) / base
            self.det.loc[self.det[(self.det["scenario"] == scenario) & (self.det["rtd"] == True)].index, "rate"] = rate

        print("Average exec. time increase = {}%".format(round(self.det["rate"].mean()*100, 2)))
