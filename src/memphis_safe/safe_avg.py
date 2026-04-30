from pandas import read_csv, Series
from sklearn.metrics import recall_score, precision_score, f1_score

class SafeAvg:
    def __init__(self, train_dataset, test_dataset, threshold=50):
        train = read_csv(train_dataset)
        self.test = read_csv(test_dataset)
        groups = train.groupby(['prod', 'cons'])
        self.means_dict = groups['latency'].mean().to_dict()
        group_keys = zip(self.test['prod'], self.test['cons'])
        test_means = Series(group_keys).map(self.means_dict)
        
        # Calculate if the difference exceeds the threshold
        self.test['mal_pred'] = (self.test['latency'] - test_means.values) > threshold

        print("\nTest recall:    {}".format(   recall_score(self.test["malicious"], self.test["mal_pred"])))
        print(  "Test precision: {}".format(precision_score(self.test["malicious"], self.test["mal_pred"])))
        print(  "Test F1:        {}".format(       f1_score(self.test["malicious"], self.test["mal_pred"])))
