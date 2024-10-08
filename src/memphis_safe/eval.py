from pandas import read_csv, DataFrame, concat
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from sklearn.utils.multiclass import type_of_target

class Eval:
    def __init__(self, test, rtd):
        self.test  = read_csv(test)
        self.rtd   = read_csv(rtd)
        self.test['anomaly_pred'] = False

    def eval(self):
        final = DataFrame(columns=['scenario', 'hops', 'size', 'prod', 'cons', 'total_time', 'anomaly', 'anomaly_pred'])
        for scenario in self.rtd["scenario"].unique():
            sc_test = self.test[self.test["scenario"] == scenario].sort_values(by='rel_timestamp').reset_index()
            sc_rtd  = self.rtd [self.rtd ["scenario"] == scenario].sort_values(by='rel_timestamp').reset_index()

            for rowidx, row in sc_rtd.iterrows():
                if sc_test.iloc[row["index"]]["prod"] == row["prod"] and sc_test.iloc[row["index"]]["cons"] == row["cons"]:
                    sc_test.iloc[row["index"], sc_test.columns.get_loc('anomaly_pred')] = True
                else:
                    raise ValueError("Detected index does not match")

            final = concat([sc_test, final], ignore_index=True)

        y      = final["anomaly"].astype('bool')
        y_pred = final["anomaly_pred"].astype('bool')

        print("\nTest recall:    {}".format(   recall_score(y, y_pred, zero_division=1.0)))
        print(  "Test precision: {}".format(precision_score(y, y_pred)))
        print(  "Test F1:        {}".format(       f1_score(y, y_pred)))

        print(confusion_matrix(y, y_pred))
