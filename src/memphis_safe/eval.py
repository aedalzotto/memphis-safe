from pandas import read_csv, DataFrame, concat
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

class Eval:
    def __init__(self, test, rtd):
        test_df = read_csv(test)
        rtd_df  = read_csv(rtd)

        self.scenarios = sorted(rtd_df["scenario"].unique())
        
        self.test = {}
        self.rtd  = {}
        for scenario in self.scenarios:
            sc_test = test_df[test_df["scenario"] == scenario].sort_values(by='rel_timestamp')
            sc_rtd  = rtd_df [rtd_df ["scenario"] == scenario].sort_values(by='rel_timestamp')

            edge_test = {}
            edge_rtd  = {}
            for prod in sc_rtd["prod"].unique():
                prod_test = sc_test[sc_test["prod"] == prod]
                prod_rtd  = sc_rtd [sc_rtd ["prod"] == prod]
                for cons in prod_rtd["cons"].unique():
                    edge_test[(prod,cons)] = prod_test[prod_test["cons"] == cons].reset_index()
                    edge_rtd [(prod,cons)] = prod_rtd [prod_rtd ["cons"] == cons].reset_index()

            self.test[scenario] = edge_test
            self.rtd [scenario] = edge_rtd

    def eval(self):
        final = DataFrame(columns=['scenario', 'hops', 'size', 'prod', 'cons', 'total_time', 'anomaly', 'anomaly_pred'])
        for scenario in self.rtd:
            sc_rtd  = self.rtd [scenario]
            sc_test = self.test[scenario]

            for edge in sc_rtd:
                edge_rtd  = sc_rtd [edge]
                edge_test = sc_test[edge]
                for rowidx, row_rtd, in edge_rtd.iterrows():
                    final = concat(
                        [
                            DataFrame(
                                [
                                    [
                                        scenario,
                                        edge_test.iloc[rowidx]["hops"],
                                        edge_test.iloc[rowidx]["size"],
                                        edge[0], 
                                        edge[1],
                                        edge_test.iloc[rowidx]["total_time"],
                                        edge_test.iloc[rowidx]["anomaly"],
                                        edge_rtd.iloc [rowidx]["anomaly"],
                                    ]
                                ], 
                                columns=final.columns
                            ), 
                            final
                        ], 
                        ignore_index=True
                    )

        y      = final["anomaly"].astype('bool')
        y_pred = final["anomaly_pred"].astype('bool')

        print("\nTest recall:    {}".format(   recall_score(y, y_pred, zero_division=1.0)))
        print(  "Test precision: {}".format(precision_score(y, y_pred)))
        print(  "Test F1:        {}".format(       f1_score(y, y_pred)))

        print(confusion_matrix(y, y_pred))
