from yaspin import yaspin
from pandas import read_csv

class Joiner:
    def __init__(self, test, rtd, threshold):
        self.name = rtd
        self.threshold = threshold
        with yaspin(text="Loading datasets...") as spinner:
            self.test = read_csv(test)
            self.rtd  = read_csv(rtd)
            spinner.ok()

    def join(self):
        with yaspin(text="Joining datasets...") as spinner:
            # Group both datasets
            grouped_rtd = self.rtd.groupby(['scenario', 'prod', 'cons'])
            grouped_test = self.test.groupby(['scenario', 'prod', 'cons'])

            for key, sub_rtd in grouped_rtd:
                if key in grouped_test.groups:
                    sub_test = grouped_test.get_group(key)

                    # Sort both by snd_time for row-to-row alignment
                    sub_rtd = sub_rtd.sort_values('snd_time')
                    sub_test = sub_test.sort_values('snd_time')

                    # Compare latency using values to bypass index alignment
                    # This assumes both sub-datasets have the same row count
                    is_malicious = (sub_rtd['latency'].values - sub_test['latency'].values) > self.threshold
                    
                    # Write back to the original dataframe using the sorted indices
                    self.rtd.loc[sub_rtd.index, 'malicious'] = is_malicious

            spinner.ok()

        print(self.rtd["malicious"].value_counts())

        self.rtd.to_csv("{}_joined_{}.csv".format(self.name[:-4], self.threshold), index=False)
        print("Joined dataset saved to {}_joined_{}.csv".format(self.name[:-4], self.threshold))
