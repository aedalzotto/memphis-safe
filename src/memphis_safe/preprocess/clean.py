from pandas import read_csv, get_dummies, Categorical
from yaspin import yaspin

class Clean:
	def __init__(self, train, test):
		with yaspin(text="Loading train dataset...") as spinner:
			self.train_name = train
			self.train   = read_csv(train)
			spinner.ok()

		with yaspin(text="Loading test dataset...") as spinner:
			self.test_name = test
			self.test   = read_csv(test)
			spinner.ok()
	
	def clean(self, export=False):
		self.train = self.train[["rel_timestamp", "prod", "cons", "hops", "size", "total_time"]]
		self.train["prod"] = Categorical(self.train["prod"])
		self.train["cons"] = Categorical(self.train["cons"])
		self.train = get_dummies(self.train, columns=["prod", "cons"])

		self.test = self.test[["rel_timestamp", "prod", "cons", "hops", "size", "total_time", "anomaly"]]
		self.test["prod"] = Categorical(self.test["prod"])
		self.test["cons"] = Categorical(self.test["cons"])
		self.test = get_dummies(self.test, columns=["prod", "cons"])

		self.train_X = self.train.drop(columns=["total_time"])
		self.train_y = self.train["total_time"]

		self.test_X = self.test.drop(columns=["total_time", "anomaly"])
		self.test_y = self.test[["anomaly", "total_time"]]

		if export:
			tokens = self.train_name.split("/")
			name   = tokens[-1].split(".")[-2]
			path   = "."
			if len(tokens) > 1:
				path   = "/".join(tokens[0:-1])
			full_name = "{}/{}".format(path, name)
			self.train.to_csv("{}_clean.csv".format(full_name), index=False)
		
		if export:
			tokens = self.test_name.split("/")
			name   = tokens[-1].split(".")[-2]
			path   = "."
			if len(tokens) > 1:
				path   = "/".join(tokens[0:-1])
			full_name = "{}/{}".format(path, name)
			self.test .to_csv("{}_clean.csv" .format(full_name), index=False)

		return self.train, self.test
