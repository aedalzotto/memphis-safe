from xgboost.callback import TrainingCallback

def neg_estimators(estimator, X=None, y=None):
    return -len(estimator.get_booster().get_dump())

class TargetRMSECallback(TrainingCallback):
    def __init__(self, target_rmse):
        self.target_rmse = target_rmse

    def after_iteration(self, model, epoch, evals_log):
        return (evals_log['validation_0']['rmse'][-1] <= self.target_rmse)
