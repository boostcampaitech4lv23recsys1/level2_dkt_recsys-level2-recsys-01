from xgboost import XGBClassifier


class XGBoost(object):
    def __init__(
            self,
            booster="gbtree",
            colsample_bylevel=0.8,
            colsample_bytree=0.8,
            gamma=0,
            max_depth=8,
            min_child_weight=4,
            n_estimators=70,
            nthread=4,
            objective="binary:logistic",
            random_state=417,
    ):
        self.model = XGBClassifier(
            booster=booster,
            colsample_bylevel=colsample_bylevel,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            n_estimators=n_estimators,
            nthread=nthread,
            objective=objective,
            random_state=random_state,
        )