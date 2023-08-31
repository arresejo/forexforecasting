from sklearn.ensemble import RandomForestClassifier

from src.config import FEATURES_PRED_DIR, RF_REPORT_DIR, RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_RANDOM_STATE
from src.models.prediction import common
from src.models.prediction.LR_model import train_model, predict


def create_model():
    model = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, random_state=RF_RANDOM_STATE)
    return model


def training_params():
    return None


def run_model(plotting=True):
    metrics_df = common.run_model(create_model,
                                  training_params,
                                  train_model,
                                  predict,
                                  FEATURES_PRED_DIR,
                                  RF_REPORT_DIR,
                                  plotting=plotting)
    return metrics_df


if __name__ == '__main__':
    run_model()
