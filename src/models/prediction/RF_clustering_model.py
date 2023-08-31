from src.config import FEATURES_PRED_CLUS_DIR, RF_CLUS_REPORT_DIR
from src.models.prediction import common
from src.models.prediction.LR_model import training_params
from src.models.prediction.RF_model import create_model, train_model, predict


def run_model(feature_method, ma_windows_size, plotting=True):
    metrics_df = common.run_model(create_model,
                                  training_params,
                                  train_model,
                                  predict,
                                  FEATURES_PRED_CLUS_DIR,
                                  RF_CLUS_REPORT_DIR,
                                  ma_windows_size=ma_windows_size,
                                  feature_method=feature_method,
                                  with_clustering=True,
                                  plotting=plotting)
    return metrics_df


if __name__ == '__main__':
    run_model()
