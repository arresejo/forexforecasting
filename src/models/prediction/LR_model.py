from sklearn.linear_model import LogisticRegression
from src.config import FEATURES_PRED_DIR, SEED, LR_REPORT_DIR
from src.models.prediction import common


def create_model():
    model = LogisticRegression(random_state=SEED)
    return model


def train_model(model, X_train_seq, y_train_seq, X_test_seq, y_test_seq, training_params):
    model.fit(X_train_seq, y_train_seq)


def predict(model, X_test_seq):
    y_pred_labels = model.predict(X_test_seq)
    y_pred_proba = model.predict_proba(X_test_seq)[:, 1]
    return y_pred_labels, y_pred_proba


def training_params():
    return None


def run_model(plotting=True):
    metrics_df = common.run_model(create_model,
                                  training_params,
                                  train_model,
                                  predict,
                                  FEATURES_PRED_DIR,
                                  LR_REPORT_DIR,
                                  plotting=plotting)
    return metrics_df


if __name__ == '__main__':
    run_model()
