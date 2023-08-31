from src.features import build_features_prediction
from src.models.prediction import LSTM_model
from src.preprocessing import preprocess_prediction

if __name__ == '__main__':
    preprocess_prediction.preprocess()
    build_features_prediction.build_features()

    metrics_df = LSTM_model.run_model(plotting=False)