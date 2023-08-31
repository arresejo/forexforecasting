SEED = 42
# DATA CONFIG
START_DATE = '2013-01-01'
END_DATE = '2023-01-01'
RAW_DATA_DIR = '/Volumes/T7/forex/data/ASCII/M1'

PROJECT_ROOT_DIR = '/Users/arresejo/Documents/Dissertation/forex_forecasting'

DATASETS_DIR = f'{PROJECT_ROOT_DIR}/datasets'
PROCESSED_DIR = f'{PROJECT_ROOT_DIR}/processed'
REPORTS_DIR = f'{PROJECT_ROOT_DIR}/reports'

PREPROCESSING_DIR = f'{PROCESSED_DIR}/preprocessing'
FEATURES_DIR = f'{PROCESSED_DIR}/features'

PREPROCESSING_PRED_DIR = f'{PREPROCESSING_DIR}/prediction'
PREPROCESSING_CLUS_DIR = f'{PREPROCESSING_DIR}/clustering'
PREPROCESSING_PRED_CLUS_DIR = f'{PREPROCESSING_DIR}/prediction_with_clustering'

FEATURES_PRED_DIR = f'{FEATURES_DIR}/prediction'
FEATURES_CLUS_DIR = f'{FEATURES_DIR}/clustering'
FEATURES_PRED_CLUS_DIR = f'{FEATURES_DIR}/prediction_with_clustering'

TOTAL_STUDY_PERIODS_FILE = f"{PREPROCESSING_DIR}/total_study_periods.txt"
CURRENCY_PAIRS_FILE = f"{PREPROCESSING_DIR}/currency_pairs.txt"

CLUSTERS_DIR = f'{PROCESSED_DIR}/clusters'
HDBS_CLUSTERS_FILENAME = 'hdbs-clusters-{}.pickle'

FILENAME_FORMAT = 'DAT_ASCII_{}_DAILY.csv'

# Preprocessing
PRED_TRAIN_FILENAME_FORMAT = 'train_{}_period{}.csv'
PRED_TEST_FILENAME_FORMAT = 'test_{}_period{}.csv'
CLUS_TRAIN_FILENAME_FORMAT = 'train_period{}.csv'
CLUS_TEST_FILENAME_FORMAT = 'test_period{}.csv'
PRED_CLUS_TRAIN_FILENAME_FORMAT = 'train_period{}_MA{}.csv'
PRED_CLUS_TEST_FILENAME_FORMAT = 'test_period{}_MA{}.csv'

# Features
PRED_X_TRAIN_FT_FILENAME_FORMAT = 'pred_X_train_seq_{}_period{}.npy'
PRED_y_TRAIN_FT_FILENAME_FORMAT = 'pred_y_train_seq_{}_period{}.npy'
PRED_X_TEST_FT_FILENAME_FORMAT = 'pred_X_test_seq_{}_period{}.npy'
PRED_y_TEST_FT_FILENAME_FORMAT = 'pred_y_test_seq_{}_period{}.npy'

CLUS_TRAIN_FT_FILENAME_FORMAT = 'train_period{}_{}.npy'
CLUS_TEST_FT_FILENAME_FORMAT = 'test_period{}_{}.npy'

PRED_CLUS_X_TRAIN_FT_FILENAME_FORMAT = 'pred_X_train_seq_{}_period{}_MA{}_{}.npy'
PRED_CLUS_y_TRAIN_FT_FILENAME_FORMAT = 'pred_y_train_seq_{}_period{}_MA{}_{}.npy'
PRED_CLUS_X_TEST_FT_FILENAME_FORMAT = 'pred_X_test_seq_{}_period{}_MA{}_{}.npy'
PRED_CLUS_y_TEST_FT_FILENAME_FORMAT = 'pred_y_test_seq_{}_period{}_MA{}_{}.npy'

# Reports
NAIVE_REPORT_DIR = f"{REPORTS_DIR}/naive"
LR_REPORT_DIR = f"{REPORTS_DIR}/LR"
RF_REPORT_DIR = f"{REPORTS_DIR}/RF"
ANN_REPORT_DIR = f"{REPORTS_DIR}/ANN"
RNN_REPORT_DIR = f"{REPORTS_DIR}/RNN"
LSTM_REPORT_DIR = f"{REPORTS_DIR}/LSTM"
GRU_REPORT_DIR = f"{REPORTS_DIR}/GRU"

LR_CLUS_REPORT_DIR = f"{REPORTS_DIR}/LR_clustering"
RF_CLUS_REPORT_DIR = f"{REPORTS_DIR}/RF_clustering"
ANN_CLUS_REPORT_DIR = f"{REPORTS_DIR}/ANN_clustering"
RNN_CLUS_REPORT_DIR = f"{REPORTS_DIR}/RNN_clustering"
LSTM_CLUS_REPORT_DIR = f"{REPORTS_DIR}/LSTM_clustering"
BLSTM_CLUS_REPORT_DIR = f"{REPORTS_DIR}/BLSTM_clustering"
GRU_CLUS_REPORT_DIR = f"{REPORTS_DIR}/GRU_clustering"

HDBSCAN_REPORT_DIR = f"{REPORTS_DIR}/HDBSCAN"

HDBSCAN_REPORT_FILENAME = 'metrics.csv'

PRED_REPORT_FILENAME_FORMAT = '{}.csv'
PRED_CLUS_REPORT_FILENAME_FORMAT = '{}_MA{}.csv'

PRED_OVERALL_REPORT_FILENAME_FORMAT = 'summary.csv'
PRED_CLUS_OVERALL_REPORT_FILENAME_FORMAT = 'summary_MA{}.csv'


# Feature extraction & reduction
TSFRESH = 'tsfresh'
TSNE = 'tsne'
TSFRESH_TSNE = f"{TSFRESH}_{TSNE}"

PCA_CONFIG = {
    'n_components': 0.95
}

TSNE_CONFIG = {
    'learning_rate': 'auto',
    'perplexity': 3,
    'n_iter': 5000,
    'random_state': SEED,
    'init': 'random'
}

TARGET_CURRENCY_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']

N_LAGS = 240

TRAINING_DAYS = 750
TESTING_DAYS = 250
STUDY_PERIOD_DAYS = TRAINING_DAYS + TESTING_DAYS

# Models config
ANN_BATCH_SIZE = 32
ANN_EPOCHS = 200
ANN_PATIENCE = 10
RNN_BATCH_SIZE = 32
RNN_EPOCHS = 500
RNN_PATIENCE = 10

RF_N_ESTIMATORS = 1000
RF_MAX_DEPTH = 20
RF_RANDOM_STATE = SEED

HDBSCAN_CONFIG = {
    'min_cluster_size': 3,
    'min_samples': 2
}

