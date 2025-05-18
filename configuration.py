
import os

class Config:
    """
    Configuration class for the project.
    Holds file paths, model parameters, and other constants.
    """
    # --- File Paths ---
    DATA_DIR = "./datasets"  # Assuming data files are in the same directory as the script
    RAW_TRAINING_FILE = os.path.join(DATA_DIR, 'training_set_VU_DM.csv')
    RAW_TEST_FILE = os.path.join(DATA_DIR, 'test_set_VU_DM.csv')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed_data")
    PROCESSED_TRAINING_FILE = os.path.join(PROCESSED_DATA_DIR, 'train_processed.csv')
    PROCESSED_TEST_FILE = os.path.join(PROCESSED_DATA_DIR, 'test_processed.csv')
    PROP_STATS_FILE = os.path.join(PROCESSED_DATA_DIR, 'prop_stats.csv')
    MODELS_DIR = os.path.join(DATA_DIR, "models")
    XGB_MODEL_PATH = os.path.join(MODELS_DIR, 'xgb_ranker_final.json')
    LGBM_MODEL_PATH = os.path.join(MODELS_DIR, 'lgbm_ranker_final.txt')
    META_MODEL_PATH = os.path.join(MODELS_DIR, 'meta_model_ridge.pkl')
    META_SCALER_PATH = os.path.join(MODELS_DIR, 'meta_scaler.pkl')
    SUBMISSIONS_DIR = os.path.join(DATA_DIR, "submissions")

    # --- Feature Engineering ---
    PRICE_CAP = 2060.0355  # Example: Cap for price_usd based on original script

    # --- Data Splitting ---
    # For creating a development test set from the main training data
    DEV_TEST_SPLIT_SIZE = 0.2
    RANDOM_STATE_MAIN_SPLIT = 42

    # For creating a validation set from the development training data (for early stopping/tuning)
    DEV_VALIDATION_SPLIT_SIZE = 0.25 # e.g., 0.25 of (1 - DEV_TEST_SPLIT_SIZE) data
    RANDOM_STATE_VALIDATION_SPLIT = 123

    # For Meta-Model K-Fold
    KFOLD_N_SPLITS = 5
    RANDOM_STATE_KFOLD = 7

    # --- Model Parameters ---
    # XGBoost Parameters (example, tune these)
    XGB_PARAMS = {
        "objective": "rank:ndcg",
        "tree_method": "hist",
        # "device": "cuda", # Use "cuda" if GPU is available and XGBoost is compiled with GPU support
        "eval_metric": "ndcg@5",
        "eta": 0.05, # Learning rate
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0, # L2 regularization
        "alpha": 0.5,  # L1 regularization
        "seed": 42
    }
    XGB_NUM_BOOST_ROUND = 1000 # Max rounds, use with early stopping
    XGB_EARLY_STOPPING_ROUNDS = 50

    # LightGBM Parameters (example, tune these)
    LGBM_PARAMS = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [5],
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 6,
        'min_data_in_leaf': 20,
        # 'device_type': 'gpu', # Use 'gpu' if GPU is available and LightGBM is compiled with GPU support
        'boosting_type': 'gbdt',
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'feature_fraction': 0.8,
        'n_estimators': 1000, # Max estimators, use with early stopping
        'verbosity': -1,
        'random_state': 42
    }
    LGBM_EARLY_STOPPING_ROUNDS = 50

    # --- Evaluation ---
    NDCG_K = 5

    # --- Ensemble Weights ---
    ENSEMBLE_XGB_WEIGHT = 0.6
    ENSEMBLE_LGB_WEIGHT = 0.4

    # --- Column Names ---
    TARGET_COL = 'relevance'
    GROUP_COL = 'srch_id'
    PROP_ID_COL = 'prop_id'
    # Columns to drop before final model training (excluding target and group)
    # These were dropped in the original 'prepare_final_features'
    COLS_TO_DROP_PRE_TRAIN = ['position', 'click_bool', 'booking_bool', 'gross_bookings_usd', 'price_usd', 'date_time']

    @staticmethod
    def create_dirs():
        """Creates necessary directories if they don't exist."""
        for path in [Config.PROCESSED_DATA_DIR, Config.MODELS_DIR, Config.SUBMISSIONS_DIR]:
            os.makedirs(path, exist_ok=True)