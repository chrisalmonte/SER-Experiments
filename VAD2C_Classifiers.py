import joblib
import os
import pandas as pd
import optuna
import xgboost as xgb
from enum import Enum
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.neighbors import NearestCentroid

# Custom modules
from cLogger import Log
from cModelManager import ModelManager
from cUtils import DataFrames

class Classifier(Enum):
    XGBOOST = 0
    KNN = 1
    RF = 2
    NC = 3

MODELS_DIR = "output/models"
DESCRIPTION = "Trained with 10 percent core of MSPP2"
C_MODE = Classifier.NC

#Define output paths
model_name = f"VAD2C_{C_MODE.name}"
model_mngr = ModelManager(f"{MODELS_DIR}/{model_name}")
log = Log(model_mngr.model_directory, prefix=model_name)
log.log_property("Model Name", model_name)
log.log_property("Description", DESCRIPTION)

EMOTION_MAP = {
    'N':0, # Neutral
    'H':1, # Happiness
    'S':2, # Sadness
    'A':3, # Anger
    'F':4, # Fear
    'D':5, # Disgust
    'C':6,  # Contempt
    'U':7   # Surprise
}
log.log_properties("Emotion Mapping", EMOTION_MAP, show=False)

data_properties = {
    "labels_train_path": 'output/processing/custom_labels/mspp2/outlier_analysis/labels_10p_core.csv',
    "labels_test_path": 'output/processing/custom_labels/mspp2/divided_labels_consensus_fs.csv',
    "drop_labels": ('EmoClass', ['X', 'O']),
    "map_labels": ('EmoClass', EMOTION_MAP),
    "test_partition": [('Split_Set', ['Test1'])],
    "test_partition": [('Split_Set', ['Test1', 'Test2'])],
}
log.log_properties("Data Properties", data_properties, show=False)

fitting_properties = {
    "scale": False,
    "SMOTE_K": None, #Can be None
    "SMOTE_strategy": { #Can be 'auto'
        2: 28000,
        7: 28000,
        6: 28000,
        5: 28000,
        4: 28000
    }
}
log.log_properties("Fitting Properties", fitting_properties, show=False)

optuna_properties = {
    "n_trials": 1000,
    "n_startup_trials": 150,
    "sampler_seed": 999,
    "direction": "maximize"
}
log.log_properties("Optuna Properties", optuna_properties, show=False)

df_train, df_test = DataFrames.make_train_test(**data_properties)

X_train = df_train[['EmoVal', 'EmoAct', 'EmoDom']]
y_train = df_train['EmoClass'] 
X_test = df_test[['EmoVal', 'EmoAct', 'EmoDom']]
y_test = df_test['EmoClass']

# Standardize the VAD scores to have a mean of 0 and a variance of 1.
if fitting_properties["scale"]:
    log.log_message("Standardizing VAD scores...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    scaler_path = os.path.join(model_mngr.model_directory, 'vad_scaler.joblib')
    joblib.dump(scaler, scaler_path)
    log.log_message(f"Scaler saved to {scaler_path}")

#SMOTE
if fitting_properties["SMOTE_K"]:    
    log.log_message(f"Applying SMOTE with k_neighbors={fitting_properties['SMOTE_K']}...")
    smote = SMOTE(random_state=999, k_neighbors=fitting_properties["SMOTE_K"], sampling_strategy=fitting_properties["SMOTE_strategy"])
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    log.log_message(f"Original Train Size: {len(X_train)}")
    log.log_message(f"SMOTE Train Size: {len(X_train_smote)}")
    X_train = X_train_smote
    y_train = y_train_smote

match C_MODE:
    case Classifier.XGBOOST:
        pass
    case Classifier.KNN:
        pass
    case Classifier.XGBOOST:
        pass
    case Classifier.NC:
        model_path = os.path.join(model_mngr.model_directory, 'centroid_model.joblib')
        log.log_message("\nTraining Nearest Centroid Classifier...")

        model = NearestCentroid() 
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
    case _:
        raise ValueError("Invalid Classifier Mode")

log.log_property("Final Macro F1", f"{f1_score(y_test, y_pred, average='macro'):.4f}")
final_report = classification_report(y_test, y_pred)
log.log_property("\nFinal Classification Report", f"\n\n{final_report}")

joblib.dump(model, model_path)
log.log_message(f"Model saved to {model_mngr.model_directory}")

log.save()
log.save_txt()
