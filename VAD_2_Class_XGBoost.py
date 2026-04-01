import joblib
import os
import pandas as pd
import optuna
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Custom modules
from cLogger import Log
from cModelManager import ModelManager

MODEL_NAME = "VAD2C_XGBoost"
MODELS_DIR = "output/models"

#Define output paths
model_mngr = ModelManager(f"{MODELS_DIR}/{MODEL_NAME}")
log = Log(model_mngr.model_directory, prefix=MODEL_NAME)

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

fitting_properties = {
    "labels": r'C:\Datasets\MSP-PODCAST-Publish-2.0\Labels\labels_consensus.csv',
    "drop_labels": ('EmoClass', ['X', 'O']), #Can be None
    "train_set": ['Train', 'Development'],
    "test_set": ['Test1', 'Test2'],
    "SMOTE_K": 5, #Can be None
    "SMOTE_strategy": { #Can be 'auto'
        2: 28000,
        7: 28000,
        6: 28000,
        5: 28000,
        4: 28000
    }
}
log.log_properties("Fitting Properties", fitting_properties, show=False)

df = pd.read_csv(fitting_properties["labels"])

if fitting_properties["drop_labels"]:
    column, labels = fitting_properties["drop_labels"]
    df = df[~df[column].isin(labels)]
df['EmoClass'] = df['EmoClass'].map(EMOTION_MAP)

df_train = df[df['Split_Set'].isin(fitting_properties["train_set"])]
df_test = df[df['Split_Set'].isin(fitting_properties["test_set"])]

X_train = df_train[['EmoVal', 'EmoAct', 'EmoDom']]
y_train = df_train['EmoClass'] 
X_test = df_test[['EmoVal', 'EmoAct', 'EmoDom']]
y_test = df_test['EmoClass']

# Standardize the VAD scores to have a mean of 0 and a variance of 1.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#SMOTE
if fitting_properties["SMOTE_K"]:    
    log.log_message(f"Applying SMOTE with k_neighbors={fitting_properties['SMOTE_K']}...")
    smote = SMOTE(random_state=999, k_neighbors=fitting_properties["SMOTE_K"], sampling_strategy=fitting_properties["SMOTE_strategy"])
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    log.log_message(f"Original Train Size: {len(X_train)}")
    log.log_message(f"SMOTE Train Size: {len(X_train_smote)}")
    X_train = X_train_smote
    y_train = y_train_smote


def xgboost_train(X_train, X_test, y_train, y_test, n_e, l_r, m_d, s, c_b, r_a, r_l, r_s, g):
    model = xgb.XGBClassifier(
        n_estimators=n_e,  # más árboles pero suaves
        learning_rate=l_r,  # bajo para evitar sobreajuste
        max_depth=m_d,  # árboles pequeños
        subsample=s,  # usar parte de los datos
        colsample_bytree=c_b,  # usar parte de las features
        reg_alpha=r_a,  # L1 regularización
        reg_lambda=r_l,  # L2 regularización
        random_state=r_s,
        gamma=g,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def objective(trial):
    n_e = trial.suggest_int('n_estimators', 200, 1000)
    l_r = trial.suggest_float('learning_rate', 0.001, 0.2, log=True)
    m_d = trial.suggest_int('max_depth', 4, 10)
    s = trial.suggest_float('subsample', 0.7, 1.0)
    c_b = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    r_a = trial.suggest_float('reg_alpha', 0, 2)
    r_l = trial.suggest_float('reg_lambda', 0.1, 5.0, log=True)
    g = trial.suggest_float('gamma', 0.0, 5.0)
    r_s = 999
    
    score = xgboost_train(X_train, X_test, y_train, y_test, n_e, l_r, m_d, s, c_b, r_a, r_l, r_s, g)

    return score

# Create study and optimize
log.log_message("Starting hyperparameter optimization with Optuna...")
sampler = optuna.samplers.TPESampler(n_startup_trials=20, seed=999)
study = optuna.create_study(sampler=sampler, direction="maximize")
study.optimize(objective, n_trials=200)
log.log_property("Best Parameters", study.best_params)
log.log_property("Best Macro F1", f"{study.best_value}")

#Train model with best parameters
winning_params = study.best_params
#winning_params['tree_method'] = 'hist'
#winning_params['n_jobs'] = -1
winning_params['random_state'] = 999

final_xgb = xgb.XGBClassifier(**winning_params)
final_xgb.fit(X_train, y_train)
y_pred = final_xgb.predict(X_test)

log.log_property("\nFinal Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
log.log_message(":\n")
log.log_property("Champion Classification Report", classification_report(y_test, y_pred))

# Save
xgb_path = os.path.join(model_mngr.model_directory, 'xgboost_best_model.joblib')
scaler_path = os.path.join(model_mngr.model_directory, 'vad_scaler.joblib')
joblib.dump(final_xgb, xgb_path)
joblib.dump(scaler, scaler_path)
log.log_message(f"Model and scaler saved to {model_mngr.model_directory}")

log.save()
log.save_txt()

