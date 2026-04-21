import joblib
import os
import pandas as pd
import optuna
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Custom modules
from cLogger import Log
from cModelManager import ModelManager

MODEL_NAME = "VAD2C_KNNClassifier"
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

def knn_train(X_train, X_test, y_train, y_test, k, w, p_val):
    model = KNeighborsClassifier(
        n_neighbors=k,
        weights=w,
        p=p_val,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average='macro')
    return macro_f1

def objective(trial):
    # 1. Define the search space
    k = trial.suggest_int('n_neighbors', 3, 150)
    w = trial.suggest_categorical('weights', ['uniform', 'distance'])
    p_val = trial.suggest_categorical('p', [1, 2])
    
    score = knn_train(X_train, X_test, y_train, y_test, k, w, p_val)
    return score

# Create study and optimize
log.log_message("Starting hyperparameter optimization with Optuna...")
sampler = optuna.samplers.TPESampler(n_startup_trials=25, seed=999)
study = optuna.create_study(sampler=sampler, direction="maximize")
study.optimize(objective, n_trials=250)
log.log_property("Best Parameters", study.best_params)
log.log_property("Best Macro F1", f"{study.best_value}")


#Train model with best parameters
winning_params = study.best_params
#winning_params['tree_method'] = 'hist'
#winning_params['n_jobs'] = -1

final_knn = KNeighborsClassifier(**winning_params)
final_knn.fit(X_train, y_train)
y_pred = final_knn.predict(X_test)

log.log_property("\nFinal Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
log.log_message(":\n")
log.log_property("Champion Classification Report", classification_report(y_test, y_pred))

# Save
knn_path = os.path.join(model_mngr.model_directory, 'knn_best_model.joblib')
scaler_path = os.path.join(model_mngr.model_directory, 'vad_scaler.joblib')
joblib.dump(final_knn, knn_path)
joblib.dump(scaler, scaler_path)

log.save()
log.save_txt()
log.log_message(f"Model and scaler saved to {model_mngr.model_directory}")
