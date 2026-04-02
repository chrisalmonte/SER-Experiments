import joblib
import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Custom modules
from cLogger import Log
from cModelManager import ModelManager

MODEL_NAME = "VAD2C_RF"
MODELS_DIR = "output/models"

model_mngr = ModelManager(f"{MODELS_DIR}/{MODEL_NAME}")
log = Log(model_mngr.model_directory, prefix=MODEL_NAME)

EMOTION_MAP = {
    'N':0, 
    'H':1, 
    'S':2, 
    'A':3, 
    'F':4, 
    'D':5, 
    'C':6, 
    'U':7 
}

fitting_properties = {
    "labels": r'C:\Datasets\MSP-PODCAST-Publish-2.0\Labels\labels_consensus.csv',
    "drop_labels": ('EmoClass', ['X', 'O']), #Can be None
    "train_set": ['Train', 'Development'],
    "test_set": ['Test1', 'Test2'],
    "RF_Trees": 500,
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

# --- Train Random Forest ---
log.log_message("\nTraining Random Forest Classifier")
rf_model = RandomForestClassifier(
    n_estimators=fitting_properties["RF_Trees"],
    random_state=999
)

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# --- Evaluate ---
log.log_property("\nFinal Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
log.log_property("Macro F1", f"{f1_score(y_test, y_pred, average='macro'):.4f}")
log.log_message(":\n")
log.log_property("Classification Report", classification_report(y_test, y_pred))

# --- Save ---
rf_path = os.path.join(model_mngr.model_directory, 'rf_model.joblib')
scaler_path = os.path.join(model_mngr.model_directory, 'vad_scaler.joblib')
joblib.dump(rf_model, rf_path)
joblib.dump(scaler, scaler_path)

log.save()
log.save_txt()
log.log_message(f"Model and scaler saved to {model_mngr.model_directory}")
