import joblib
import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

#Custom modules
from cLogger import Log
from cModelManager import ModelManager

MODEL_NAME = "VAD2C_KNN"
MODELS_DIR = "output/models"

#Define output paths
model_mngr = ModelManager(f"{MODELS_DIR}/{MODEL_NAME}")
log = Log(model_mngr.model_directory, prefix=MODEL_NAME)

fitting_properties = {
    "K": 10,
    "weights": 'distance',
    "labels": r'C:\Datasets\MSP-PODCAST-Publish-2.0\Labels\labels_consensus.csv',
    "drop_labels": ('EmoClass', ['X', 'O']), #Can be None
    "train_set": ['Train', 'Development'],
    "test_set": ['Test1', 'Test2'],
    "SMOTE_K": 5, #Can be None
}
log.log_properties("Fitting Properties", fitting_properties, show=False)

df = pd.read_csv(fitting_properties["labels"])

if fitting_properties["drop_labels"]:
    column, labels = fitting_properties["drop_labels"]
    df = df[~df[column].isin(labels)]

df_train = df[df['Split_Set'].isin(fitting_properties["train_set"])]
df_test = df[df['Split_Set'].isin(fitting_properties["test_set"])]

X_train = df_train[['EmoVal', 'EmoAct', 'EmoDom']]
y_train = df_train['EmoClass'] 
X_test = df_test[['EmoVal', 'EmoAct', 'EmoDom']]
y_test = df_test['EmoClass']

# Standardize the VAD scores to have a mean of 0 and a variance of 1.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#SMOTE
if fitting_properties["SMOTE_K"]:    
    log.log_message(f"Applying SMOTE with k_neighbors={fitting_properties['SMOTE_K']}...")
    smote = SMOTE(random_state=999, k_neighbors=fitting_properties["SMOTE_K"])
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    
    log.log_message(f"Original Train Size: {len(X_train_scaled)}")
    log.log_message(f"SMOTE Train Size: {len(X_train_smote)}")
    X_train_scaled = X_train_smote
    y_train = y_train_smote

knn = KNeighborsClassifier(n_neighbors=fitting_properties["K"], weights=fitting_properties["weights"]) 
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

log.log_property("Overall Accuracy", f"\n{accuracy_score(y_test, y_pred):.4f}")
log.log_property("Classification Report", f"\n{classification_report(y_test, y_pred)}")

knn_path = os.path.join(model_mngr.model_directory, 'knn_emotion_model.joblib')
scaler_path = os.path.join(model_mngr.model_directory, 'vad_scaler.joblib')

joblib.dump(knn, knn_path)
joblib.dump(scaler, scaler_path)
log.save()
log.save_txt()
