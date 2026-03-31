import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

LABELS_PATH = "output/processing/custom_labels/ravdess/ravdess_labels_speech.csv"

df = pd.read_csv(LABELS_PATH)
df['SpkrID'] = df['SpkrID'].astype(int)
df['Fold'] = 0

sgkf = StratifiedGroupKFold(n_splits=6)

# Assign fold numbers (1 through 6)
for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X=df, y=df['Gender'], groups=df['SpkrID'])):
    df.loc[test_idx, 'Fold'] = fold_idx + 1
    
for i in range(1, 7):
    fold_data = df[df['Fold'] == i]
    actors = sorted(fold_data['SpkrID'].unique())
    print(f"Fold {i} Actors: {actors}")

df.to_csv("output/processing/custom_labels/ravdess/labels_w_folds.csv", index=False)



# --- How to use it ---
# master_df = assign_ravdess_folds("your_dataset.csv")
# master_df.to_csv("ravdess_with_folds.csv", index=False)