import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

LABELS_PATH = "output/processing/custom_labels/crema-d/cremad_labels.csv"
OUTPUT_PATH = "output/processing/custom_labels/crema-d/cremad_labels_folds_6.csv"
FOLDS = 6

df = pd.read_csv(LABELS_PATH)
df['SpkrID'] = df['SpkrID'].str.removeprefix('RVS_')
df['SpkrID'] = df['SpkrID'].astype(int)
df['Fold'] = 0

sgkf = StratifiedGroupKFold(n_splits=FOLDS)

# Assign fold numbers (1 through 5)
for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X=df, y=df['Gender'], groups=df['SpkrID'])):
    df.loc[test_idx, 'Fold'] = fold_idx + 1
    
for i in range(1, FOLDS + 1):
    fold_data = df[df['Fold'] == i]
    actors = sorted(fold_data['SpkrID'].unique())
    print(f"Fold {i} Actors: {actors}")
    #print number of males and females
    male_count = fold_data[fold_data['Gender'] == 'Male'].shape[0]
    female_count = fold_data[fold_data['Gender'] == 'Female'].shape[0]
    print(f"Fold {i} - Males: {male_count}, Females: {female_count}, Total: {len(fold_data)}")


df.to_csv(OUTPUT_PATH, index=False)
