#Script to choose x random samples from each class in MSP-Podcast dataset for undersampling experiments
import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(script_dir, '..', 'output', 'logs', 'msp-missing_files.txt')
results_path = os.path.join(script_dir, '..', 'data', 'labels_msp_consensus_undersampled.csv')

#Read missing filenames
missing_files = []
with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        file = line.split(':')[2].strip()
        missing_files.append(file)

df = pd.read_csv('C:/Datasets/MSP-PODCAST-Publish-2.0/Labels/labels_consensus.csv')
df_filtered = df[~df['FileName'].isin(missing_files)]

#Remove test and dev samples
#df_filtered = df_filtered[df_filtered['Split_Set'] == 'Train']
##Keep only name and targets
#df_filtered = df_filtered.drop(columns=['Split_Set', 'Gender', 'SpkrID'])
#
##Define classes and desired samples per class
#classes = df_filtered['EmoClass'].unique()
#samples_per_class = 500
#undersampled_dfs = []
#for emo_class in classes:
#    class_df = df_filtered[df_filtered['EmoClass'] == emo_class]
#    if len(class_df) > samples_per_class:
#        sampled_df = class_df.sample(n=samples_per_class, random_state=42)
#    else:
#        sampled_df = class_df
#    undersampled_dfs.append(sampled_df)
#undersampled_df = pd.concat(undersampled_dfs)
#
##Drop unneeded columns
#undersampled_df = undersampled_df.drop(columns=['EmoClass'])
#
#undersampled_df.to_csv(results_path, index=False)

#Select x random samples from test set for evaluation
test_df = df_filtered[df_filtered['Split_Set'] == 'Train']
test_df = test_df.drop(columns=['Split_Set', 'Gender', 'SpkrID'])
total_samples = 1500
if len(test_df) > total_samples:
    test_sampled_df = test_df.sample(n=total_samples, random_state=42)
else:
    test_sampled_df = test_df
test_results_path = os.path.join(script_dir, '..', 'data', 'labels_dev_msp_consensus_undersampled.csv')

#Drop unneeded columns
test_sampled_df = test_sampled_df.drop(columns=['EmoClass'])

test_sampled_df.to_csv(test_results_path, index=False)