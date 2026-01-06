import pandas as pd

def create_csv(set_name, dataframe):
    set_df = dataframe[dataframe['Split_Set'] == set_name]
    set_df_filtered = set_df.drop(columns=['Split_Set', 'Gender', 'SpkrID'])
    print(f"{set_name} set size: {len(set_df_filtered)}")

    set_df_VAD = set_df_filtered.drop(columns=['EmoClass'])
    set_df_category = set_df_filtered.drop(columns=['EmoAct', 'EmoVal', 'EmoDom'])
    
    set_df_filtered.to_csv(f'data/labels_{set_name.lower()}_combined.csv', index=False)
    set_df_VAD.to_csv(f'data/labels_{set_name.lower()}_VAD.csv', index=False)
    set_df_category.to_csv(f'data/labels_{set_name.lower()}_category.csv', index=False)
    print(f"CSV file created for {set_name} partition.")

#Read missing filenames
missing_files = []
with open("logs\msp-missing_files.txt", 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        file = line.split(':')[2].strip()
        missing_files.append(file)

df = pd.read_csv('C:/Datasets/MSP-PODCAST-Publish-2.0/Labels/labels_consensus.csv')
df_filtered = df[~df['FileName'].isin(missing_files)]

create_csv('Train', df_filtered)
create_csv('Development', df_filtered)
create_csv('Test1', df_filtered)
create_csv('Test2', df_filtered)
