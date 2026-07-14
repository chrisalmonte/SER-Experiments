import pandas as pd

def calculate_emoact_stats(input_csv, output_csv):
    # Load the data
    df = pd.read_csv(input_csv)
    
    # Group by EmoClass and EmoInt, then calculate mean and std for EmoAct
    agg_df = df.groupby(['EmoClass', 'EmoInt'])['EmoAct'].agg(['mean', 'std']).reset_index()
    
    # Rename the columns to match your desired output format
    agg_df.rename(columns={'mean': 'AvgEmoAct', 'std': 'StdEmoAct'}, inplace=True)
    
    # Save the resulting DataFrame to a new CSV file
    agg_df.to_csv(output_csv, index=False)
    print(f"Data successfully processed and saved to '{output_csv}'")
    
    return agg_df

# Example usage with the RAVDESS file:
input_file = r"C:\Users\emith\OneDrive - Instituto Politecnico Nacional\Tesis\Programas\SER-Experiments\output\models\wavlemo\intensity_test_CREMAD_vad_predictions.csv"
output_file = "ravdess_emoact_stats.csv"

# Run the function
result_df = calculate_emoact_stats(input_file, output_file)

# Display the first few rows of the result
print(result_df.head())