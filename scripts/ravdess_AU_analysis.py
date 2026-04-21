import pandas as pd
import os
import glob

EMOTION_MAP = {'01':'neutral', '02':'calm', '03':'happy', '04':'sad', '05':'angry', '06':'fearful', '07':'disgust', '08':'surprised'}
INTENSITY_MAP = {'01':'normal', '02':'strong'}
DROPPED_AUS = ['AU45']

def summarize_ravdess_extended(input_folder, output_file):
    summary_data = []
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        parts = file_name.replace('.csv', '').split('-')
        if len(parts) < 4: continue 
        
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        total_frames = len(df)

        trial_summary = {
            'filename': file_name,
            'emotion': EMOTION_MAP.get(parts[2], "unknown"),
            'instructed_intensity': INTENSITY_MAP.get(parts[3], "n/a"),
            'total_frames': total_frames
        }

        au_list = [col.replace('_r', '') for col in df.columns if col.endswith('_r') and col.replace('_r','') not in DROPPED_AUS]

        for au in au_list:
            p_col, i_col = f"{au}_c", f"{au}_r"

            # 1. Presence & Intensity
            active_frames = df[p_col].sum() if p_col in df.columns else 0
            trial_summary[f"{au}_AUPercent"] = round((active_frames / total_frames * 100), 2)
            trial_summary[f"{au}_AUMaxI"] = df[i_col].max() if i_col in df.columns else 0
            
            # 2. NEW: Standard Deviation (Intensity over time in THIS file)
            trial_summary[f"{au}_AUStdI"] = round(df[i_col].std(), 3) if i_col in df.columns else 0

        summary_data.append(trial_summary)

    final_df = pd.DataFrame(summary_data)
    final_df.to_csv(output_file, index=False)
    print(f"Extraction complete with Standard Deviation.")

def analyze_emotion_signatures(summary_csv):
    df = pd.read_csv(summary_csv)
    
    # 1. Identify all intensity columns (AUMaxI)
    intensity_cols = [c for c in df.columns if c.endswith('_AUMaxI')]
    
    # 2. Group by emotion and calculate the mean intensity for every AU
    emotion_means = df.groupby('emotion')[intensity_cols].mean()
    
    print("--- Top 10 Contributing AUs per Emotion ---")
    for emotion, row in emotion_means.iterrows():
        # Get top 10 AUs for this emotion
        top_10 = row.sort_values(ascending=False).head(10)
        
        # Clean up names for display (e.g., AU06_AUMaxI -> AU06)
        top_aus = [idx.replace('_AUMaxI', '') for idx in top_10.index]
        scores = [round(val, 2) for val in top_10.values]
        
        print(f"{emotion.upper()}:")
        for au, score in zip(top_aus, scores):
            print(f"  - {au} (Avg Max Intensity: {score})")
        print("-" * 30)

def rank_all_aus(summary_csv):
    df = pd.read_csv(summary_csv)
    
    # 1. Select the metrics we want to rank (Intensity and Duration)
    intensity_cols = [c for c in df.columns if c.endswith('_AUMaxI')]
    percent_cols = [c for c in df.columns if c.endswith('_AUPercent')]
    
    # 2. Group by emotion
    emotion_groups = df.groupby('emotion')

    print("=== FULL AU RANKING BY EMOTION (Sorted by Max Intensity) ===")
    
    for emotion, data in emotion_groups:
        print(f"\nRANKING FOR: {emotion.upper()}")
        print(f"{'Rank':<5} | {'AU':<7} | {'Avg Max Intensity':<18} | {'Avg % Duration':<15}")
        print("-" * 55)
        
        # Calculate mean for all AUs for this specific emotion
        means = data[intensity_cols].mean()
        durations = data[percent_cols].mean()
        
        # Sort AUs by intensity descending
        sorted_aus = means.sort_values(ascending=False)
        
        for i, (au_col, val) in enumerate(sorted_aus.items(), 1):
            au_name = au_col.replace('_AUMaxI', '')
            # Match the duration value for the same AU
            dur_val = durations.get(f"{au_name}_AUPercent", 0)
            
            print(f"{i:<5} | {au_name:<7} | {val:<18.3f} | {dur_val:<15.2f}%")
            
    # 3. Calculate cross-actor variability
    # High standard deviation here means different actors express this emotion very differently
    print("\n\n=== CONSISTENCY CHECK (Standard Deviation across all trials) ===")
    std_devs = df[intensity_cols].std().sort_values(ascending=False)
    print("AUs with the most variation (least consistent across actors):")
    print(std_devs.head(5))

def export_au_rankings_to_txt(summary_csv, output_txt):
    df = pd.read_csv(summary_csv)
    
    # 1. Prepare column groups
    intensity_cols = [c for c in df.columns if c.endswith('_AUMaxI')]
    percent_cols = [c for c in df.columns if c.endswith('_AUPercent')]
    
    emotion_groups = df.groupby('emotion')

    with open(output_txt, 'w') as f:
        f.write("===========================================================\n")
        f.write("      RAVDESS OPENFACE EMOTION PROFILE REPORT\n")
        f.write("===========================================================\n\n")

        for emotion, data in emotion_groups:
            f.write(f"EMOTION: {emotion.upper()}\n")
            f.write(f"{'-'*55}\n")
            f.write(f"{'Rank':<5} | {'AU':<7} | {'Avg Max Int.':<15} | {'Avg % Dur.':<10}\n")
            f.write(f"{'-'*55}\n")
            
            # Calculate averages
            means = data[intensity_cols].mean()
            durations = data[percent_cols].mean()
            
            # Sort by Intensity
            sorted_aus = means.sort_values(ascending=False)
            
            for i, (au_col, val) in enumerate(sorted_aus.items(), 1):
                au_name = au_col.replace('_AUMaxI', '')
                dur_val = durations.get(f"{au_name}_AUPercent", 0)
                
                f.write(f"{i:<5} | {au_name:<7} | {val:<15.3f} | {dur_val:<10.2f}%\n")
            
            # Summary Highlight
            top_au = sorted_aus.index[0].replace('_AUMaxI', '')
            f.write(f"\nSIGNATURE MOVEMENT: {top_au}\n")
            f.write(f"\n\n")

        # Global Consistency Check at the bottom
        f.write("===========================================================\n")
        f.write("        GLOBAL VARIABILITY (Consistency Check)\n")
        f.write("===========================================================\n")
        f.write("Higher values mean actors used this AU very differently.\n\n")
        
        std_devs = df[intensity_cols].std().sort_values(ascending=False)
        for au, val in std_devs.items():
            f.write(f"{au.replace('_AUMaxI', ''):<7}: {val:.3f}\n")

    print(f"Report successfully generated: {output_txt}")

import pandas as pd

def export_intensity_comparison_report(summary_csv, output_txt):
    df = pd.read_csv(summary_csv)
    
    # 1. Identify AU Intensity columns
    intensity_cols = [c for c in df.columns if c.endswith('_AUMaxI')]
    emotions = df['emotion'].unique()

    with open(output_txt, 'w') as f:
        f.write("===========================================================\n")
        f.write("    RAVDESS: NORMAL VS. STRONG INTENSITY COMPARISON\n")
        f.write("===========================================================\n")
        f.write("This report shows how much the Max Intensity (0-5 scale) \n")
        f.write("increases when an actor is told to perform 'Strongly'.\n\n")

        for emotion in sorted(emotions):
            f.write(f"EMOTION: {emotion.upper()}\n")
            f.write(f"{'-'*65}\n")
            f.write(f"{'Action Unit':<12} | {'Normal Avg':<12} | {'Strong Avg':<12} | {'Increase'}\n")
            f.write(f"{'-'*65}\n")
            
            # Filter data for this emotion
            emo_data = df[df['emotion'] == emotion]
            
            # Split by instructed intensity
            normal_data = emo_data[emo_data['instructed_intensity'] == 'normal']
            strong_data = emo_data[emo_data['instructed_intensity'] == 'strong']
            
            # Calculate means for both groups
            normal_means = normal_data[intensity_cols].mean()
            strong_means = strong_data[intensity_cols].mean()
            
            # Sort by the ones that show the biggest difference
            diffs = (strong_means - normal_means).sort_values(ascending=False)
            
            for au_col, diff in diffs.items():
                au_name = au_col.replace('_AUMaxI', '')
                n_val = normal_means[au_col]
                s_val = strong_means[au_col]
                
                # We only show AUs that actually moved (diff > 0.01) to keep it clean
                if diff > 0.01:
                    f.write(f"{au_name:<12} | {n_val:<12.3f} | {s_val:<12.3f} | +{diff:.3f}\n")
            
            # Identify the "Power AU" (The one that increased the most)
            if not diffs.empty:
                power_au = diffs.index[0].replace('_AUMaxI', '')
                f.write(f"\nMOST RESPONSIVE AU: {power_au}\n")
            f.write(f"\n\n")

    print(f"Intensity Comparison Report generated: {output_txt}")

#summarize_ravdess_extended(r'C:\Datasets\ravdess\FacialTracking_Actors_01-24', 'output/ravdess_AU_Summary_2.csv')
#analyze_emotion_signatures('output/ravdess_AU_Summary_2.csv')
#rank_all_aus('output/ravdess_AU_Summary_2.csv')
#export_au_rankings_to_txt('output/ravdess_AU_Summary_2.csv', 'output/ravdess_AU_Report.txt')
export_intensity_comparison_report('output/ravdess_AU_Summary_2.csv', 'output/ravdess_Intensity_Comparison_Report.txt')
#Activated AU's
#AuS max 
#Each aus max
