import pandas as pd
import numpy as np
import glob
import pickle
import re
import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(1, parent_dir)

#Custom imports
from cLogger import Log

MODEL_PATH = "output/models/WavLM_BP_Class_LoRa_RVS/Desp_I"
TOTAL_EPOCHS = 100

search_pattern = f"{MODEL_PATH}/F*/*.pkl"
matching_files = glob.glob(search_pattern)
folds_log = Log(f"{MODEL_PATH}", prefix="Overall_Metrics")

def aggregate_cross_val_metrics(parsed_logs_list, class_names, target_model="test_results_Best"):
    """
    Aggregates metrics across multiple folds.
    
    parsed_logs_list: List of dictionaries returned by `parse_ser_log`
    class_names: List of strings for the emotion labels
    target_model: Which set of test results to aggregate ('test_results_Best' or 'test_results_Final')
    """
    # Lists to collect metrics from all folds
    accuracies = []
    f1_scores = []
    precisions = []
    uars = []
    confusion_matrices = []
    
    for log_data in parsed_logs_list:
        metrics = log_data[target_model]
        
        accuracies.append(metrics["accuracy"])
        f1_scores.append(metrics["f1_macro"])
        precisions.append(metrics["precision_macro"])
        uars.append(metrics["recall_macro_uar"])
        confusion_matrices.append(metrics["confusion_matrix"])
        
    # Aggregate the Scalars (Mean ± Standard Deviation)
    agg_results = {
        "Folds_Aggregated": len(parsed_logs_list),
        "Overall_Accuracy": f"{np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}",
        "Overall_UAR": f"{np.mean(uars):.4f} ± {np.std(uars):.4f}",
        "Overall_F1_Macro": f"{np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}",
        "Overall_Precision": f"{np.mean(precisions):.4f} ± {np.std(precisions):.4f}"
    }
    
    # Aggregate the Confusion Matrices
    master_matrix = np.sum(confusion_matrices, axis=0)
    df_cm = pd.DataFrame(master_matrix, index=class_names, columns=class_names)
    df_cm.index.name = "True \\ Pred"
    
    agg_results["Master_Confusion_Matrix"] = master_matrix
    agg_results["Master_Confusion_Matrix_Str"] = f"\n\n{df_cm.to_string()}\n"
    
    return agg_results

folds = []

for filename in matching_files:
    with open(filename, 'rb') as file:
        log = pickle.load(file)

    fold_data = {
        "test_results_Best": {},
        "test_results_Final": {}
    }

    for section_name in ('Best', 'Final'):
        metrics = {}
        content = log.properties[f'Test_results ({section_name} up to {TOTAL_EPOCHS})']

        pattern = r'\s*\n\s*Accuracy:\s*([\d\.]+)\s*\n\s*F1_Score_Macro:\s*([\d\.]+)\s*\n\s*Precision_Macro:\s*([\d\.]+)\s*\n\s*Recall_Macro:\s*([\d\.]+)'
        match = re.search(pattern, content)
        
        if match:
            metrics["accuracy"] = float(match.group(0))
            metrics["f1_macro"] = float(match.group(1))
            metrics["precision_macro"] = float(match.group(2))
            metrics["recall_macro_uar"] = float(match.group(3))
        
        cm_pattern = r'.*?(?:Confusion_Matrix: )(\[\[.*?\]\])'
        cm_match = re.search(cm_pattern, content, re.DOTALL)
        if cm_match:
            # Clean the string and convert to NumPy array
            cm_str = cm_match.group(1).replace('[', '').replace(']', '')
            lines = cm_str.strip().split('\n')
            matrix_data = [[int(x) for x in line.split()] for line in lines]
            metrics["confusion_matrix"] = np.array(matrix_data)
        else:
            print(f"Confusion matrix not found in section '{section_name}' of file '{filename}'.")
        fold_data[f'test_results_{section_name}'] = metrics
    folds.append(fold_data)

for fold in range(len(folds)):
    folds_log.log_properties(f"Fold_{fold+1}_Metrics", folds[fold])

classes = ["Neutral", "Happiness", "Sadness", "Anger", "Fear", "Disgust", "Surprise"]
aggregated_metrics = aggregate_cross_val_metrics(folds, class_names=classes, target_model="test_results_Best")
folds_log.log_properties("Aggregated_CV_Metrics_Best", aggregated_metrics)
aggregated_metrics = aggregate_cross_val_metrics(folds, class_names=classes, target_model="test_results_Final")
folds_log.log_properties("Aggregated_CV_Metrics_Final", aggregated_metrics)
folds_log.save()
folds_log.save_txt()

# Save metrics pickle:
output_file = f"{MODEL_PATH}/overall_metrics.pkl"
with open(output_file, 'wb') as file:
    pickle.dump(folds, file)

