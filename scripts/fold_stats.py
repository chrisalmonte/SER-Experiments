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

MODEL_PATH = "output/models/WavLM_BP_IntClass_RVS/Mod_I"
TOTAL_EPOCHS = 50
CLASSES = ["Normal", "Strong"]

search_pattern = f"{MODEL_PATH}/F*/*.pkl"
matching_files = glob.glob(search_pattern)
folds_log = Log(f"{MODEL_PATH}", prefix="Overall_Metrics")

def aggregate_cross_val_metrics(parsed_logs_list, class_names, target_model="test_results_Best"):
    """
    Aggregates metrics across multiple folds.
    """
    accuracies = []
    f1_scores = []
    precisions = []
    uars = []
    confusion_matrices = []
    
    for log_data in parsed_logs_list:
        metrics = log_data.get(target_model)
        # Skip if the metrics failed to parse for this fold
        if not metrics:
            continue
            
        accuracies.append(metrics["accuracy"])
        f1_scores.append(metrics["f1_macro"])
        precisions.append(metrics["precision_macro"])
        uars.append(metrics["recall_macro_uar"])
        confusion_matrices.append(metrics["confusion_matrix"])
        
    agg_results = {
        "Folds_Aggregated": len(accuracies),
        "Overall_Accuracy": f"{np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}",
        "Overall_UAR": f"{np.mean(uars):.4f} ± {np.std(uars):.4f}",
        "Overall_F1_Macro": f"{np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}",
        "Overall_Precision": f"{np.mean(precisions):.4f} ± {np.std(precisions):.4f}"
    }
    
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
        # Fetch the formatted string from the logger
        content = log.properties.get(f'Test_results ({section_name} up to {TOTAL_EPOCHS})', "")

        # 1. Extract the scalar metrics (Notice the groups start at 1, 2, 3, 4!)
        pattern = r'Accuracy:\s*([\d\.]+)\s*\n\s*F1_Score_Macro:\s*([\d\.]+)\s*\n\s*Precision_Macro:\s*([\d\.]+)\s*\n\s*Recall_Macro:\s*([\d\.]+)'
        match = re.search(pattern, content)
        
        if match:
            metrics["accuracy"] = float(match.group(1))
            metrics["f1_macro"] = float(match.group(2))
            metrics["precision_macro"] = float(match.group(3))
            metrics["recall_macro_uar"] = float(match.group(4))
        else:
            print(f"Warning: Scalar metrics not found in section '{section_name}' of '{filename}'.")
            
        # 2. Extract the Confusion Matrix
        cm_pattern = r'.*?(?:Confusion_Matrix:\s*)(\[\[.*?\]\])'
        cm_match = re.search(cm_pattern, content, re.DOTALL)
        
        if cm_match:
            cm_str = cm_match.group(1).replace('[', '').replace(']', '')
            lines = cm_str.strip().split('\n')
            matrix_data = [[int(x) for x in line.split()] for line in lines]
            metrics["confusion_matrix"] = np.array(matrix_data)
        else:
            print(f"Warning: Confusion matrix not found in section '{section_name}' of '{filename}'.")
            
        fold_data[f'test_results_{section_name}'] = metrics
        
    folds.append(fold_data)

# --- Logging the Results ---
for fold_idx, fold in enumerate(folds):
    folds_log.log_properties(f"Fold_{fold_idx+1}_Metrics", fold)

aggregated_metrics_best = aggregate_cross_val_metrics(folds, class_names=CLASSES, target_model="test_results_Best")
folds_log.log_properties("Aggregated_CV_Metrics_Best", aggregated_metrics_best)

aggregated_metrics_final = aggregate_cross_val_metrics(folds, class_names=CLASSES, target_model="test_results_Final")
folds_log.log_properties("Aggregated_CV_Metrics_Final", aggregated_metrics_final)

folds_log.save()
folds_log.save_txt()

# Save metrics pickle
output_file = f"{MODEL_PATH}/overall_metrics.pkl"
with open(output_file, 'wb') as file:
    pickle.dump(folds, file)

print(f"Results saved to {output_file}")
