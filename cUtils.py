import numpy as np
import pandas as pd
import torch

class Imbalance():
    @staticmethod
    def smoothed_inverse_weights(class_counts):
        """
        Smoothed Inverse Frequency
        Formula: W = 1 / sqrt(Count)
        """
        counts = np.array(class_counts)

        # 1. Calculate raw weights
        raw_weights = 1.0 / np.sqrt(counts)

        # 2. Normalize weights (Sum of weights equals the number of classes)
        num_classes = len(counts)
        normalized_weights = raw_weights * (num_classes / np.sum(raw_weights))

        # Return as a PyTorch FloatTensor ready for the loss function
        return torch.tensor(normalized_weights, dtype=torch.float)

    @staticmethod
    def effective_number_weights(class_counts, beta=0.999):
        """
        Effective Number of Samples
        Formula: W = (1 - beta) / (1 - beta^Count)
        """
        counts = np.array(class_counts)

        # 1. Calculate the effective number of samples for each class
        effective_num = 1.0 - np.power(beta, counts)

        # 2. Calculate raw weights
        raw_weights = (1.0 - beta) / effective_num

        # 3. Normalize weights (Sum of weights equals the number of classes)
        num_classes = len(counts)
        normalized_weights = raw_weights * (num_classes / np.sum(raw_weights))

        return torch.tensor(normalized_weights, dtype=torch.float)
    
class DataFrames:
    @staticmethod
    def make_train_test(labels_train_path, labels_test_path=None, drop_labels=None, map_labels=None, 
                        train_partition=None, test_partition=None):
        labels_test_path = labels_test_path if labels_test_path else labels_train_path
        
        df_train = pd.read_csv(labels_train_path)
        df_test = pd.read_csv(labels_test_path)

        if drop_labels:
            df_train = df_train[~df_train[drop_labels[0]].isin(drop_labels[1])]
            df_test = df_test[~df_test[drop_labels[0]].isin(drop_labels[1])]
        if map_labels:
            df_train[map_labels[0]] = df_train[map_labels[0]].map(map_labels[1])
            df_test[map_labels[0]] = df_test[map_labels[0]].map(map_labels[1])         
        if train_partition:
            for partition in train_partition:
                df_train = df_train[df_train[partition[0]].isin(partition[1])]
        if test_partition:
            for partition in test_partition:
                df_test = df_test[df_test[partition[0]].isin(partition[1])]

        return (df_train, df_test)
