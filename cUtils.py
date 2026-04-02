import numpy as np
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
