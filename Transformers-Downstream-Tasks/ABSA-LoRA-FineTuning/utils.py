import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(predictions, labels):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def compute_ece(probs, labels, n_bins=15):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    ece = 0
    total_samples = len(labels)
    for bin_lower in np.linspace(0, 1, n_bins+1)[:-1]:
        bin_upper = bin_lower + 1/n_bins
        bin_mask = (confidences >= bin_lower) & (confidences < bin_upper)
        if np.any(bin_mask):
            bin_accuracy = np.mean(accuracies[bin_mask])
            bin_confidence = np.mean(confidences[bin_mask])
            bin_samples = np.sum(bin_mask)
            ece += (bin_samples / total_samples) * np.abs(bin_accuracy - bin_confidence)
    return ece
