import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, \
    classification_report





# Calculate accuracy
def calculate_accuracy(true, predicted):
    return accuracy_score(true, predicted)


# Calculate F1 score (macro, micro, or weighted)
def calculate_f1(true, predicted, average='macro'):
    return f1_score(true, predicted, average=average)


# Calculate precision
def calculate_precision(true, predicted, average='macro'):
    return precision_score(true, predicted, average=average)


# Calculate recall
def calculate_recall(true, predicted, average='macro'):
    return recall_score(true, predicted, average=average)


# Generate a confusion matrix
def generate_confusion_matrix(true, predicted):
    return confusion_matrix(true, predicted)


# Generate a classification report
def generate_classification_report(true, predicted):
    return classification_report(true, predicted)


# Function to visualize confusion matrix
def plot_confusion_matrix(conf_matrix, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

