"""
Course Number: ENGR 13300
Semester: Fall 2025

Description:
    Replace this line with a description of your program.

Assignment Information:
    Assignment:     11.2.2 tp3 team 1
    Team ID:        LC4 - 18 
    Author:         Arav Srivastava, sriva222@purdue.edu
                    Justin, 
                    Gina,
                    Ayona Kuriaksoe, akuriak@purude.edu
    Date:           10/10/2025

Contributors:
    Name, login@purdue [repeat for each]

    My contributor(s) helped me:
    [ ] understand the assignment expectations without
        telling me how they will approach it.
    [ ] understand different ways to think about a solution
        without helping me plan my solution.
    [ ] think through the meaning of a specific error or
        bug present in my code without looking at my code.
    Note that if you helped somebody else with their code, you
    have to list that person as a contributor here as well.

Academic Integrity Statement:
    I have not used source code obtained from any unauthorized
    source, either modified or unmodified; nor have I provided
    another student access to my code.  The project I am
    submitting is my own original work.
"""


import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

def load_dataset(file_path, feature_cols, label_col, shuffle, seed):
    """
    Loads a dataset from a CSV file, separates features and labels,
    and optionally shuffles the data.
    """
    df = pd.read_csv(file_path)
    X = df[feature_cols].to_numpy()
    y = df[label_col].to_numpy()

    if shuffle:
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(len(y))
        X = X[shuffled_indices]
        y = y[shuffled_indices]

    return X, y

def train_val_test_split(X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):

    if not train_ratio + val_ratio + test_ratio == 1:
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1

    array_length = len(X)

    train_end = int(array_length * train_ratio)
    val_end = int(array_length * (train_ratio + val_ratio))

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test

def scale_features(X_train, X_val, X_test):

    mean = np.mean(X_train, axis = 0)
    std_dev = np.std(X_train, axis = 0)

    X_train_scaled = (X_train - mean) / std_dev
    X_val_scaled = (X_val - mean) / std_dev
    X_test_scaled = (X_test - mean) / std_dev

    return X_train_scaled, X_val_scaled, X_test_scaled

def calculate_metrics(predicted_labels, true_labels):

    correct_predictions = 0

    for i in range(len(predicted_labels)):
        if predicted_labels[i] == true_labels[i]:
            correct_predictions += 1
    
    accuracy = correct_predictions / len(true_labels)

    error = 1 - accuracy

    return accuracy, error

def knn_single_prediction(new_example, X_train, y_train, k):


    distances = []
    for i in range(len(X_train)):
        distance = np.linalg.norm(new_example - X_train[i])
        distances.append((distance, y_train[i]))

    distances.sort(key=lambda x: x[0])

    k_nearest_labels = [label for _, label in distances[:k]] #something to look at again

    unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)

    predicted_label = unique_labels[np.argmax(counts)]
    
    return predicted_label

def predict_labels_knn(X_new, X_train, y_train, k):

    predicted = []


    for x in range(len(X_new)):
        predicted_label = knn_single_prediction(X_new[x], X_train, y_train, k)
        predicted.append(predicted_label)

    predicted_labels = np.array(predicted)
    
    return predicted_labels

def tune_k_values(k_values, X_train, y_train, X_val, y_val, is_shuffle):

    metrics = {
        "acc": {"train_acc": [], "val_acc": []},
        "error": {"train_error": [], "val_error": []}
    }
     
    # Loop through each k value
    for k in k_values:
        # Predict on training and validation sets using KNN
        train_pred = predict_labels_knn(X_train, X_train, y_train, k)
        val_pred = predict_labels_knn(X_val, X_train, y_train, k)

        # Compute metrics (accuracy & error)
        train_acc, train_error = calculate_metrics(train_pred, y_train)
        val_acc, val_error = calculate_metrics(val_pred, y_val)

        # Store metrics for plotting later
        metrics["acc"]["train_acc"].append(train_acc)
        metrics["acc"]["val_acc"].append(val_acc)
        metrics["error"]["train_error"].append(train_error)
        metrics["error"]["val_error"].append(val_error)

    # After testing all k values, visualize results
    plot_knn_performance(metrics, k_values)
    
    best_val = np.argmax(metrics["acc"]["val_acc"])
    best_k = k_values[best_val]

    return best_k
    

def plot_knn_performance(metrics, k_values):
 
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].plot(k_values, metrics["acc"]["train_acc"], marker ='o', label="Training Accuracy")
    axes[0].plot(k_values, metrics["acc"]["val_acc"], marker ='o', label="Validation Accuracy")
    axes[0].set_title("Accuracy vs. Number of Neighbours (k)", fontsize=14)
    axes[0].set_xlabel("k (Number of Neighbors)", fontsize=12)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].grid(True)


    axes[1].plot(k_values, metrics["error"]["train_error"], marker ='o', label="Training Error Rate")
    axes[1].plot(k_values, metrics["error"]["val_error"], marker = 'o', label="Validation Error Rate")
    axes[1].set_title("Error Rate vs. Number of Neighbors (k)", fontsize=14)
    axes[1].set_xlabel("k (Number of Neighbors)", fontsize=12)
    axes[1].set_ylabel("Error Rate", fontsize=12)
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def main():

    path = input("Enter the path to the feature dataset: ")
    shuffle = input("Shuffle the dataset? (yes/no): ")
    seed = int(input("Enter a seed for loading the dataset: "))

    if shuffle == "yes":
        shuffle = True
    else:
        shuffle = False


    feature_cols = ['hue_mean', 'hue_std', 'saturation_mean', 'saturation_std', 'value_mean', 'value_std', 'num_lines', 'has_circle']
    label_col = "ClassId"
    X, y = load_dataset(path, feature_cols, label_col, shuffle, seed)

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

    X_train_scaled, X_val_scaled, X_test_scaled = scale_features(X_train, X_val, X_test)

    print("\nData loaded and split into\n")
    print(f"  Training set:\n    size: {X_train_scaled.shape[0]}")
    print(f"  Validation set:\n    size: {X_val_scaled.shape[0]}")
    print(f"  Test set:\n    size: {X_test_scaled.shape[0]}")

    k_values = [1, 9, 20, 40, 80, 130, 200, 300, 500, 750, 1000]

    best_k = tune_k_values(k_values, X_train_scaled, y_train, X_val_scaled, y_val, shuffle)

    print(f"\nBased on the plots, the best k appears to be: {best_k}\n")
    print(f"Evaluating final model on the test set with k = {best_k}...")

    test_pred = predict_labels_knn(X_test_scaled, X_train_scaled, y_train, best_k)

    test_acc, test_error = calculate_metrics(test_pred, y_test)
    
    print("\n---Final Model Predictions---")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Set Error Rate: {test_error:.4f}")
if __name__ == "__main__":
    main()
