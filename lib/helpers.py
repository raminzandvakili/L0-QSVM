import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def plot_weight_matrices(W, b, display=True, title=None, save_path=None):
    """
    Plots the weight matrix W as a heatmap and the linear term b as a column heatmap.

    Parameters:
    - W: numpy.ndarray
        The quadratic weight matrix to plot.
    - b: numpy.ndarray
        The linear term to plot as a column.
    - title: str
        The title for the plot.
    - display: bool
        Whether to display the plot.
    - save_path: str
        Path to save the plot, if provided.
    """
    # Compute the absolute values of W and b
    W_abs = np.abs(W)
    b_abs = np.abs(b)

    # Reshape b to a column if it's a flattened array
    if b.ndim == 1:
        b_abs = b_abs.reshape(-1, 1)

    # Set up the matplotlib figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [9, 1]})

    # Create a colormap
    cmap = 'Greys'

    # Decide whether to annotate based on matrix size
    annot = W.shape[0] <= 20 and W.shape[1] <= 20

    # Plot the heatmap for W
    sns.heatmap(
        W_abs,
        cmap=cmap,
        annot=annot,
        fmt=".2f",
        square=True,
        cbar_kws={"shrink": .8},
        linewidths=.5,
        vmin=0,
        vmax=np.max(W_abs) if np.max(W_abs) != 0 else 1,
        ax=axes[0]
    )
    axes[0].set_title("W", fontsize=14)
    axes[0].set_xlabel("Features")
    axes[0].set_ylabel("Features")

    # Plot the heatmap for b as a column
    sns.heatmap(
        b_abs,
        cmap=cmap,
        annot=annot,
        fmt=".2f",
        square=True,
        cbar_kws={"shrink": .8},
        linewidths=.5,
        vmin=0,
        vmax=np.max(b_abs) if np.max(b_abs) != 0 else 1,
        ax=axes[1]
    )
    axes[1].set_title("b", fontsize=14)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Features")
    axes[1].set_yticklabels(axes[0].get_yticklabels())  # Sync y-ticks with W

    # Add an overall title if provided
    if title is not None:
        fig.suptitle(title, fontsize=16, y=1.02)

    # Adjust layout
    plt.tight_layout()

    # Save the plot if a save path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    # Show the plot
    if display:
        plt.show()


def log_eval_stats(X, y, pipeline, param_grid, n_tests):
    logging.info("------------EVAL STATS------------")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)
    # Collect results for accuracy, precision, recall, and F1
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_f1_scores = []

    # Iterate over the StratifiedKFold splits
    for train_index, test_index in cv.split(X, y):
        # Split data into training (3 parts), validation (1 part), and testing (remaining 1 part)
        X_train, y_train= X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        
        # Perform grid search with cross-validation on the training + validation folds
        random_search = RandomizedSearchCV(pipeline, param_grid, scoring='accuracy', 
                                        n_jobs=-1, n_iter=n_tests, verbose=0, random_state=1234)
        random_search.fit(X_train, y_train)
        
        # Evaluate the best model on the test fold
        y_pred = random_search.predict(X_test)
        test_accuracies.append(accuracy_score(y_test, y_pred))
        test_precisions.append(precision_score(y_test, y_pred, average='weighted'))
        test_recalls.append(recall_score(y_test, y_pred, average='weighted'))
        test_f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

    # Calculate and print mean and standard deviation for each metric
    mean_accuracy = np.mean(test_accuracies)
    std_accuracy = np.std(test_accuracies)
    mean_precision = np.mean(test_precisions)
    std_precision = np.std(test_precisions)
    mean_recall = np.mean(test_recalls)
    std_recall = np.std(test_recalls)
    mean_f1_score = np.mean(test_f1_scores)
    std_f1_score = np.std(test_f1_scores)

    logging.info(f"Mean test accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    logging.info(f"Mean test precision: {mean_precision:.4f} ± {std_precision:.4f}")
    logging.info(f"Mean test recall: {mean_recall:.4f} ± {std_recall:.4f}")
    logging.info(f"Mean test F1 score: {mean_f1_score:.4f} ± {std_f1_score:.4f}")


def setup_logging(log_file="experiment.log", level=logging.INFO):
    """
    Sets up logging to write messages to a file.
    
    Parameters:
    - log_file: str, path to the log file.
    - level: logging level (e.g., logging.INFO, logging.DEBUG).
    """
    # Remove any existing handlers to reset logging configuration
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure logging to write to file
    logging.basicConfig(
        filename=log_file,
        filemode='w',  # 'w' to overwrite each run; use 'a' to append
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Also add a stream handler if you want console output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    # Log the start of the logging session
    logging.info("Logging initialized. Writing to %s", os.path.abspath(log_file))
