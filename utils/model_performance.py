import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_auc(y_true, y_pred, title='Receiver Operating Characteristic'):
    """
    Plot the AUC-ROC curve for binary classification results.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1)
    y_pred : array-like
        Target scores/probability estimates of the positive class
    title : str, optional
        Title of the plot (default: 'Receiver Operating Characteristic')
    save_path : str, optional
        If provided, saves the plot to this path
        
    Returns:
    --------
    float
        The Area Under the Curve (AUC) score
    """
    
    # Calculate ROC curve points
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    
    # Plot diagonal line (randomly guessing)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Guess')
    
    # Customize the plot
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)       
    plt.show()
    
    return roc_auc