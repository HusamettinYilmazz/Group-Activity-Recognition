from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def plot_conf_matrix(y_true, y_pred, class_names, save_path=None):
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Confusion matrix saved to {save_path}")

    plt.close(fig)

    return fig  

def get_f1_score(y_true, y_pred, average='weighted', report=False):
    if report:
       
        print("Classification Report:\n")
        print(classification_report(y_true, y_pred))
    else:
        f1 = f1_score(y_true, y_pred, average=average)
        return f1
