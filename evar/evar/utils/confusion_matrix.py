import pandas as pd
from sklearn.metrics import confusion_matrix as cm

# Example DataFrame
data = {'true': ['cat', 'dog', 'cat', 'cat', 'dog'],
        'pred': ['cat', 'dog', 'dog', 'cat', 'dog']}
df = pd.DataFrame(data)


def confusion_matrix(df: pd.DataFrame):
    # Get unique labels
    labels = sorted(df['true'].unique())

    # Compute confusion matrix
    conf_matrix = cm(df['true'], df['pred'], labels=labels)

    # Convert confusion matrix to DataFrame
    cm_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)

    print(cm_df)
    return cm_df
