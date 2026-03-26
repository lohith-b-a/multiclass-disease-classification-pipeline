import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def compute_sample_weights(y):
    """
    Compute sample weights for imbalanced multiclass classification.
    """
    classes = np.unique(y)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )

    class_weights = dict(zip(classes, weights))

    sample_weights = np.array([class_weights[label] for label in y])

    return sample_weights, class_weights