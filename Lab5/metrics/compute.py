from sklearn.metrics import precision_recall_fscore_support
import numpy as np
def calculate_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    true = y_true
    p_macro, r_macro, f_macro, support_macro = precision_recall_fscore_support(true, preds, average='weighted')
    return f_macro