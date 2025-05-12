# /*****************************************************************************/
#  * File: app.py
#  * Author: Olavo Alves Barros Silva
#  * Contact: olavo.barros@ufv.com
#  * Date: 2025-05-11
#  * License: [License Type]
#  * Description: Main application file for the TreeLUT project.
# /*****************************************************************************/

import os
import sys
import numpy as np
from torchvision import datasets
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


from treelut import TreeLUTClassifier


def data_quatization(w_feature:float, X_train:np.array, X_test:np.array) -> np.array:
    """
    Quantize the data to a specified number of bits.
    Args:
        w_feature (float): Number of bits for quantization.
        X_train (np.array): Training data.
        X_test (np.array): Testing data.
    Returns:
        np.array: Quantized training and testing data.
    """
    scaler = MinMaxScaler()

    X_train_quantized = np.round(scaler.fit_transform(X_train)*(2**w_feature-1))
    X_test_quantized = np.clip(np.round(scaler.transform(X_test)*(2**w_feature-1)), 0, 2**w_feature-1)

    return X_train_quantized, X_test_quantized


def main():

    # Argument parsing


    

    xgb_clf = XGBClassifier(**xgb_params)
    xgb_clf.fit(X_train_quantized, y_train)
    y_pred_xgb = xgb_clf.predict(X_test_quantized)
    print(f"XGB Model Accuracy: {accuracy_score(y_pred_xgb, y_test):.3f}")