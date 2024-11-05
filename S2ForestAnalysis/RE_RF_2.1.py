#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, gdal_array
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
import rasterio
from rasterio.plot import show as rshow
import joblib

def print_data_size(name, data):
    print(f'Our {name} matrix is sized: {data.shape}')

#data_dir = os.chdir("C:\\_geodata\\scripts\\forest_analysis\\data")
# Define the working directory
wd = 'D:/DEV/python/AutoAtes/raster_data'

# Set the working directory for the script
os.chdir(wd)

# Read satellite imagery stack
img_ds = gdal.Open('stack.tif', gdal.GA_ReadOnly)
roi_ds = gdal.Open('training_data.tif', gdal.GA_ReadOnly)


# Clip input images to extent of training data
num_columns_roi = roi_ds.shape[1]
num_columns_img = img.shape[1]

if num_columns_roi == num_columns_img:
    print("The number of columns in roi and img match.")

    img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
                gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
    for b in range(img.shape[2]):
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()
    roi = roi_ds.GetRasterBand(1).ReadAsArray().astype(np.uint16)

    # Further processing and training data splitting
    n_samples = (roi < 2).sum()
    labels = np.unique(roi[roi < 2])
    X = img[roi < 2, :]
    y = roi[roi < 2]

    print_data_size('X', X)
    print_data_size('y', y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=109)

    print_data_size('X train', X_train)
    print_data_size('y train', y_train)
    print_data_size('X test', X_test)
    print_data_size('y test', y_test)

    # Hyperparameter grid
    param_grid = {
        'n_estimators': np.linspace(10, 200).astype(int),
        'max_depth': [None] + list(np.linspace(3, 20).astype(int)),
        'max_features': ['sqrt', None] + list(np.arange(0.5, 1, 0.1)),
        'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),
        'min_samples_split': [2, 5, 10],
        'bootstrap': [True, False]
    }
    # Estimator for use in random search
    estimator = RandomForestClassifier(random_state=50)

    # Create the random search model
    rs = RandomizedSearchCV(estimator, param_grid, n_jobs=-1, scoring='accuracy', cv=5, n_iter=10, verbose=1, random_state=50)
    rs.fit(X_train, y_train)
    rf_best = rs.best_estimator_

    importances = rf_best.feature_importances_
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Saving the model
    filename = 'REv1_RFv2.sav'
    joblib.dump(rf_best, filename)

    # Loading the model
    rf_best = joblib.load(filename)

    # Prediction and Metrics
    y_best = rf_best.predict(X_test)
    rf_probs = rf_best.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_probs)
    print(f'Random Forest: AUROC = {rf_auc}')
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_best)}")
    print(f"Precision: {metrics.precision_score(y_test, y_best, average='weighted')}")
    print(f"Recall: {metrics.recall_score(y_test, y_best, average='weighted')}")
    print(f"AUC: {metrics.roc_auc_score(y_test, rf_probs)}")

    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)

    # Reshape and predict for each pixel
    new_shape = (img.shape[0] * img.shape[1], img.shape[2])
    img_as_array = img[:, :, :].reshape(new_shape)
    class_prediction = rf_best.predict(img_as_array).reshape(img[:, :, 0].shape)
    rshow(class_prediction, cmap='Spectral', title='RF Class')

    # Save classification image to folder
    with rasterio.open('forestmask.tif', 'w', driver='GTiff', height=class_prediction.shape[0], width=class_prediction.shape[1], count=1, dtype=class_prediction.dtype, crs='+proj=latlong', transform=img_ds.GetGeoTransform()) as class_out:
        class_out.write(class_prediction.astype('uint16'), 1)
else:
    print("The number of columns in roi and img do not match.")
    print("Number of columns in roi:", num_columns_roi)
    print("Number of columns in img:", num_columns_img)