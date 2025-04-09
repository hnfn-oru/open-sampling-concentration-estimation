# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 22:32:01 2025

@author: fanha
"""


import numpy as np

# Import the SensorDataProcessor class from a custom module
from LST_SensorDataProcessor import SensorDataProcessor

# Example usage:
processor = SensorDataProcessor(base_path="train/experiment")
# Prepare regression and target data for the sensor
data4reg = processor.prepare_regression_data(sensor_type='MiCS5524', num_layers=4)
data4target = processor.prepare_target_data(sensor_type='PID-sensor', num_layers=4)

# Split the data into training and testing sets
train_data_X, train_data_Y, test_data_X, test_data_Y = processor.train_test_split(
    data4reg, data4target, train_portion=0.8
)

seq_len = 80  # Sequence length for training data

# Prepare training and testing datasets
X_train, X_test, y_train, y_test = processor.prepare_training_data(
    data4reg, data4target, train_portion=0.8, seq_len=seq_len, target_layer=0
)

# Now, X_train, X_test, y_train, and y_test are ready for model training and testing
print("Training data shape:", X_train.shape, y_train.shape)
print("Testing data shape:", X_test.shape, y_test.shape)

# Replace these with your actual data arrays
C_PID_real = np.hstack((train_data_Y[:, 0], test_data_Y[:, 0]))  # Real PID readings
R_g_real = np.hstack((train_data_X[:, 0], test_data_X[:, 0]))    # Real MOX sensor responses

# Ensure time alignment (assuming equal-length datasets and uniform sampling)
time = np.linspace(0, len(C_PID_real)-1, len(C_PID_real))
    

from elovich_kalman_filter import KalmanFilterWithElovich

# Initialize UKF parameters
x0 = np.array([C_PID_real[0], 0.5])  # Initial state [C, R_g]
P0 = np.eye(2) * 0.1  # Initial covariance
Q = np.eye(2) * 0.01  # Process noise covariance
R = np.array([[0.1]])  # Measurement noise covariance
ka, kd, a, b, dt = 0.09, 0.04, 0.02, 0.01, 1  # Dynamic model parameters
k_Q, k_R = 0.1, 0.1  # Adaptive UKF parameters

# 1. Just adaptive UKF filtering.
ukf = KalmanFilterWithElovich(Q, R, x0, P0, ka, kd, a, b, dt)
filtered_signal = ukf.ukf(R_g_real)  # filtered raw MOX sensor responses

from elovich_kalman_filter import evaluate_kf_results
results = evaluate_kf_results(filtered_signal, C_PID_real, R_g_real, starting_point=100)

import matplotlib.pyplot as plt
# Plot the results
plt.figure(figsize=(12, 8))
#plt.subplot(2, 1, 1)
plt.plot(results["C_PID"], alpha=0.3, label="True PID Readings (Ground Truth)")
plt.plot(results["R_g"], alpha=0.3, label="MOX Readings")

plt.plot(results["virtual_sensor"], label="UKF Estimated PID Concentration (Aligned)")
plt.ylabel("Concentration (ppm) / Sensor Response")
plt.grid(True)
plt.legend()
plt.title("Kalman Filter Evaluation with Offset Compensation")
plt.show()


# 1. Clustering guided adaptive UKF filtering.
from clustering_guidence import ssa_clustering_basic, ssa_clustering_guidence_multistep

x_est, filtered_guided = ssa_clustering_guidence_multistep(
    R_g_real = R_g_real, 
    kalman_filter = ukf,
    eta = 0.00005,
    momentum_coef=0.8
)


results_with_guidence = evaluate_kf_results(filtered_guided, C_PID_real, R_g_real, starting_point=100)



# Plot the results
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(results["R_g"], alpha=0.85, label="MOX Readings (Baseline)")
plt.plot(results_with_guidence["C_PID"], alpha=0.85, label="PID ground truth")
plt.plot(results_with_guidence["virtual_sensor"], alpha=0.5, label="UKF Estimated and Clustering Guided PID Concentration Estimation (Aligned)")
plt.plot(results["virtual_sensor"], alpha=0.3, label="UKF Estimated PID Concentration (Aligned)")
plt.ylabel("Concentration (ppm) / Sensor Response")
plt.grid(True)
plt.legend()
plt.title("Kalman Filter Evaluation with Offset Compensation")
plt.show()