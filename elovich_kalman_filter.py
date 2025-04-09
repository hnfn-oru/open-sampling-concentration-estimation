#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 13:45:02 2024

@author: han
"""

import numpy as np

class KalmanFilterWithElovich:
    """
    Kalman filter module for signal processing using Elovich dynamics.
    Implements Unscented Kalman Filter (UKF) and Adaptive UKF.
    """

    def __init__(self, Q, R, x0, P0, ka, kd, a, b, dt, alpha=1e-3, beta=2, kappa=0):
        """
        Initialize the Kalman Filter with required parameters.
        
        Parameters:
            Q: np.ndarray
                Process noise covariance matrix.
            R: np.ndarray
                Measurement noise covariance matrix.
            x0: np.ndarray
                Initial state vector [C, R_g].
            P0: np.ndarray
                Initial covariance matrix.
            ka, kd, a, b: float
                Parameters for the dynamic model.
            dt: float
                Time step for the dynamics.
            alpha, beta, kappa: float
                UKF hyperparameters.
        """
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        self.ka = ka
        self.kd = kd
        self.a = a
        self.b = b
        self.dt = dt
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        # Derived parameters
        self.n = len(x0)  # State dimension
        self.L = self.n + self.kappa  # Scaling factor
        self.gamma = np.sqrt(self.L)  # Sigma point scaling
        
        # Weights for mean and covariance
        self.W_m = np.full(2 * self.n + 1, 1 / (2 * self.L))
        self.W_m[0] = self.kappa / self.L
        self.W_c = self.W_m.copy()
        self.W_c[0] += 1 - self.alpha**2 + self.beta

    def elovich_dynamics(self, C, R_g):
        """
        Compute nonlinear system dynamics using the Elovich equation.
        
        Parameters:
            C: float
                Concentration of the input gas.
            R_g: float
                Gas response.
        
        Returns:
            float: Updated gas response R_g.
        """
        dR_g_adsorption = self.a * np.exp(-self.b * R_g)  # Elovich adsorption term
        dR_g = self.ka * C - self.kd * R_g + dR_g_adsorption
        return R_g + dR_g * self.dt

    def ukf(self, y):
        """
        Run the Unscented Kalman Filter (UKF) on observations.
        
        Parameters:
            y: np.ndarray
                Observed MOX sensor responses.
        
        Returns:
            np.ndarray: State estimates over time.
        """
        x_est = np.zeros((len(y), self.n))  # Store state estimates

        for t in range(len(y)):
            # --- 1. Generate Sigma Points ---
            sigma_points = self._generate_sigma_points()

            # --- 2. Propagate Sigma Points Through Dynamics ---
            sigma_points_pred = self._propagate_sigma_points(sigma_points)

            # --- 3. Predicted Mean and Covariance ---
            x_pred, P_pred = self._predict_mean_and_covariance(sigma_points_pred)

            # --- 4. Transform Predicted Sigma Points into Observation Space ---
            z_pred, P_zz, P_xz = self._predict_observation(sigma_points_pred, x_pred)

            # --- 5. Measurement Update ---
            self._update(y[t], z_pred, P_zz, P_xz, x_pred, P_pred)

            x_est[t] = self.x

        return x_est

    def adaptive_ukf(self, y, k_Q=0.1, k_R=0.1):
        """
        Run the Adaptive Unscented Kalman Filter (AUKF) on observations.
        
        Parameters:
            y: np.ndarray
                Observed MOX sensor responses.
            k_Q, k_R: float
                Adaptation rates for process and measurement noise covariances.
        
        Returns:
            np.ndarray: State estimates over time.
        """
        x_est = np.zeros((len(y), self.n))  # Store state estimates

        for t in range(len(y)):
            # --- 1. Generate Sigma Points ---
            sigma_points = self._generate_sigma_points()

            # --- 2. Propagate Sigma Points Through Dynamics ---
            sigma_points_pred = self._propagate_sigma_points(sigma_points)

            # --- 3. Predicted Mean and Covariance ---
            x_pred, P_pred = self._predict_mean_and_covariance(sigma_points_pred)

            # --- 4. Transform Predicted Sigma Points into Observation Space ---
            z_pred, P_zz, P_xz = self._predict_observation(sigma_points_pred, x_pred)

            # --- 5. Measurement Update ---
            self._update(y[t], z_pred, P_zz, P_xz, x_pred, P_pred)

            # --- 6. Adaptive Noise Update ---
            self.Q = k_Q * np.cov(sigma_points_pred.T) + self.Q * 0.01
            self.R = k_R * np.var(y[:t + 1] - z_pred) + self.R * 0.01

            x_est[t] = self.x

        return x_est

    def _generate_sigma_points(self):
        """Generate sigma points based on the current state and covariance."""
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sqrt_P = np.linalg.cholesky(self.P)
        sigma_points[0] = self.x
        for i in range(self.n):
            sigma_points[i + 1] = self.x + self.gamma * sqrt_P[:, i]
            sigma_points[i + self.n + 1] = self.x - self.gamma * sqrt_P[:, i]
        return sigma_points

    def _propagate_sigma_points(self, sigma_points):
        """Propagate sigma points through the Elovich dynamics."""
        sigma_points_pred = np.zeros_like(sigma_points)
        for i, sp in enumerate(sigma_points):
            sigma_points_pred[i] = [
                sp[0],  # Concentration does not evolve
                self.elovich_dynamics(sp[0], sp[1])
            ]
        return sigma_points_pred

    def _predict_mean_and_covariance(self, sigma_points_pred):
        """Predict mean and covariance from propagated sigma points."""
        x_pred = np.sum(self.W_m[:, None] * sigma_points_pred, axis=0)
        P_pred = self.Q + sum(
            self.W_c[i] * np.outer(sigma_points_pred[i] - x_pred, sigma_points_pred[i] - x_pred)
            for i in range(2 * self.n + 1)
        )
        return x_pred, (P_pred + P_pred.T) / 2  # Ensure symmetry

    def _predict_observation(self, sigma_points_pred, x_pred):
        """Transform predicted sigma points into observation space."""
        z_sigma_points = sigma_points_pred[:, 1]
        z_pred = np.sum(self.W_m * z_sigma_points)
        P_zz = self.R + sum(
            self.W_c[i] * (z_sigma_points[i] - z_pred)**2 for i in range(2 * self.n + 1)
        )
        P_xz = sum(
            self.W_c[i] * np.outer(sigma_points_pred[i] - x_pred, z_sigma_points[i] - z_pred)
            for i in range(2 * self.n + 1)
        )
        return z_pred, P_zz, P_xz

    def _update(self, y_t, z_pred, P_zz, P_xz, x_pred, P_pred):
        """Update state and covariance based on observation."""
        observation = np.array([y_t]) if np.ndim(y_t) == 0 else y_t
        K = P_xz @ np.linalg.inv(P_zz)  # Kalman Gain
        self.x = x_pred + K @ (observation - z_pred)  # Update state estimate
        self.P = P_pred - K @ P_zz @ K.T  # Update covariance matrix
        self.P = (self.P + self.P.T) / 2  # Ensure symmetry


import torch
import torch.distributions as dist
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

import math
class HeavyTailNoiseModel(nn.Module):
    """学生t分布噪声模型"""
    def __init__(self, df_init=5.0):
        super().__init__()
        self.log_df = nn.Parameter(torch.log(torch.tensor(df_init)))
        
    def log_prob(self, residuals):
        df = torch.exp(self.log_df)
        return dist.StudentT(df=df, loc=0, scale=1.0).log_prob(residuals)


# ==============================================
class VariationalElovichUKF:
    """
    修复类型错误的变分自适应Elovich-UKF滤波器
    主要修改点：
    1. 强制类型转换：确保所有PyTorch运算使用Tensor类型
    2. 张量初始化：显式创建Tensor并保留计算图
    """
    
    def __init__(self, Q, R_init, x0, P0, ka, kd, a, b, dt):
        # UKF参数初始化
        self.n = 2  # 状态维度 [C, R_g]
        self.Q = Q  # 过程噪声协方差 (2x2)
        self.R = R_init  # 初始测量噪声方差
        self.x = x0.astype(np.float32)  # 初始状态 [C, R_g]
        self.P = P0.astype(np.float32)  # 初始协方差矩阵 (2x2)
        self.ka, self.kd = ka, kd  # 吸附/解吸系数
        self.a, self.b = a, b  # Elovich参数
        self.dt = dt
        # 在VOED-KF类中替换噪声模型
        self.noise_model = HeavyTailNoiseModel()
        
        # 变分参数初始化改进
        self.mu_R = torch.nn.Parameter(torch.log(torch.tensor(R_init, dtype=torch.float32))  )# 保持LogNormal均值
        self.log_var_R = torch.nn.Parameter(
                torch.ones(1, dtype=torch.float32) * 0.5  # 初始化更大的方差
                )
        self.optimizer = torch.optim.Adam(
                [self.mu_R, self.log_var_R], 
                lr=0.1,  # 初始学习率
                weight_decay=0.001  # 添加L2正则
                )

    def elovich_dynamics(self, C, R_g):
        """Elovich动力学方程"""
        dR_g_adsorption = self.a * np.exp(-self.b * R_g)
        dR_g = self.ka * C - self.kd * R_g + dR_g_adsorption
        return R_g + dR_g * self.dt

    def generate_sigma_points(self):
        """生成Sigma点"""
        n = self.n
        kappa = 3 - n
        sqrt_P = np.linalg.cholesky(self.P + kappa * np.eye(n))
        sigma_points = np.zeros((2*n+1, n))
        sigma_points[0] = self.x
        for i in range(n):
            sigma_points[i+1] = self.x + sqrt_P[:, i]
            sigma_points[i+1+n] = self.x - sqrt_P[:, i]
        return sigma_points.astype(np.float32)

    def predict_update(self, y):
        """单步预测和更新"""
        # 生成Sigma点
        sigma_points = self.generate_sigma_points()
        
        # 通过动力学方程传播Sigma点
        sigma_pred = np.zeros_like(sigma_points)
        for i, (C, R_g) in enumerate(sigma_points):
            sigma_pred[i, 0] = C  # 假设浓度不变
            sigma_pred[i, 1] = self.elovich_dynamics(C, R_g)
        
        # 计算预测均值和协方差
        x_pred = np.mean(sigma_pred, axis=0)
        P_pred = self.Q + np.cov(sigma_pred.T, ddof=0)
        
        # 观测预测（假设观测为R_g）
        z_pred = x_pred[1]
        P_zz = self.R + np.var(sigma_pred[:, 1])
        P_xz = np.cov(sigma_pred[:, 0], sigma_pred[:, 1])[0, 1]
        
        # 卡尔曼增益
        K = P_xz / P_zz
        
        # 状态更新
        self.x = x_pred + K * (y - z_pred)
        self.P = P_pred - np.outer(K, K) * P_zz
        
        return self.x, self.P

    def variational_update(self, y):
        """修复后的变分推断步骤"""
        # 显式类型转换
        y_tensor = torch.tensor(y, dtype=torch.float32)
        R_g_pred = torch.tensor(self.x[1], dtype=torch.float32)
        
        # # 计算对数似然（全部使用Tensor）
        # residual = (y_tensor - R_g_pred)**2
        # E_inv_R = torch.exp(-self.mu_R + 0.5 * torch.exp(self.log_var_R))
        # log_likelihood = -0.5 * (residual * E_inv_R + self.mu_R + math.log(2 * math.pi))
        
        # 使用学生t分布似然
        residuals = (y_tensor - R_g_pred) / torch.exp(self.mu_R)
        log_likelihood = self.noise_model.log_prob(residuals).sum()
        
        # 计算KL散度（使用Tensor参数）
        prior = dist.LogNormal(torch.tensor([0.0], dtype=torch.float32), 
                             torch.tensor([0.1], dtype=torch.float32))
        q_R = dist.LogNormal(self.mu_R, torch.exp(0.5 * self.log_var_R))
        kl = dist.kl_divergence(q_R, prior)
        
        # 优化ELBO
        loss = -(log_likelihood - kl)
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_([self.mu_R, self.log_var_R], max_norm=1.0)

        # 打印梯度信息（调试用）
        print(f"mu_R grad: {self.mu_R.grad.numpy()}, log_var_R grad: {self.log_var_R.grad.numpy()}")

        self.optimizer.step()
        
        # 更新R为后验均值（保留numpy兼容性）
        with torch.no_grad():
            self.R = torch.exp(self.mu_R + 0.5 * torch.exp(self.log_var_R)).item()

    def filter(self, y_sequence):
        """在线滤波主流程"""
        estimates = np.zeros((len(y_sequence), self.n))
        R_history = np.zeros(len(y_sequence))
        scheduler = CosineAnnealingLR(self.optimizer, T_max=len(y_sequence), eta_min=0.001)
        
        warmup_steps = 10000  # 前50步仅运行UKF
        for t, y in enumerate(y_sequence):
            # UKF预测和更新
            x_est, _ = self.predict_update(y)
            
            # 变分推断更新R
            # 阶段性变分更新
            if t >= warmup_steps:
                self.variational_update(y)
            
            estimates[t] = x_est
            R_history[t] = self.R
            
            # 更新学习率
            scheduler.step()
        
        return estimates, R_history


    
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_kf_results(kf_estimates, C_PID_real, R_g_real, starting_point=100):
    """
    Evaluate Kalman Filter results against PID sensor readings.

    Parameters:
        kf_estimates (np.ndarray): The Kalman Filter estimates (2D array, where column 0 is the MOX estimate).
        C_PID_real (np.ndarray): The real PID sensor readings (1D array).
        R_g_real (np.ndarray): The real MOX sensor readings (1D array).
        starting_point (int): The index from which to start the evaluation.

    Returns:
        dict: A dictionary containing the MSE, R² score, initial offset, and aligned signals.
    """
    # Extract and align results
    kf_mox = kf_estimates[starting_point:-1, 0]  # Filtered MOX responses
    C_pid = C_PID_real[starting_point:-1]       # True PID readings
    R_g = R_g_real[starting_point:-1]           # Measured MOX responses
    
    # Estimate the initial offset using the first available PID reading
    initial_offset = C_pid[0] - kf_mox[0]
    
    # Compute evaluation metrics
    mse = mean_squared_error(C_pid, kf_mox + initial_offset)
    r2 = r2_score(C_pid, kf_mox + initial_offset)
    
    # Return results
    return {
        "MSE": mse,
        "R2": r2,
        "offset_0": initial_offset,
        "virtual_sensor": kf_mox + initial_offset,
        "C_PID": C_pid,
        "R_g": R_g,
    } 


from pyts.decomposition import SingularSpectrumAnalysis

def ssa_kalman_pipeline(raw_signal, ssa_window_size, kalman_filter):
    """
    Coupled pipeline for Singular Spectrum Analysis and Elovich Kalman Filter.

    Parameters:
        raw_signal (np.ndarray): The raw sensor data to process.
        ssa_window_size (int): The window size for SSA decomposition.
        ssa_components (list[int]): Indices of SSA components to reconstruct the signal.
        kalman_filter (KalmanFilterWithElovich): An instance of the Elovich Kalman Filter.

    Returns:
        np.ndarray: Kalman filter state estimates over time.
    """
    # Step 1: Singular Spectrum Analysis (SSA) Decomposition
    ssa = SingularSpectrumAnalysis(window_size=ssa_window_size)
    components = ssa.fit_transform(raw_signal.reshape(1, -1))  # Shape (1, n_samples)

    # Step 2: Reconstruct signal using selected components
    reconstructed_signal = np.sum(components[0,0:-1], axis=0)

    # Step 3: Apply Kalman Filter
    kalman_estimates = kalman_filter.ukf(reconstructed_signal)

    return components, kalman_estimates

# Example Usage
# Assuming `C_PID_real` is your sensor signal (1D array) and `kf` is an instance of KalmanFilterWithElovich
# from elovich_kalman_filter import KalmanFilterWithElovich

# Initialize UKF parameters
# x0 = np.array([C_PID_real[0], 0.5])  # Initial state [C, R_g]
# P0 = np.eye(2) * 0.1  # Initial covariance
# Q = np.eye(2) * 0.01  # Process noise covariance
# R = np.array([[0.1]])  # Measurement noise covariance
# ka, kd, a, b, dt = 0.09, 0.04, 0.02, 0.01, 1  # Dynamic model parameters
# k_Q, k_R = 0.1, 0.1  # Adaptive UKF parameters

# ukf = KalmanFilterWithElovich(Q, R, x0, P0, ka, kd, a, b, dt)
# filtered_signal = ukf.ukf(R_g_real)  # 提取R_g分量

# from elovich_kalman_filter import evaluate_kf_results
# results = evaluate_kf_results(filtered_signal, C_PID_real, R_g_real, starting_point=100)

# import matplotlib.pyplot as plt
# # Plot the results
# plt.figure(figsize=(12, 8))
# #plt.subplot(2, 1, 1)
# plt.plot(results["C_PID"], alpha=0.3, label="True PID Readings (Ground Truth)")
# plt.plot(results["R_g"], alpha=0.3, label="MOX Readings")

# plt.plot(results["virtual_sensor"], label="UKF Estimated PID Concentration (Aligned)")
# plt.ylabel("Concentration (ppm) / Sensor Response")
# plt.grid(True)
# plt.legend()
# plt.title("Kalman Filter Evaluation with Offset Compensation")
# plt.show()

# Use singular spectrum analysis
# kf = KalmanFilterWithElovich(Q, R, x0, P0, ka, kd, a, b, dt)
# components, estimates = ssa_kalman_pipeline(raw_data, ssa_window_size=20,  kalman_filter=kf)
