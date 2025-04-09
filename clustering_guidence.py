# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:04:47 2025

@author: fanha
"""
import numpy as np
from scipy.stats import multivariate_normal

class ClassifierGuidance:
    def __init__(self, mu, Sigma, pi):
        """
        Initialize Classifier Guidance.
        
        Parameters:
        - mu: List of means for GMM, each of shape (n,).
        - Sigma: List of covariance matrices for GMM, each of shape (n, n).
        - pi: List of prior probabilities for GMM.
        """
        self.mu = mu  # GMM means
        self.Sigma = Sigma  # GMM covariance matrices
        self.pi = pi  # GMM prior probabilities

    def compute_gradient_log_p_c_given_x(self, x, c):
        """
        Compute the gradient ∇x log p(c|x).
        
        Parameters:
        - x: Current state estimate, shape (n,).
        - c: Target class.
        
        Returns:
        - grad_log_p_c_given_x: Gradient, shape (n,).
        """
        K = len(self.pi)  # Number of classes
        p_c_given_x = self._compute_p_c_given_x(x)
        
        # Compute ∇x log p(c|x)
        grad_log_p_c_given_x = -np.linalg.inv(self.Sigma[c]) @ (x - self.mu[c])
        for k in range(K):
            grad_log_p_c_given_x -= p_c_given_x[k] * (-np.linalg.inv(self.Sigma[k]) @ (x - self.mu[k]))
        
        return grad_log_p_c_given_x

    def _compute_p_c_given_x(self, x):
        """
        Compute the output probability p(c|x) for GMM.
        
        Parameters:
        - x: Current state estimate, shape (n,).
        
        Returns:
        - p_c_given_x: Probability for each class, shape (K,).
        """
        K = len(self.pi)  # Number of classes
        p_c_given_x = np.zeros(K)
        for c in range(K):
            p_c_given_x[c] = self.pi[c] * multivariate_normal.pdf(x, mean=self.mu[c], cov=self.Sigma[c])
        p_c_given_x /= np.sum(p_c_given_x)  # Normalize
        return p_c_given_x

    def predict_class(self, x):
        """
        Predict the class of the current state x.
        
        Parameters:
        - x: Current state estimate, shape (n,).
        
        Returns:
        - c: Predicted class.
        """
        p_c_given_x = self._compute_p_c_given_x(x)
        return np.argmax(p_c_given_x)  # Return the class with the highest probability


    
from scipy.stats import gamma

class ConcentrationPriorModel:
    def __init__(self, alpha, beta, lower_bound=0.05, upper_bound=0.3,  
                 initial_weight=1.35, total_steps=60000, annealing_type='linear'):
        """
        Initializes the prior model with a truncated Gamma distribution and sets up parameters for
        dynamic weighting using an annealing schedule.

        Parameters:
        - alpha (float): Shape parameter of the Gamma distribution (alpha > 0).
        - beta (float): Rate parameter of the Gamma distribution (beta > 0).
        - lower_bound (float): Lower truncation bound for the Gamma distribution.
        - upper_bound (float): Upper truncation bound for the Gamma distribution.
        - initial_weight (float): Initial weight for the prior gradient (default is 1.35).
        - total_steps (int): Total number of time steps for the annealing process (default is 60000).
        - annealing_type (str): Type of annealing schedule to use; options are 'linear' or 'cosine'.
        """
        self.alpha = alpha
        self.beta = beta
        self.initial_weight = initial_weight
        self.total_steps = total_steps
        self.annealing_type = annealing_type
        
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # Compute the normalization constant for the truncated Gamma distribution
        self.normalization_constant = (
            gamma.cdf(upper_bound, a=alpha, scale=1/beta) -
            gamma.cdf(lower_bound, a=alpha, scale=1/beta)
        )
        
    def compute_grad_log_likelihood(self, x):
        """
        Computes the gradient of the log-likelihood (log p(x)) of the truncated Gamma distribution
        with respect to x. This represents the prior gradient.

        Parameters:
        - x (float or np.ndarray): Gas concentration value(s).

        Returns:
        - grad (float or np.ndarray): Gradient of the log-likelihood with respect to x.
        """
        # Ensure x is within the specified bounds
        x = np.clip(x, self.lower_bound, self.upper_bound)
        
        # Compute the gradient: ∇_x log p(x) = (alpha - 1) / x - beta
        grad = (self.alpha - 1) / x - self.beta
        return grad
    
    def compute_weight(self, t):
        """
        Computes the weight for the prior gradient at a given time step t using the specified
        annealing schedule.

        Parameters:
        - t (int): Current time step.

        Returns:
        - weight (float): Computed weight for the prior gradient.
        """
        if self.annealing_type == 'linear':
            # Linear annealing: weight decreases linearly over time
            weight = self.initial_weight * (1 - t / self.total_steps)
        elif self.annealing_type == 'cosine':
            # Cosine annealing: weight follows a cosine decay
            weight = self.initial_weight * (1 + np.cos(np.pi * t / self.total_steps)) / 2
        else:
            raise ValueError("annealing_type must be 'linear' or 'cosine'")
        
        # Ensure the weight is non-negative
        weight = max(weight, 0)
        return weight

    
    
class EnhancedClassifierGuidance:
    def __init__(self, gmm_model, concentration_prior_model):
        """
        Initialize the EnhancedClassifierGuidance with a trained Gaussian Mixture Model (GMM)
        and a concentration prior model.

        Parameters:
        - gmm_model: Trained GMM classifier.
        - concentration_prior_model: Instance of the ConcentrationPriorModel.
        """
        self.gmm = gmm_model
        self.concentration_prior_model = concentration_prior_model

        # Extract parameters from the GMM
        self.mu = self.gmm.means_            # Means of each Gaussian component
        self.Sigma = self.gmm.covariances_   # Covariance matrices of each component
        self.pi = self.gmm.weights_          # Weights (prior probabilities) of each component

    def compute_guidance_gradient(self, x, t):
        """
        Compute the combined guidance gradient from the GMM classifier and the prior model.

        Parameters:
        - x: Current state estimate, shape (n_features,).
        - t: Current time step.

        Returns:
        - total_gradient: Combined gradient for guidance.
        """
        # --------------------------
        # Step 1: Compute classifier gradient
        # --------------------------
        # Predict the probability of each class given x
        p_c_given_x = self.gmm.predict_proba(x.reshape(1, -1))[0]
        # Determine the class with the highest probability
        c = np.argmax(p_c_given_x)
        # Compute the gradient of log p(c|x)
        grad_log_p_c = self.compute_gradient_log_p_c_given_x(x, c)

        # --------------------------
        # Step 2: Obtain variational likelihood term
        # --------------------------
        # Compute the gradient of the log-likelihood from the concentration prior model
        grad_log_likelihood = self.concentration_prior_model.compute_grad_log_likelihood(x)

        # --------------------------
        # Step 3: Dynamically adjust parameters
        # --------------------------
        # Calculate uncertainty as 1 minus the maximum class probability
        uncertainty = 1 - np.max(p_c_given_x)
        # Determine lambda value based on uncertainty
        lambda_val = 0.05 * (1 - uncertainty)

        # --------------------------
        # Step 4: Combine gradients for final guidance
        # --------------------------
        # Compute the total gradient as a weighted sum of classifier and prior gradients
        total_gradient = lambda_val * grad_log_p_c + grad_log_likelihood

        return total_gradient

    def compute_gradient_log_p_c_given_x(self, x, c):
        """
        Compute the gradient of the log posterior probability ∇x log p(c|x).

        Parameters:
        - x: Current state estimate, shape (n_features,).
        - c: Target class index.

        Returns:
        - grad_log_p_c_given_x: Gradient vector, shape (n_features,).
        """
        K = len(self.pi)  # Number of classes/components
        p_c_given_x = self._compute_p_c_given_x(x)

        # Compute the gradient for the target class c
        grad_log_p_c_given_x = -np.linalg.inv(self.Sigma[c]) @ (x - self.mu[c])
        # Adjust the gradient by subtracting the weighted sum of gradients for all classes
        for k in range(K):
            grad_log_p_c_given_x -= p_c_given_x[k] * (-np.linalg.inv(self.Sigma[k]) @ (x - self.mu[k]))

        return grad_log_p_c_given_x

    def _compute_p_c_given_x(self, x):
        """
        Compute the posterior probabilities p(c|x) for each class using the GMM.

        Parameters:
        - x: Current state estimate, shape (n_features,).

        Returns:
        - p_c_given_x: Array of posterior probabilities for each class, shape (K,).
        """
        K = len(self.pi)  # Number of classes/components
        p_c_given_x = np.zeros(K)
        # Calculate the probability for each class/component
        for c in range(K):
            p_c_given_x[c] = self.pi[c] * multivariate_normal.pdf(x, mean=self.mu[c], cov=self.Sigma[c])
        # Normalize the probabilities so they sum to 1
        p_c_given_x /= np.sum(p_c_given_x)
        return p_c_given_x

    
    
from sklearn.mixture import GaussianMixture
from pyts.decomposition import SingularSpectrumAnalysis

def ssa_clustering_basic(R_g_real, kalman_filter, ssa_window_size=10):
    """
    Perform Singular Spectrum Analysis (SSA) on the input signal, apply Gaussian Mixture Model (GMM) clustering,
    and refine the state estimates using gradient-based corrections before processing with a Kalman filter.

    Parameters:
    -----------
    R_g_real : np.array
        The real sensor response signal (e.g., from a MOX sensor).
    kalman_filter : object
        An instance of a Kalman filter (with an adaptive Unscented Kalman Filter, UKF) for subsequent processing.
    ssa_window_size : int, optional
        Window size for Singular Spectrum Analysis (SSA). Default is 10.

    Returns:
    --------
    x_est : np.array
        The state estimates after SSA, GMM-based corrections, and Kalman filtering.
    kf_estimates : np.array
        The UKF estimates computed from the corrected state estimates.
    """

    # Step 1: Decompose the signal using Singular Spectrum Analysis (SSA)
    # This extracts the key components of the input signal.
    ssa = SingularSpectrumAnalysis(window_size=ssa_window_size)
    components = ssa.fit_transform(R_g_real.reshape(1, -1))  # Reshape to (1, n_samples) for processing
    ssa_components = abs(components[0, 0:3, :].T)  # Use the first 3 components and take absolute values

    # Step 2: Train a Gaussian Mixture Model (GMM) on the SSA components
    # The GMM serves as a classifier to capture the underlying clusters in the data.
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(ssa_components)

    # Step 3: Extract GMM parameters
    # These parameters include means (mu), covariances (Sigma), and mixture weights (pi).
    mu = gmm.means_  # Means of the GMM components
    Sigma = gmm.covariances_  # Covariance matrices of the GMM components
    pi = [0.5, 0.5]  # Prior probabilities for each GMM component (assuming equal priors)

    # Step 4: Initialize the Classifier Guidance using GMM parameters
    classifier_guidance = ClassifierGuidance(mu, Sigma, pi)

    # Step 5: Set parameters for the gradient-based correction
    eta = 0.005  # Learning rate for gradient updates
    x_est = ssa_components.copy()  # Initial state estimates based on SSA components

    # Step 6: Iterate over each sample to refine state estimates
    for t in range(x_est.shape[0] - 1):
        x = x_est[t]

        # Predict the current state's class using GMM
        p_c_given_x = classifier_guidance._compute_p_c_given_x(x)
        c = np.argmax(p_c_given_x)  # Predicted class index

        # Adjust lambda based on uncertainty
        uncertainty = 1 - np.max(p_c_given_x)  # Measure of uncertainty in classification
        lambda_val = 0.05 * (1 - uncertainty)  # Dynamic lambda value inversely related to uncertainty

        # Compute the gradient of the log-probability for the predicted class
        grad_log_p_c_given_x = classifier_guidance.compute_gradient_log_p_c_given_x(x, c)

        # Update the state estimate using the computed gradient
        x_est[t] += eta * lambda_val * grad_log_p_c_given_x

        # Optional: Print progress every 1000 steps
        # if (t + 1) % 1000 == 0:
        #     print(f"Processed step {t + 1}")

    # Step 7: Process the corrected state estimates with the adaptive Unscented Kalman Filter (UKF)
    kf_estimates = kalman_filter.adaptive_ukf(y=x_est[:, 0])

    return x_est, kf_estimates



def ssa_clustering_guidance(R_g_real, kalman_filter, ssa_window_size=10, eta=0.0035):
    """
    Perform Singular Spectrum Analysis (SSA) on the input signal, apply Gaussian Mixture Model (GMM) clustering,
    and refine the state estimates using gradient-based corrections before processing with a Kalman filter.

    Parameters:
    -----------
    R_g_real : np.array
        The real sensor response signal (e.g., from a MOX sensor).
    kalman_filter : object
        An instance of a Kalman filter (with an adaptive Unscented Kalman Filter, UKF) for subsequent processing.
    ssa_window_size : int, optional
        Window size for Singular Spectrum Analysis (SSA). Default is 10.
    eta : float, optional
        Learning rate for gradient-based corrections. Default is 0.0035.

    Returns:
    --------
    x_est : np.array
        The state estimates after SSA, GMM-based corrections, and Kalman filtering.
    kf_estimates : np.array
        The UKF estimates computed from the corrected state estimates.
    """

    # Step 1: Decompose the signal using Singular Spectrum Analysis (SSA)
    # This extracts the key components of the input signal.
    ssa = SingularSpectrumAnalysis(window_size=ssa_window_size)
    components = ssa.fit_transform(R_g_real.reshape(1, -1))  # Reshape to (1, n_samples) for processing
    ssa_components = abs(components[0, 0:3, :].T)  # Use the first 3 components and take absolute values

    # Step 2: Train a Gaussian Mixture Model (GMM) on the SSA components
    # The GMM serves as a classifier to capture the underlying clusters in the data.
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(ssa_components)

    # Step 3: Initialize the Concentration Prior Model
    # This model incorporates prior knowledge about the concentration of the data points.
    concentration_prior_model = ConcentrationPriorModel(alpha=2, beta=1)

    # Step 4: Initialize the Enhanced Classifier Guidance using GMM and concentration prior
    classifier_guidance = EnhancedClassifierGuidance(gmm, concentration_prior_model)

    # Step 5: Set parameters for the gradient-based correction
    x_est = ssa_components.copy()  # Initial state estimates based on SSA components

    # Step 6: Iterate over each sample to refine state estimates
    for t in range(x_est.shape[0] - 1):
        x = x_est[t]

        # Compute the total guidance gradient using the classifier guidance
        total_gradient = classifier_guidance.compute_guidance_gradient(x, t)

        # Update the state estimate using the computed gradient and learning rate (eta)
        x_est[t] += eta * total_gradient

        # Optional: Compute the current weight from the concentration prior model
        # current_weight = classifier_guidance.concentration_prior_model.compute_weight(t)

        # Optional: Print progress every 1000 steps
        # if (t + 1) % 1000 == 0:
        #     print(f"Processed step {t + 1} with the current weight", current_weight)

    # Step 7: Process the corrected state estimates with the adaptive Unscented Kalman Filter (UKF)
    kf_estimates = kalman_filter.adaptive_ukf(y=x_est[:, 0])

    return x_est, kf_estimates



def ssa_clustering_guidence_multistep(R_g_real, kalman_filter, ssa_window_size=10, 
                                       num_inner_steps=50, eta=0.00035, momentum_coef=0.9):
    """
    Perform multi-step gradient correction with adaptive updates and momentum,
    guided by SSA and a GMM-based enhanced classifier guidance.
    
    Parameters:
    -----------
    R_g_real : np.array
        The real sensor response signal (e.g., from a MOX sensor).
    kalman_filter : object
        An instance of a Kalman filter (with an adaptive UKF) for subsequent processing.
    ssa_window_size : int, optional
        Window size for Singular Spectrum Analysis (SSA). Default is 10.
    num_inner_steps : int, optional
        Number of gradient update steps per sample. Default is 100.
    eta : float, optional
        Initial learning rate (step size) for gradient updates. Default is 0.00035.
    momentum_coef : float, optional
        Momentum coefficient used to smooth the update direction. Default is 0.9.
    
    Returns:
    --------
    x_est : np.array
        The state estimates after multi-step gradient correction.
    kf_estimates : np.array
        The UKF estimates computed from the corrected state estimates.
    """

    # Step 1: Decompose the signal using Singular Spectrum Analysis (SSA)
    # This extracts the key components of the input signal.
    ssa = SingularSpectrumAnalysis(window_size=ssa_window_size)
    components = ssa.fit_transform(R_g_real.reshape(1, -1))  # Shape: (1, n_samples)
    ssa_components = abs(components[0, 0:3, :].T)  # Use the first 3 components and take absolute values
    
    # Step 2: Train a Gaussian Mixture Model (GMM) on the SSA components
    # The GMM serves as a classifier to capture the underlying clusters in the data.
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(ssa_components)
    
    # Step 3: Initialize the concentration prior model (using a Gamma prior)
    concentration_prior_model = ConcentrationPriorModel(alpha=2, beta=1)
    
    # Step 4: Initialize the enhanced classifier guidance using GMM and the concentration prior model.
    classifier_guidance = EnhancedClassifierGuidance(gmm, concentration_prior_model)
    
    # Step 5: Set the initial state estimate from SSA components (this is the initial guess)
    x_est = ssa_components.copy()
    
    # Step 6: For each sample, apply multi-step gradient correction:
    for t in range(x_est.shape[0]-1):
        x = x_est[t]
        
        # Initialize momentum term for adaptive update.
        momentum = np.zeros_like(x)
        
        for i in range(num_inner_steps):
            # Compute the guidance gradient for the current state 'x'
            grad = classifier_guidance.compute_guidance_gradient(x, t)
            
            # Update momentum: incorporate previous momentum to smooth the gradient update.
            momentum = momentum_coef * momentum + (1 - momentum_coef) * grad
            
            # Compute adaptive learning rate using cosine decay
            # The learning rate decays from the initial eta to near zero as the iterations progress.
            adaptive_eta = eta * (np.cos(np.pi * i / num_inner_steps) + 1) / 2
            
            # Update the state 'x' using the adaptive learning rate and momentum-adjusted gradient.
            x = x + adaptive_eta * momentum
            
        # Store the updated state for the current sample.
        x_est[t] = x
    
    # Step 7: Process the corrected state estimates with the adaptive Unscented Kalman Filter (UKF)
    kf_estimates = kalman_filter.adaptive_ukf(y=x_est[:, 0])
    
    return x_est, kf_estimates
