#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:58:00 2024

@author: han
"""

import numpy as np
import pandas as pd
import os

class SensorDataProcessor:
    """
    A class for processing sensor data from the Low-Speed Wind Tunnel (LST) dataset in Marknesse, Netherlands.
    This class is designed to handle sensor data for regression and target preparation, as well as
    train-test splitting and sequence preparation.

    Attributes:
    -----------
    base_path : str
        The base directory where sensor data files are located.
    """

    def __init__(self, base_path):
        """
        Initializes the SensorDataProcessor with the given base directory path.

        Parameters:
        -----------
        base_path : str
            The directory path containing the sensor data files.
        """
        self.base_path = base_path

    def extract_layer_data(self, sensor_type='PID-sensor', sensor_layer_to_plot=0):
        """
        Extracts sensor data from a specific layer and sensor type.

        Parameters:
        -----------
        sensor_type : str
            Type of sensor ('PID-sensor' or 'MiCS5524').
        sensor_layer_to_plot : int
            Layer index to extract data from.

        Returns:
        --------
        DataFrame
            Extracted data with specified columns for the given sensor layer.
        """
        file_path = f"{self.base_path}/sensor-layer_{sensor_layer_to_plot}.csv"
        df_layer = pd.read_csv(file_path, delimiter=',')

        # Specify columns to extract, including sensor data and positional/wind measurements.
        columns_to_extract = [sensor_type, 'x_reported', 'y_reported', 'z_reported',
                              'wind-u', 'wind-v', 'wind-w']
        return df_layer[columns_to_extract]

    def prepare_regression_data(self, sensor_type, num_layers):
        """
        Prepares regression data by extracting and concatenating sensor data across all layers.

        Parameters:
        -----------
        sensor_type : str
            Type of sensor ('PID-sensor' or 'MiCS5524').
        num_layers : int
            Number of layers to process.

        Returns:
        --------
        DataFrame
            Regression data with NaN values removed and values normalized.
        """
        data_list = []
        for i in range(num_layers):
            df = self.extract_layer_data(sensor_type=sensor_type, sensor_layer_to_plot=i)
            data_list.append(df[sensor_type])

        # Combine data from all layers and clean up
        data4reg = pd.concat(data_list, axis=1).dropna()
        data4reg[data4reg > 2] = 1
        data4reg[data4reg > 1] = 0.7
        return data4reg

    def prepare_target_data(self, sensor_type, num_layers):
        """
        Prepares target data by extracting and concatenating sensor data across all layers.

        Parameters:
        -----------
        sensor_type : str
            Type of sensor ('PID-sensor' or 'MiCS5524').
        num_layers : int
            Number of layers to process.

        Returns:
        --------
        DataFrame
            Target data with NaN values removed.
        """
        data_list = []
        for i in range(num_layers):
            df = self.extract_layer_data(sensor_type=sensor_type, sensor_layer_to_plot=i)
            data_list.append(df[sensor_type])

        # Combine data from all layers and clean up
        data4target = pd.concat(data_list, axis=1).dropna()
        return data4target

    def train_test_split(self, data4reg, data4target, train_portion):
        """
        Splits data into training and testing sets.

        Parameters:
        -----------
        data4reg : DataFrame
            The regression data.
        data4target : DataFrame
            The target data.
        train_portion : float
            Proportion of data to use for training.

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Train and test splits for regression (X) and target (Y) data.
        """
        time_len = data4reg.shape[0]
        train_size = int(time_len * train_portion)

        train_data_X = np.array(data4reg.iloc[10000:train_size,:])
        train_data_Y = np.array(data4target.iloc[10000:train_size,:])
        test_data_X = np.array(data4reg.iloc[train_size:,:])
        test_data_Y = np.array(data4target.iloc[train_size:,:])

        return train_data_X, train_data_Y, test_data_X, test_data_Y

    def sequence_data_preparation(self, seq_len, train_data_X, train_data_Y, test_data_X, test_data_Y):
        """
        Prepares sequential data for training and testing.

        Parameters:
        -----------
        seq_len : int
            Length of the sequences.
        train_data_X : np.ndarray
            Training regression data.
        train_data_Y : np.ndarray
            Training target data.
        test_data_X : np.ndarray
            Testing regression data.
        test_data_Y : np.ndarray
            Testing target data.

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Sequences and targets for both training and testing.
        """
        trainX, trainY, testX, testY = [], [], [], []

        for i in range(train_data_X.shape[1] - int(seq_len - 1)):
            a = train_data_X[:, i : i + seq_len ]
            c = train_data_Y[:, (i+ 5) : i + seq_len ]
            trainX.append(a[:, :seq_len]) 
            trainY.append(0.*c.mean(axis=1)+1*c[:,-1]  )
       

        for i in range(test_data_X.shape[1] - int(seq_len  - 1)):
            b = test_data_X[:, i : i + seq_len ]
            d = test_data_Y[:, (i+ 5) : i + seq_len ]
            testX.append(b[:, :seq_len])      
            testY.append(0.*d.mean(axis=1)+1*d[:,-1])

        return np.array(trainX), np.array(trainY), np.array(testX), np.array(testY)

    def prepare_training_data(self, data4reg, data4target, train_portion=0.8, seq_len=80, target_layer=0):
        """
        Integrates data preparation steps for training and testing.

        Parameters:
        -----------
        data4reg : DataFrame
            The regression data.
        data4target : DataFrame
            The target data.
        train_portion : float
            Proportion of data to use for training.
        seq_len : int
            Length of sequences for model input.
        target_layer : int
            Specific layer to use for regression data.

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Prepared training and testing data (X, Y) ready for modeling.
        """
        train_data_X, train_data_Y, test_data_X, test_data_Y = self.train_test_split(
            data4reg, data4target, train_portion
        )
     
        trainX, trainY, testX, testY = self.sequence_data_preparation( seq_len,
                                                           train_data_X.T, 
                                                           train_data_Y.T, 
                                                           test_data_X.T, 
                                                           test_data_Y.T)
    
        # Reshape and extract the target layer
        trainX = np.copy(trainX.reshape(-1, seq_len, 4))
        testX = np.copy(testX.reshape(-1, seq_len, 4))
        trainY = np.copy(trainY.reshape(-1, 4))
        testY = np.copy(testY.reshape(-1, 4))

        X_train = np.copy(trainX[:, :, target_layer])
        X_test = np.copy(testX[:, :, target_layer])
        y_train = np.copy(trainY[:, target_layer])
        y_test = np.copy(testY[:, target_layer])

        return X_train, X_test, y_train, y_test
    

from pathlib import Path

class TestDataProcessor:
    def __init__(self, base_path):
        self.base_path = Path(base_path).resolve() #/ "train/test"
        print(f"Initialized data path: {self.base_path}")
        
        # Verify the critical path exists
        if not (self.base_path / "batch-1").exists():
            raise FileNotFoundError(f"Could not find batch directories in: {self.base_path}")

    def load_test_data_from_batches(self, sensor_type='PID-sensor', num_layers=4):
        batch_dirs = sorted(self.base_path.glob("batch-*"))
        all_data = []
    
        print(f"\n{'='*40}")
        print(f"Loading data with temporal separation")
        print(f"Total batches: {len(batch_dirs)}")
        print(f"Processing {num_layers} layers per batch")
        print(f"{'='*40}")
    
        for batch_dir in batch_dirs:
            batch_name = batch_dir.name
            print(f"\nProcessing batch: {batch_name}")
            
            for layer in range(num_layers):
                layer_files = list(batch_dir.glob(f"*sensor_layer_{layer}.csv"))
                if not layer_files:
                    print(f"  Layer {layer}: ❌ No files found")
                    continue
    
                file_path = layer_files[0]
                print(f"  Layer {layer}: ✅ Found {file_path.name}")
    
                try:
                    # Load and process individual file
                    df = pd.read_csv(file_path)
                    df['index'] = df['index'].str.replace('t', '').astype(int)
                    df = df.sort_values(by='index')
                    df['time_seconds'] = df['index'] * 0.1
                    df = df[['time_seconds', sensor_type]]
                    
                    # Add metadata columns
                    df['batch'] = batch_name
                    df['layer'] = layer
                    df['experiment_id'] = f"{batch_name}_layer{layer}"
                    
                    all_data.append(df)
                    print(f"    ✔️ Added {len(df)} time points "
                          f"({df['time_seconds'].min():.1f}s - {df['time_seconds'].max():.1f}s)")
    
                except Exception as e:
                    print(f"    ❌ Error processing: {str(e)}")
                    continue
    
        if all_data:
            final_df = pd.concat(all_data, axis=0).reset_index(drop=True)
            print(f"\n{'='*40}")
            print("Data organization:")
            print(f"Total files loaded: {len(all_data)}")
            print(f"Unique batches: {final_df['batch'].nunique()}")
            print(f"Unique experiments: {final_df['experiment_id'].nunique()}")
            print(f"Final dataframe shape: {final_df.shape}")
            print(f"Columns: {list(final_df.columns)}")
            return final_df
        else:
            raise FileNotFoundError("No valid data loaded")

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # =================================================================
    # Example Usage - Modify These Values for Your System
    # =================================================================
    DATA_ROOT = "train/test"  
    SENSOR_TYPE = 'PID-sensor'                   # Sensor type to analyze
    NUM_LAYERS = 4                               # Number of sensor layers
    
    try:
        # Initialize processor
        print("Initializing data processor...")
        processor = TestDataProcessor(DATA_ROOT)
        
        # Load data with temporal separation
        print("\nLoading data...")
        df = processor.load_test_data_from_batches(
            sensor_type=SENSOR_TYPE,
            num_layers=NUM_LAYERS
        )
        
        # =============================================================
        # Basic Data Inspection
        # =============================================================
        print("\nData Overview:")
        print(f"Total entries: {len(df):,}")
        print(f"Time range: {df['time_seconds'].min():.1f}s to {df['time_seconds'].max():.1f}s")
        print(f"Unique batches: {df['batch'].nunique()}")
        print(f"Unique experiments: {df['experiment_id'].nunique()}")
        
        print("\nFirst 5 entries:")
        print(df.head())
        
        print("\nDataframe Info:")
        print(df.info())

        # =============================================================
        # Example Analysis 1: Basic Statistics
        # =============================================================
        print("\nStatistical Summary:")
        print(df[SENSOR_TYPE].describe())
        
        # =============================================================
        # Example Analysis 2: Batch Comparison
        # =============================================================
        batch_stats = df.groupby('batch')[SENSOR_TYPE].agg(['mean', 'std'])
        print("\nBatch-wise Statistics:")
        print(batch_stats)

        # =============================================================
        # Example Visualization 1: Single Experiment Timeline
        # =============================================================
        sample_exp = df['experiment_id'].iloc[0]  # Get first experiment
        exp_data = df[df['experiment_id'] == sample_exp]
        
        plt.figure(figsize=(10, 5))
        plt.plot(exp_data['time_seconds'], exp_data[SENSOR_TYPE])
        plt.title(f"Sensor Data for {sample_exp}")
        plt.xlabel("Time (seconds)")
        plt.ylabel(SENSOR_TYPE)
        plt.grid(True)
        plt.show()

        # =============================================================
        # Example Visualization 2: Layer Comparison
        # =============================================================
        layer_means = df.groupby('layer')[SENSOR_TYPE].mean()
        
        plt.figure(figsize=(8, 4))
        layer_means.plot(kind='bar')
        plt.title("Average Sensor Readings by Layer")
        plt.xlabel("Layer Number")
        plt.ylabel(f"Mean {SENSOR_TYPE}")
        plt.xticks(rotation=0)
        plt.show()

    except FileNotFoundError as e:
        print(f"\nERROR: {str(e)}")
        print(f"Check if path exists: {Path(DATA_ROOT).resolve()}")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")     
        


# # Import the SensorDataProcessor class from a custom module
# from LST_SensorDataProcessor import SensorDataProcessor

# # Example usage:
# processor = SensorDataProcessor(base_path="train/experiment")
# # Prepare regression and target data for the sensor
# data4reg = processor.prepare_regression_data(sensor_type='MiCS5524', num_layers=4)
# data4target = processor.prepare_target_data(sensor_type='PID-sensor', num_layers=4)

# # Split the data into training and testing sets
# train_data_X, train_data_Y, test_data_X, test_data_Y = processor.train_test_split(
#     data4reg, data4target, train_portion=0.8
# )

# seq_len = 80  # Sequence length for training data

# # Prepare training and testing datasets
# X_train, X_test, y_train, y_test = processor.prepare_training_data(
#     data4reg, data4target, train_portion=0.8, seq_len=seq_len, target_layer=0
# )

# # Now, X_train, X_test, y_train, and y_test are ready for model training and testing
# print("Training data shape:", X_train.shape, y_train.shape)
# print("Testing data shape:", X_test.shape, y_test.shape)

# # Replace these with your actual data arrays
# C_PID_real = np.hstack((train_data_Y[:, 0], test_data_Y[:, 0]))  # Real PID readings
# R_g_real = np.hstack((train_data_X[:, 0], test_data_X[:, 0]))    # Real MOX sensor responses
