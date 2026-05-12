#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:16:41 2025

@author: lucaromeo
"""

import os
import warnings
import seaborn as sns
# Suppress warnings
warnings.filterwarnings('ignore')

# Essential libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from catboost import CatBoostRegressor
import torch
import pandas as pd
import numpy as np
import scipy.stats as stats
# Set random seed for reproducibility

seed = 69
torch.manual_seed(seed)
np.random.seed(seed)

# Define root directory
root = '.'

# Load dataset
(samples, siss, ohss, okss, participants) = torch.load(os.path.join(root, 'dataset/dataset-' + 'daily' + '.pt'))


# Convert tensors to pandas DataFrame
df = pd.DataFrame({
    "SISS": siss,
    "OHSS": ohss,
    "OKSS": okss
})

# =============================================================================
# # Compute correlation matrices
# correlation_matrix_pearson = df.corr(method='pearson')  # Linear relationships
# correlation_matrix_spearman = df.corr(method='spearman')  # Rank-based relationships
# correlation_matrix_kendall = df.corr(method='kendall')  # Ordinal relationships
# 
# # Function to plot correlation matrix
# def save_correlation_matrix(corr_matrix, title, filename):
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
#     plt.title(title)
#     # Save the figure as pdf
#     plt.savefig(filename, bbox_inches='tight')
#     plt.close()
# 
# # Save heatmaps as images
# save_correlation_matrix(correlation_matrix_pearson, "Pearson Correlation Matrix", "results/pearson_correlation.pdf")
# save_correlation_matrix(correlation_matrix_spearman, "Spearman Correlation Matrix", "results/spearman_correlation.pdf")
# save_correlation_matrix(correlation_matrix_kendall, "Kendall Correlation Matrix", "results/kendall_correlation.pdf")
# 
# # Print results
# print("\nPearson Correlation Matrix:\n", correlation_matrix_pearson)
# print("\nSpearman Correlation Matrix:\n", correlation_matrix_spearman)
# print("\nKendall Correlation Matrix:\n", correlation_matrix_kendall)
# 
# =============================================================================


# =============================================================================
# # Compute quartiles for SISS, OHSS, and OKSS
# siss_q1, siss_q2, siss_q3 = np.percentile(df["SISS"], [25, 50, 75])
# ohss_q1, ohss_q2, ohss_q3 = np.percentile(df["OHSS"], [25, 50, 75])
# okss_q1, okss_q2, okss_q3 = np.percentile(df["OKSS"], [25, 50, 75])
# 
# # Define quartile-based bins and labels
# quartile_labels = ["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"]
# 
# # Discretization for SISS
# siss_bins = [df["SISS"].min(), siss_q1, siss_q2, siss_q3, df["SISS"].max()]
# df["SISS_Category_Q"] = pd.cut(df["SISS"], bins=siss_bins, labels=quartile_labels, include_lowest=True)
# 
# # Discretization for OHSS
# ohss_bins = [df["OHSS"].min(), ohss_q1, ohss_q2, ohss_q3, df["OHSS"].max()]
# df["OHSS_Category_Q"] = pd.cut(df["OHSS"], bins=ohss_bins, labels=quartile_labels, include_lowest=True)
# 
# # Discretization for OKSS
# okss_bins = [df["OKSS"].min(), okss_q1, okss_q2, okss_q3, df["OKSS"].max()]
# df["OKSS_Category_Q"] = pd.cut(df["OKSS"], bins=okss_bins, labels=quartile_labels, include_lowest=True)
# 
# # Compute frequency distributions
# siss_freq_q = df["SISS_Category_Q"].value_counts().sort_index()
# ohss_freq_q = df["OHSS_Category_Q"].value_counts().sort_index()
# okss_freq_q = df["OKSS_Category_Q"].value_counts().sort_index()
# 
# # Print frequency distributions
# print("\nSISS Quartile Frequency Distribution:\n", siss_freq_q)
# print("\nOHSS Quartile Frequency Distribution:\n", ohss_freq_q)
# print("\nOKSS Quartile Frequency Distribution:\n", okss_freq_q)
# 
# # Plot frequency distributions
# def save_quartile_distribution(freq_data, title, filename):
#     plt.figure(figsize=(8, 5))
#     sns.barplot(x=freq_data.index, y=freq_data.values, palette="coolwarm")
#     plt.xlabel("Quartile")
#     plt.ylabel("Frequency")
#     plt.title(title)
#     plt.xticks(rotation=45)
#     plt.savefig(filename, bbox_inches='tight')
#     plt.close()
#      
# 
# # Generate plots
# save_quartile_distribution(siss_freq_q, "Frequency Distribution of SISS Quartiles", "results/quartile_siss.pdf")
# save_quartile_distribution(ohss_freq_q, "Frequency Distribution of OHSS Quartiles", "results/quartile_ohss.pdf")
# save_quartile_distribution(okss_freq_q, "Frequency Distribution of OKSS Quartiles", "results/quartile_okss.pdf")
# =============================================================================



# Function to create and save boxplots for each discretized variable
def save_boxplot(data, feature, title, filename):
    plt.figure(figsize=(8, 5))
    sns.boxplot(y=data[feature], palette="coolwarm")
    plt.ylabel(feature)
    plt.title(title)
    plt.xticks(rotation=45)
    
    # Save the figure as PNG
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Save boxplots for each discretized variable
save_boxplot(df, "SISS", "SISS box plot", "results/siss_boxplot.pdf")
save_boxplot(df, "OHSS",  "OHSS box plot", "results/ohss_boxplot.pdf")
save_boxplot(df, "OKSS", "OKSS box plot", "results/okss_boxplot.pdf")

