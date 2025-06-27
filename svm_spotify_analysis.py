"""
SPOTIFY MUSIC CLASSIFICATION ANALYSIS USING SVM
===============================================

This script analyzes Spotify song data to classify songs into categories using Support Vector Machine (SVM).
We'll compare two approaches:
1. Direct SVM classification - using all original song features
2. PCA + SVM classification - first reducing dimensionality, then classifying

The goal is to predict song categories based on audio features and compare performance with/without PCA.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time
import warnings
warnings.filterwarnings('ignore')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import os

# Create folders for organized outputs
def create_output_directories():
    # Create main output directories
    directories = [
        'outputs',
        'outputs/figures',
        'outputs/figures/exploration',  # For initial data analysis
        'outputs/figures/classification',    # For classification results
        'outputs/figures/classification/direct',  # For direct classification
        'outputs/figures/classification/pca',     # For PCA classification
        'outputs/figures/comparison',    # For comparison results
        'outputs/data'                   # For CSV files
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

# Call this early in the script
create_output_directories()

# Set the style for better visualizations - with fallback for compatibility
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        # Fallback for older matplotlib versions
        plt.style.use('seaborn-whitegrid')
    except:
        print("Warning: seaborn-whitegrid style not found. Using default style.")
sns.set_palette("viridis")

# SECTION 1: LOADING AND EXPLORING THE DATA
print("="*80)
print("SECTION 1: LOADING AND EXPLORING THE DATA")
print("="*80)
print("""
PURPOSE: First, we need to understand what our data looks like and create target labels for classification.

We'll look at:
- How many songs we have
- What information we have about each song (tempo, energy, etc.)
- Create classification targets based on musical characteristics
- If any data is missing
- How the different musical features relate to each other
""")

# Load the dataset with path handling
# file is ./spotify_top_songs_audio_features.csv
file_path = './spotify_top_songs_audio_features.csv'
if not os.path.exists(file_path):
    # Fallback to relative path if absolute path not found
    file_path = './spotify_top_songs_audio_features.csv'
    if not os.path.exists(file_path):
        print(f"Error: Could not find dataset at {file_path}")
        print("Please update the file_path variable with the correct location.")
        import sys
        sys.exit(1)

df = pd.read_csv(file_path)

# Display basic information about the dataset
print("\nDataset Overview:")
print(f"Number of rows and columns: {df.shape}")
print("(Each row = one song, each column = one piece of information about the song)")

print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nThis shows us the first 5 songs and all their characteristics")

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())
print("Zero means no missing data for that feature - that's good!")

# Select only numerical features relevant for classification
numerical_features = ['danceability', 'energy', 'speechiness', 
                      'acousticness', 'valence']

print(f"\nWe'll analyze these {len(numerical_features)} musical characteristics:")
print("- danceability: How suitable for dancing (0=not danceable, 1=very danceable)")
print("- energy: How intense/powerful the song feels (0=calm, 1=energetic)")
print("- speechiness: How much spoken words vs singing (0=music, 1=speech)")
print("- acousticness: How acoustic vs electronic (0=electronic, 1=acoustic)")
print("- valence: How positive/happy the song sounds (0=sad, 1=happy)")

# Create a dataframe with only numerical features
df_numerical = df[numerical_features].copy()

# Create target labels for classification based on musical characteristics
def create_music_categories(df):
    """
    Create music categories based on audio features for classification
    """
    categories = []
    
    for _, row in df.iterrows():
        # Define thresholds for categorization
        high_energy = row['energy'] > 0.7
        high_dance = row['danceability'] > 0.7
        high_valence = row['valence'] > 0.6
        high_acoustic = row['acousticness'] > 0.5
        high_speech = row['speechiness'] > 0.33
        
        # Create categories based on combinations of features
        if high_energy and high_dance and high_valence:
            categories.append('Energetic_Dance')
        elif high_acoustic and not high_energy:
            categories.append('Acoustic_Calm')
        elif high_speech:
            categories.append('Speech_Heavy')
        elif high_valence and not high_energy:
            categories.append('Happy_Mellow')
        elif not high_valence and not high_energy:
            categories.append('Sad_Calm')
        else:
            categories.append('Moderate')
    
    return categories

# Create target variable for classification
print("\nCreating music categories for classification...")
df['music_category'] = create_music_categories(df)

# Display category distribution
category_counts = df['music_category'].value_counts()
print("\nMusic Category Distribution:")
print(category_counts)
print(f"\nTotal categories: {len(category_counts)}")

# Visualize category distribution
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
category_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Music Categories')
plt.xlabel('Category')
plt.ylabel('Number of Songs')
plt.xticks(rotation=45)

plt.subplot(2, 1, 2)
plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
plt.title('Music Categories - Percentage Distribution')
plt.tight_layout()
plt.savefig('outputs/figures/exploration/category_distribution.png')
plt.close()

print("""
MUSIC CATEGORIES EXPLANATION:
- Energetic_Dance: High energy, danceable, positive songs
- Acoustic_Calm: Acoustic, low energy songs
- Speech_Heavy: Songs with significant spoken content
- Happy_Mellow: Positive but not energetic songs
- Sad_Calm: Low valence, low energy songs
- Moderate: Songs that don't fit other clear categories
""")

# Display the correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = df_numerical.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Audio Features', fontsize=15)
plt.tight_layout()
plt.savefig('outputs/figures/exploration/correlation_matrix.png')
plt.close()

# Create histograms for each numerical feature
plt.figure(figsize=(15, 12))
for i, feature in enumerate(numerical_features):
    plt.subplot(4, 3, i+1)
    sns.histplot(df_numerical[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.tight_layout()
plt.savefig('outputs/figures/exploration/feature_distributions.png')
plt.close()

# Create boxplots to identify outliers
plt.figure(figsize=(15, 12))
for i, feature in enumerate(numerical_features):
    plt.subplot(4, 3, i+1)
    sns.boxplot(y=df_numerical[feature])
    plt.title(f'Boxplot of {feature}')
    plt.tight_layout()
plt.savefig('outputs/figures/exploration/feature_boxplots.png')
plt.close()

# SECTION 2: PREPROCESSING AND NORMALIZATION
print("\n" + "="*80)
print("SECTION 2: PREPROCESSING AND NORMALIZATION")
print("="*80)
print("""
PURPOSE: Prepare the data for SVM classification.

PROBLEM: Different features have different scales and SVM is sensitive to feature scaling.
SOLUTION: Normalize everything to the same scale and encode target labels.
""")

# Check for and handle any missing values
print("\nBefore handling missing values:")
print(df_numerical.isnull().sum())

# Fill missing values with the mean of the respective column
df_numerical = df_numerical.fillna(df_numerical.mean())

print("\nAfter handling missing values:")
print(df_numerical.isnull().sum())
print("All zeros means no missing data - ready for analysis!")

# Apply standard scaling to normalize the data
print("\nApplying standardization (normalizing the data)...")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numerical)
df_scaled = pd.DataFrame(scaled_data, columns=df_numerical.columns)

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['music_category'])
print(f"\nTarget labels encoded: {dict(enumerate(label_encoder.classes_))}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    scaled_data, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nData split:")
print(f"Training set: {X_train.shape[0]} songs")
print(f"Testing set: {X_test.shape[0]} songs")
print(f"Features: {X_train.shape[1]}")

# SECTION 3: SVM WITHOUT PCA
print("\n" + "="*80)
print("SECTION 3: SVM CLASSIFICATION WITHOUT PCA")
print("="*80)
print("""
PURPOSE: Train SVM classifier using all original features.

We'll:
1. Find the best SVM parameters using grid search
2. Train the model on training data
3. Evaluate performance on test data
""")

# Grid search for optimal SVM parameters
print("Finding optimal SVM parameters...")

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}

direct_start_time = time.time()

# Grid search with cross-validation
svm_grid = GridSearchCV(
    SVC(random_state=42), 
    param_grid, 
    cv=3, 
    scoring='accuracy',
    n_jobs=-1
)

svm_grid.fit(X_train, y_train)
direct_end_time = time.time()
direct_execution_time = direct_end_time - direct_start_time

print(f"Best parameters for direct SVM: {svm_grid.best_params_}")
print(f"Best cross-validation score: {svm_grid.best_score_:.4f}")
print(f"Training time: {direct_execution_time:.2f} seconds")

# Train final model with best parameters
best_svm_direct = svm_grid.best_estimator_

# Make predictions
y_pred_direct = best_svm_direct.predict(X_test)

# Calculate metrics
accuracy_direct = accuracy_score(y_test, y_pred_direct)
precision_direct = precision_score(y_test, y_pred_direct, average='weighted')
recall_direct = recall_score(y_test, y_pred_direct, average='weighted')
f1_direct = f1_score(y_test, y_pred_direct, average='weighted')

print(f"\nDirect SVM Performance:")
print(f"Accuracy: {accuracy_direct:.4f}")
print(f"Precision: {precision_direct:.4f}")
print(f"Recall: {recall_direct:.4f}")
print(f"F1-Score: {f1_direct:.4f}")

# Detailed classification report
print("\nDetailed Classification Report (Direct SVM):")
print(classification_report(y_test, y_pred_direct, target_names=label_encoder.classes_))

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm_direct = confusion_matrix(y_test, y_pred_direct)
sns.heatmap(cm_direct, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix - Direct SVM')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('outputs/figures/classification/direct/confusion_matrix.png')
plt.close()

print("""
CONFUSION MATRIX INTERPRETATION:
- Rows = actual categories
- Columns = predicted categories
- Diagonal values = correct predictions
- Off-diagonal = misclassifications
- Higher diagonal values = better performance
""")

# SECTION 4: SVM WITH PCA
print("\n" + "="*80)
print("SECTION 4: SVM CLASSIFICATION WITH PCA")
print("="*80)
print("""
PURPOSE: Apply PCA to reduce dimensionality, then train SVM classifier.

PCA BENEFITS FOR SVM:
1. Reduces computational complexity
2. Removes noise and redundant features
3. May improve generalization
4. Faster training and prediction
""")

# Apply PCA to find optimal number of components
pca_full = PCA()
pca_data_full = pca_full.fit_transform(X_train)

# Analyze explained variance
explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Plot explained variance
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Individual Explained Variance')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
plt.axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/figures/classification/pca/explained_variance.png')
plt.close()

# Determine optimal number of components (80% variance)
n_components = np.argmax(cumulative_variance >= 0.8) + 1
print(f"\nNumber of components for 80% variance: {n_components}")
print(f"Explained variance with {n_components} components: {cumulative_variance[n_components-1]:.4f}")

# Apply PCA with selected components
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"\nDimensionality reduction:")
print(f"Original features: {X_train.shape[1]}")
print(f"PCA features: {X_train_pca.shape[1]}")
print(f"Reduction: {((X_train.shape[1] - X_train_pca.shape[1]) / X_train.shape[1] * 100):.1f}%")

# Grid search for SVM with PCA data
print("\nFinding optimal SVM parameters for PCA data...")
pca_start_time = time.time()

svm_grid_pca = GridSearchCV(
    SVC(random_state=42), 
    param_grid, 
    cv=3, 
    scoring='accuracy',
    n_jobs=-1
)

svm_grid_pca.fit(X_train_pca, y_train)
pca_end_time = time.time()
pca_execution_time = pca_end_time - pca_start_time

print(f"Best parameters for PCA SVM: {svm_grid_pca.best_params_}")
print(f"Best cross-validation score: {svm_grid_pca.best_score_:.4f}")
print(f"Training time: {pca_execution_time:.2f} seconds")

# Train final model with PCA data
best_svm_pca = svm_grid_pca.best_estimator_

# Make predictions with PCA
y_pred_pca = best_svm_pca.predict(X_test_pca)

# Calculate metrics for PCA approach
accuracy_pca = accuracy_score(y_test, y_pred_pca)
precision_pca = precision_score(y_test, y_pred_pca, average='weighted')
recall_pca = recall_score(y_test, y_pred_pca, average='weighted')
f1_pca = f1_score(y_test, y_pred_pca, average='weighted')

print(f"\nPCA SVM Performance:")
print(f"Accuracy: {accuracy_pca:.4f}")
print(f"Precision: {precision_pca:.4f}")
print(f"Recall: {recall_pca:.4f}")
print(f"F1-Score: {f1_pca:.4f}")

# Detailed classification report for PCA
print("\nDetailed Classification Report (PCA SVM):")
print(classification_report(y_test, y_pred_pca, target_names=label_encoder.classes_))

# Confusion Matrix for PCA
plt.figure(figsize=(10, 8))
cm_pca = confusion_matrix(y_test, y_pred_pca)
sns.heatmap(cm_pca, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix - PCA SVM')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('outputs/figures/classification/pca/confusion_matrix.png')
plt.close()

# Visualize PCA components
if X_train_pca.shape[1] >= 2:
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, 
                         cmap='viridis', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Category')
    plt.title('Training Data in PCA Space (First 2 Components)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.tight_layout()
    plt.savefig('outputs/figures/classification/pca/pca_2d_visualization.png')
    plt.close()

# SECTION 5: COMPARISON AND CONCLUSION
print("\n" + "="*80)
print("SECTION 5: COMPARISON AND CONCLUSION")
print("="*80)
print("""
PURPOSE: Compare SVM performance with and without PCA and draw conclusions.
""")

# Create performance comparison
metrics_comparison = pd.DataFrame({
    'Direct SVM': [accuracy_direct, precision_direct, recall_direct, f1_direct, direct_execution_time],
    'PCA SVM': [accuracy_pca, precision_pca, recall_pca, f1_pca, pca_execution_time]
}, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Training Time (s)'])

print("\nPerformance Comparison:")
print(metrics_comparison)

# Visualize performance comparison
plt.figure(figsize=(15, 10))

# Performance metrics comparison
plt.subplot(2, 2, 1)
metrics_comparison.iloc[:-1].plot(kind='bar', ax=plt.gca())
plt.title('Classification Performance Comparison')
plt.ylabel('Score')
plt.legend()
plt.xticks(rotation=45)

# Training time comparison
plt.subplot(2, 2, 2)
training_times = [direct_execution_time, pca_execution_time]
methods = ['Direct SVM', 'PCA SVM']
plt.bar(methods, training_times, color=['#1DB954', '#191414'])
plt.title('Training Time Comparison')
plt.ylabel('Time (seconds)')
for i, v in enumerate(training_times):
    plt.text(i, v + 0.1, f"{v:.2f}s", ha='center')

# Feature importance (for linear kernel if available)
plt.subplot(2, 2, 3)
if best_svm_direct.kernel == 'linear':
    feature_importance = np.abs(best_svm_direct.coef_[0])
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.title('Feature Importance (Direct SVM)')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
else:
    plt.text(0.5, 0.5, 'Feature importance\navailable only\nfor linear kernel', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Feature Importance (Direct SVM)')

# Dimensionality reduction benefit
plt.subplot(2, 2, 4)
dimension_comparison = [X_train.shape[1], X_train_pca.shape[1]]
methods = ['Original', 'PCA']
plt.bar(methods, dimension_comparison, color=['red', 'blue'])
plt.title('Dimensionality Comparison')
plt.ylabel('Number of Features')
for i, v in enumerate(dimension_comparison):
    plt.text(i, v + 0.1, f"{v}", ha='center')

plt.tight_layout()
plt.savefig('outputs/figures/comparison/svm_performance_comparison.png')
plt.close()

# Save results
results_df = pd.DataFrame({
    'actual_category': [label_encoder.classes_[i] for i in y_test],
    'direct_svm_prediction': [label_encoder.classes_[i] for i in y_pred_direct],
    'pca_svm_prediction': [label_encoder.classes_[i] for i in y_pred_pca],
    'direct_correct': y_test == y_pred_direct,
    'pca_correct': y_test == y_pred_pca
})

results_df.to_csv('outputs/data/svm_classification_results.csv', index=False)

# Final conclusions
print(f"\nFINAL CONCLUSIONS:")
print(f"1. Direct SVM Accuracy: {accuracy_direct:.4f}")
print(f"2. PCA SVM Accuracy: {accuracy_pca:.4f}")
print(f"3. Training time reduction with PCA: {((direct_execution_time - pca_execution_time) / direct_execution_time * 100):.1f}%")
print(f"4. Dimensionality reduction: {((X_train.shape[1] - X_train_pca.shape[1]) / X_train.shape[1] * 100):.1f}%")

if accuracy_pca > accuracy_direct:
    print(f"5. PCA improved accuracy by {((accuracy_pca - accuracy_direct) / accuracy_direct * 100):.1f}%")
else:
    print(f"5. Direct SVM performed better by {((accuracy_direct - accuracy_pca) / accuracy_pca * 100):.1f}%")

print(f"\nRecommendation: {'Use PCA SVM' if accuracy_pca >= accuracy_direct else 'Use Direct SVM'} for this dataset.")

print("\nAnalysis complete! All results saved to outputs directory.")


