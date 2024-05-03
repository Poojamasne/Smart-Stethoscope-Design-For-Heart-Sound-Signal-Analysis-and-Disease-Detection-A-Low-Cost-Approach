import librosa
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from sklearn.feature_selection import SelectKBest, f_regression


def extract_features(file_path):
    # Load the audio file using librosa
    y, sr = librosa.load(file_path)

    # Extract features
    # Example features - you can add more as needed
    features = {
        'mean': np.mean(y),
        'variance': np.var(y),
        'skewness': skew(y),
        'kurtosis': kurtosis(y),
        # Add more features as required
    }

    return features

# Example usage:
file_path = r'C:\Users\Pooja\OneDrive\Desktop\power spetrum\Normal heart sound\D1.wav'
extracted_features = extract_features(file_path)
print("Extracted features:", extracted_features)


# Example data
# X contains your feature matrix (features extracted from audio files), y contains your target labels.

# For demonstration purposes, consider X and y (replace with your actual data)
X = np.random.rand(100, 10)  # Example feature matrix
y = np.random.randint(0, 2, 100)  # Example target labels

# Feature selection
selector = SelectKBest(score_func=f_regression, k=5)  # Select top 5 features
X_selected = selector.fit_transform(X, y)

# Get the indices of selected features
selected_feature_indices = selector.get_support(indices=True)
print("Indices of selected features:", selected_feature_indices)


from sklearn.ensemble import RandomForestClassifier
import numpy as np


# Example data
# X contains your feature matrix (features extracted from audio files), y contains your target labels.

# For demonstration purposes, consider X and y (replace with your actual data)
X = np.random.rand(100, 10)  # Example feature matrix
y = np.random.randint(0, 2, 100)  # Example target labels

# Create a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X, y)

# Get feature importances
feature_importances = clf.feature_importances_

# Get indices of features sorted by importance
sorted_indices = np.argsort(feature_importances)[::-1]

# Print feature importances
print("Feature Importances:")
for i, idx in enumerate(sorted_indices):
    print(f"{i + 1}. Feature {idx} - Importance: {feature_importances[idx]}")


