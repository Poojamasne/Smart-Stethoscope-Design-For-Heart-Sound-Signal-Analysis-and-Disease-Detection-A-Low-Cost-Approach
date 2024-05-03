import librosa
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# Feature extraction function
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    features = {
        'mean': np.mean(y),
        'variance': np.var(y),
        'skewness': skew(y),
        'kurtosis': kurtosis(y),
    }
    return features

# Feature extraction
file_path = r'C:\Users\Pooja\OneDrive\Desktop\power spetrum\Normal heart sound\D1.wav'
extracted_features = extract_features(file_path)
print("Extracted features:", extracted_features)

# Visualize extracted audio waveform
y, sr = librosa.load(file_path)
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title('Audio Waveform (Feature Extraction)')
plt.ylabel('Amplitude')
plt.show()

# Example data
X = np.random.rand(100, 10)  # Example feature matrix
y = np.random.randint(0, 2, 100)  # Example target labels

# Feature selection
selector = SelectKBest(score_func=f_regression, k=5)
X_selected = selector.fit_transform(X, y)
selected_feature_indices = selector.get_support(indices=True)
print("Indices of selected features:", selected_feature_indices)

# Create sample data for plotting (this is just for demonstration, replace with actual data)
selected_features_to_plot = X[0, selected_feature_indices]

# Visualize selected features
plt.figure(figsize=(6, 4))
plt.bar(range(len(selected_feature_indices)), selected_features_to_plot)
plt.title('Selected Features (Feature Selection)')
plt.xlabel('Feature Index')
plt.ylabel('Feature Value')
plt.show()

# Feature importance ranking with Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)
feature_importances = clf.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]
print("Feature Importances:")
for i, idx in enumerate(sorted_indices):
    print(f"{i + 1}. Feature {idx} - Importance: {feature_importances[idx]}")

# Visualize feature importances
plt.figure(figsize=(8, 6))
plt.bar(range(len(feature_importances)), feature_importances)
plt.title('Feature Importances (Feature Ranking)')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.show()
