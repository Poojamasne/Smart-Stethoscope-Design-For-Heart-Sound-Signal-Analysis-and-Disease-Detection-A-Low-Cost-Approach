#Feature Extraction Visualization:
# Assuming 'y' is the audio signal loaded by librosa
# Display the waveform
import matplotlib.pyplot as plt  # Importing Matplotlib

# Your code that uses Matplotlib here
plt.figure(figsize=(10, 4))  # Example usage of plt.figure()
# Additional plotting code

plt.show()  # Display the plot (if any)

librosa.display.waveshow(y, sr=sr)
plt.title('Waveform')
plt.show()

# Display the spectrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
plt.figure(figsize=(10, 4))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()


#Feature Selection Visualization:
# Assuming X contains your feature matrix and y contains your target labels
selector = SelectKBest(score_func=f_regression, k=5)
X_selected = selector.fit_transform(X, y)

# Visualize selected features
selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = [f'Feature {i}' for i in selected_feature_indices]

plt.bar(range(len(selected_feature_indices)), selector.scores_[selected_feature_indices])
plt.xticks(range(len(selected_feature_indices)), selected_feature_names, rotation='vertical')
plt.xlabel('Selected Features')
plt.ylabel('Score')
plt.title('Selected Features')
plt.show()

#Feature Ranking
# Assuming X contains your feature matrix and y contains your target labels
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Get feature importances
feature_importances = clf.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

plt.bar(range(X.shape[1]), feature_importances[sorted_indices], align='center')
plt.xticks(range(X.shape[1]), sorted_indices)
plt.xlabel('Feature Index')
plt.ylabel('Feature Importance')
plt.title('Feature Importances from Random Forest')
plt.show()

