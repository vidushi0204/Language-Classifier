import pickle
import pandas as pd

# Load predictions from predictions.pkl
with open('predictions.pkl', 'rb') as f:
    predictions = pickle.load(f)

# Load class labels from public_test.csv
test_labels_df = pd.read_csv('public_test.csv')
true_labels = test_labels_df['class'].values  # Adjust the column name if it's different

# Compare predictions with true labels
comparison_df = pd.DataFrame({
    'True Labels': true_labels,
    'Predictions': predictions
})

# Print the comparison
print(comparison_df)

# Optional: Calculate accuracy
accuracy = (comparison_df['True Labels'] == comparison_df['Predictions']).mean()
print(f'Accuracy: {accuracy * 100:.2f}%')
