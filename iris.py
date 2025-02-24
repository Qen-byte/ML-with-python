from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target (0, 1, 2 for species)

# Convert to DataFrame for easier exploration
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(y, iris.target_names)

# Peek at the data
print(df.head())
print("\nShape:", df.shape)


# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Split features and target
X = df.drop('species', axis=1)  # Features
y = df['species']  # Target

# Convert to NumPy arrays (scikit-learn expects this)
X = X.to_numpy()
y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)



# Initialize and train
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Accuracy on training data
train_accuracy = model.score(X_train, y_train)
print("Training accuracy:", train_accuracy)

# Test accuracy
test_accuracy = model.score(X_test, y_test)
print("Test accuracy:", test_accuracy)

# Predictions
y_pred = model.predict(X_test)
print("Sample predictions:", y_pred[:5])
print("Actual labels:", y_test[:5])



# Scatter plot (petal length vs. petal width)
plt.scatter(X_train[:, 2], X_train[:, 3], c=[iris.target_names.tolist().index(sp) for sp in y_train], label='Train')
plt.scatter(X_test[:, 2], X_test[:, 3], c=[iris.target_names.tolist().index(sp) for sp in y_test], marker='x', label='Test')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend()
plt.show()
