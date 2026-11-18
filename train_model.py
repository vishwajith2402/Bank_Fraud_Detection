import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

print("Loading dataset...")

df = pd.read_csv("creditcard.csv")

print("Dataset loaded")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Balancing dataset...")

smote = SMOTE()

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train,
    y_train
)

print("Training model...")

model = RandomForestClassifier(n_estimators=120)

model.fit(X_train_resampled, y_train_resampled)

pred = model.predict(X_test)

print(classification_report(y_test, pred))

joblib.dump(model, "model.pkl")

print("Model saved")

# confusion matrix
cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(4,3))

sns.heatmap(cm, annot=True, fmt="d")

plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")

plt.close()

# feature importance
importance = model.feature_importances_

plt.figure(figsize=(6,3))

plt.bar(range(len(importance)), importance)

plt.title("Feature Importance")

plt.savefig("feature_importance.png")

plt.close()

print("Graphs created successfully")