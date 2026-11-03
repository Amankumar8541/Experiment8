
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the original dataset
df = pd.read_csv("Iris - Iris.csv.xls")

# Separate features and target
X = df[["SepalLengthCm", "SepalWidthCm",
        "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"]

# Split the data in the same way as training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load the trained model
model = joblib.load("Iris_Model.pkl")

# Predict and evaluate
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print("Model testing completed!")
print("Accuracy of the model is:", acc)
