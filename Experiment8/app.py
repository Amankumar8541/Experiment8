
import joblib
import pandas as pd

model = joblib.load("Iris_Model.pkl")

sample = pd.DataFrame(
    [[1.3, 3.4, 4.2, 3.4]],
    columns=[
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
)

prediction = model.predict(sample)

print("Prediction of the sample is:", prediction[0])
