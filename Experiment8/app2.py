
import pickle
from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)

# Enable Prometheus metrics
metrics = PrometheusMetrics(app)


@app.route("/")
def home():
    return jsonify({"message": "app is running"})


@app.route("/health")
def health():
    return jsonify({"message": "app is healthy"})


@app.route("/predict", methods=["GET", "POST"])
def predict():
    return jsonify({"message": "Prediction endpoint is working"})


if __name__ == "__main__":
    app.run(debug=True)
