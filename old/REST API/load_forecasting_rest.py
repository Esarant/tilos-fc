from flask import Flask
from utils import load_forecasting_functions

app = Flask(__name__)

# return random specific prediction
@app.route("/decision_tree", methods=["POST"])
def decision_tree_prediction():
    result = load_forecasting_functions.decision_tree_rest()
    return result

@app.route("/lightgbm", methods=["POST"])
def lightgbm_prediction():
    result = load_forecasting_functions.lightgbm_rest()
    return result

@app.route("/xgboost", methods=["POST"])
def xgboost_prediction():
    result = load_forecasting_functions.xgboost_rest()
    return result

@app.route("/ensemble", methods=["POST"])
def ensemble_prediction():
    result = load_forecasting_functions.ensemble_rest()
    return result

if __name__ == '__main__':
	app.run(host="0.0.0.0", port=7000, debug=True)
