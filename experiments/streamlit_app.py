#Written using GPT-5.5

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

plt.style.use('dark_background')
plt.rcParams.update({
    'axes.facecolor': '#181C14',
    'figure.facecolor': 'none',
    'savefig.facecolor': 'none',
    'axes.edgecolor': '#697565',
    'axes.labelcolor': '#CFC4B3',
    'xtick.color': '#CFC4B3',
    'ytick.color': '#CFC4B3',
    'text.color': '#ECDFCC',
    'axes.titleweight': 'bold',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'font.family': 'serif'
})


# ------------------------------------------------------------
# Make imports work from the experiments/ directory
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from preprocessing_data import preprocess


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

MODEL_PATH = PROJECT_ROOT / "model_parameters.npz"


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def load_model(model_path):
    model = np.load(model_path)

    return {
        "weights": model["weights"],
        "bias": model["bias"],
        "mean": model["mean"],
        "std_deviation": model["standard_deviation"],
        "headers": model["headers"],
        "continuous_idx": model["continuous_idx"],
        "mse": model["mse"] if "mse" in model else None,
        "rmse": model["rmse"] if "rmse" in model else None,
        "mae": model["mae"] if "mae" in model else None,
        "r_squared": model["r_squared"] if "r_squared" in model else None,
        "J_history": model["J_history"] if "J_history" in model else None,
    }


def predict(X, w, b):
    return X @ w + b


def create_partial_effect_data(
    X,
    feature_idx,
    mean,
    std_deviation,
    w,
    b,
    number_of_points=200
):
    x_values = np.linspace(
        X[:, feature_idx].min(),
        X[:, feature_idx].max(),
        number_of_points
    )

    x_values_normalized = (
        x_values - mean[feature_idx]
    ) / std_deviation[feature_idx]

    X_partial = np.zeros((len(x_values), X.shape[1]))
    X_partial[:, feature_idx] = x_values_normalized

    y_partial = predict(X_partial, w, b)

    return x_values, y_partial


def plot_partial_effect(X, y, x_values, y_partial, feature_idx, feature_name):
    fig, ax = plt.subplots()

    ax.scatter(X[:, feature_idx], y, alpha=0.5, label="Actual data", color="#81957A")
    ax.plot(x_values, y_partial, label="Partial effect line", color="#D5D0C3")

    ax.set_xlabel(feature_name)
    ax.set_ylabel("Salary")
    ax.set_title(f"Partial effect of {feature_name} on salary")
    ax.legend()

    return fig


def plot_learning_curve(J_history):
    fig, ax = plt.subplots()

    ax.plot(np.arange(len(J_history)), J_history, color="#D5D0C3")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set_title("Learning curve")

    return fig


def plot_weights(headers, weights):
    fig, ax = plt.subplots()

    y_positions = np.arange(len(headers))

    ax.barh(y_positions, weights, color="#D5D0C3")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(headers)
    ax.set_xlabel("Weight value")
    ax.set_title("Learned weights by feature")

    return fig


# ------------------------------------------------------------
# Streamlit app
# ------------------------------------------------------------

st.set_page_config(
    page_title="MLR Training Dashboard",
    layout="wide"
)

st.title("Multiple Linear Regression Dashboard")

st.write(
    "Experimental GUI for inspecting the saved multiple linear regression model."
)


# ------------------------------------------------------------
# Load model and data
# ------------------------------------------------------------

if not MODEL_PATH.exists():
    st.error(f"Could not find model file at: {MODEL_PATH}")
    st.stop()

model = load_model(MODEL_PATH)

w_learned = model["weights"]
b_learned = model["bias"]
mean = model["mean"]
std_deviation = model["std_deviation"]
headers = model["headers"]
continuous_idx = model["continuous_idx"]

features, targets, _ = preprocess()

X = np.array(features)
y = np.array(targets)

if X.shape[1] != len(w_learned):
    st.error(
        f"Feature count mismatch: X has {X.shape[1]} columns, "
        f"but model has {len(w_learned)} weights."
    )
    st.stop()


# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------

st.sidebar.header("Controls")

continuous_feature_names = [
    headers[idx] for idx in continuous_idx
]

selected_feature_name = st.sidebar.selectbox(
    "Select feature for partial effect",
    continuous_feature_names
)

selected_feature_idx = int(
    continuous_idx[
        continuous_feature_names.index(selected_feature_name)
    ]
)

number_of_points = st.sidebar.slider(
    "Number of synthetic points",
    min_value=50,
    max_value=500,
    value=200,
    step=50
)


# ------------------------------------------------------------
# Model summary
# ------------------------------------------------------------

st.header("Model Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if model["rmse"] is not None:
        st.metric("RMSE", f"{model['rmse']:.4f}")
    else:
        st.metric("RMSE", "Not saved")

with col2:
    if model["mae"] is not None:
        st.metric("MAE", f"{model['mae']:.4f}")
    else:
        st.metric("MAE", "Not saved")

with col3:
    if model["mse"] is not None:
        st.metric("MSE", f"{model['mse']:.4f}")
    else:
        st.metric("MSE", "Not saved")

with col4:
    if model["r_squared"] is not None:
        st.metric("R²", f"{model['r_squared']:.4f}")
    else:
        st.metric("R²", "Not saved")


st.subheader("Loaded model artifact")

st.code(
    f"""
Weights shape: {w_learned.shape}
Bias: {b_learned}
Number of features: {X.shape[1]}
Continuous feature indices: {continuous_idx}
    """.strip()
)


# ------------------------------------------------------------
# Partial effect plot
# ------------------------------------------------------------

st.header("Partial Effect Plot")

x_values, y_partial = create_partial_effect_data(
    X=X,
    feature_idx=selected_feature_idx,
    mean=mean,
    std_deviation=std_deviation,
    w=w_learned,
    b=b_learned,
    number_of_points=number_of_points
)

partial_effect_fig = plot_partial_effect(
    X=X,
    y=y,
    x_values=x_values,
    y_partial=y_partial,
    feature_idx=selected_feature_idx,
    feature_name=selected_feature_name
)

st.pyplot(partial_effect_fig)

st.write(
    "The scatter points are real observations. "
    "The line is generated from synthetic examples where only the selected feature changes."
)


# ------------------------------------------------------------
# Learning curve
# ------------------------------------------------------------

st.header("Learning Curve")

if model["J_history"] is not None:
    learning_curve_fig = plot_learning_curve(model["J_history"])
    st.pyplot(learning_curve_fig)
else:
    st.info("J_history was not saved in model_parameters.npz.")


# ------------------------------------------------------------
# Learned weights
# ------------------------------------------------------------

st.header("Learned Weights")

weights_fig = plot_weights(headers, w_learned)
st.pyplot(weights_fig)


# ------------------------------------------------------------
# Raw model arrays
# ------------------------------------------------------------

with st.expander("Inspect raw model arrays"):
    st.write("Headers")
    st.write(headers)

    st.write("Weights")
    st.write(w_learned)

    st.write("Mean")
    st.write(mean)

    st.write("Standard deviation")
    st.write(std_deviation)