from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# ------------------------------------------------------------
# Project path setup
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
# Blog-inspired theme constants
# ------------------------------------------------------------

BACKGROUND = "#181C14"
PANEL = "rgba(60, 61, 55, 0.62)"
PANEL_DARK = "rgba(40, 46, 33, 0.86)"
TEXT = "#ECDFCC"
MUTED_TEXT = "#CFC4B3"
SUBTLE_TEXT = "#A9A295"
ACCENT = "#697565"
SOFT_GREEN = "#81957A"
GRID = "#3C3D37"


# ------------------------------------------------------------
# Streamlit page config
# ------------------------------------------------------------

st.set_page_config(
    page_title="MLR Dashboard",
    page_icon="📜",
    layout="wide"
)


# ------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------

st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Volkhov:wght@400;700&display=swap');

        .stApp {{
            background: {BACKGROUND};
            color: {TEXT};
            font-family: 'Libre Baskerville', serif;
        }}

        [data-testid="stHeader"] {{
            background: rgba(24, 28, 20, 0.85);
        }}

        section[data-testid="stSidebar"] {{
            background: {PANEL_DARK};
            border-right: 1px solid {ACCENT};
        }}

        section[data-testid="stSidebar"] * {{
            color: {MUTED_TEXT};
            font-family: 'Libre Baskerville', serif;
        }}

        h1, h2, h3 {{
            font-family: 'Volkhov', serif;
            color: {TEXT};
            letter-spacing: 0.02em;
        }}

        p, li, label, div {{
            color: {MUTED_TEXT};
        }}

        .main-title {{
            border: 1px solid {ACCENT};
            border-radius: 12px;
            background: {PANEL};
            padding: 1.5rem 1.8rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 0 30px rgba(0,0,0,0.25);
        }}

        .main-title h1 {{
            margin-bottom: 0.4rem;
            font-size: 2.2rem;
        }}

        .main-title p {{
            color: {MUTED_TEXT};
            font-style: italic;
            margin-bottom: 0;
        }}

        .metric-card {{
            border: 1px solid rgba(105, 117, 101, 0.65);
            border-radius: 10px;
            background: {PANEL};
            padding: 1rem;
            text-align: center;
            min-height: 105px;
        }}

        .metric-label {{
            color: {SUBTLE_TEXT};
            font-size: 0.9rem;
            margin-bottom: 0.4rem;
        }}

        .metric-value {{
            color: {TEXT};
            font-family: 'Volkhov', serif;
            font-size: 1.45rem;
            font-weight: 700;
        }}

        .section-card {{
            border: 1px solid rgba(105, 117, 101, 0.5);
            border-radius: 12px;
            background: rgba(60, 61, 55, 0.35);
            padding: 1.2rem;
            margin-top: 1rem;
            margin-bottom: 1.5rem;
        }}

        .small-note {{
            color: {SUBTLE_TEXT};
            font-size: 0.95rem;
            font-style: italic;
        }}

        div[data-testid="stCodeBlock"] {{
            border: 1px solid rgba(105, 117, 101, 0.5);
            border-radius: 8px;
        }}

        .stButton > button {{
            background: rgba(60, 61, 55, 0.7);
            color: {TEXT};
            border: 1px solid {ACCENT};
            border-radius: 8px;
            font-family: 'Libre Baskerville', serif;
        }}

        .stButton > button:hover {{
            background: {ACCENT};
            color: {BACKGROUND};
            border: 1px solid {ACCENT};
        }}

        a {{
            color: {SOFT_GREEN};
        }}

        hr {{
            border: none;
            height: 2px;
            background: {GRID};
            margin: 1.5rem 0;
        }}
    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

@st.cache_data
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


@st.cache_data
def load_dataset():
    features, targets, _ = preprocess()
    return np.array(features), np.array(targets)


def predict(X, w, b):
    return X @ w + b


def style_matplotlib_axis(ax):
    ax.set_facecolor(BACKGROUND)

    ax.tick_params(colors=MUTED_TEXT)
    ax.xaxis.label.set_color(MUTED_TEXT)
    ax.yaxis.label.set_color(MUTED_TEXT)
    ax.title.set_color(TEXT)

    for spine in ax.spines.values():
        spine.set_color(ACCENT)

    ax.grid(True, alpha=0.18)


def create_partial_effect_data(
    X,
    feature_idx,
    continuous_idx,
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

    # Convert the observed feature matrix into the representation expected by
    # the trained model. One-hot and cyclic features are already model-ready;
    # only the continuous columns were standardized during training.
    X_model = np.array(X, dtype=float, copy=True)
    for idx in continuous_idx:
        idx = int(idx)
        if std_deviation[idx] != 0:
            X_model[:, idx] = (
                X_model[:, idx] - mean[idx]
            ) / std_deviation[idx]

    # Partial dependence at a probe value is the average prediction obtained
    # after replacing the selected feature for every observed employee while
    # leaving that employee's other encoded features unchanged.
    y_partial = np.empty(len(x_values), dtype=float)
    for probe_idx, x_value in enumerate(x_values):
        X_probe = X_model.copy()
        X_probe[:, feature_idx] = (
            x_value - mean[feature_idx]
        ) / std_deviation[feature_idx]
        y_partial[probe_idx] = np.mean(predict(X_probe, w, b))

    return x_values, y_partial


def plot_partial_effect(X, y, x_values, y_partial, feature_idx, feature_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BACKGROUND)

    ax.scatter(
        X[:, feature_idx],
        y,
        alpha=0.55,
        label="Actual data",
        color=SOFT_GREEN,
        edgecolors="none"
    )

    ax.plot(
        x_values,
        y_partial,
        label="Average partial-effect trend",
        color=TEXT,
        linewidth=1.7,
        alpha=0.72
    )

    ax.scatter(
        x_values,
        y_partial,
        label=f"Synthetic probes (n={len(x_values)})",
        color=TEXT,
        edgecolors=BACKGROUND,
        linewidths=0.7,
        marker="D",
        s=34,
        zorder=3
    )

    ax.set_xlabel(feature_name)
    ax.set_ylabel("Salary")
    ax.set_title(f"Partial effect of {feature_name} on salary")
    ax.legend(facecolor=BACKGROUND, edgecolor=ACCENT, labelcolor=MUTED_TEXT)

    style_matplotlib_axis(ax)
    fig.tight_layout()

    return fig


def plot_learning_curve(J_history):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BACKGROUND)

    ax.plot(
        np.arange(len(J_history)),
        J_history,
        color=SOFT_GREEN,
        linewidth=2
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set_title("Learning curve")

    style_matplotlib_axis(ax)
    fig.tight_layout()

    return fig


def plot_weights(headers, weights):
    fig_height = max(5, len(headers) * 0.35)

    fig, ax = plt.subplots(figsize=(8, fig_height))
    fig.patch.set_facecolor(BACKGROUND)

    y_positions = np.arange(len(headers))

    ax.barh(
        y_positions,
        weights,
        color=SOFT_GREEN,
        alpha=0.85
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(headers)
    ax.set_xlabel("Weight value")
    ax.set_title("Learned weights by feature")
    ax.axvline(0, color=TEXT, linewidth=1, alpha=0.55)

    style_matplotlib_axis(ax)
    fig.tight_layout()

    return fig


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.markdown(
    """
    <div class="main-title">
        <h1>Multiple Linear Regression Dashboard</h1>
        <p>
            A small interpretability chamber for the salary estimation model:
            learning curves, learned weights, and partial effects.
        </p>
    </div>
    """,
    unsafe_allow_html=True
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

X, y = load_dataset()

if X.shape[1] != len(w_learned):
    st.error(
        f"Feature count mismatch: X has {X.shape[1]} columns, "
        f"but model has {len(w_learned)} weights."
    )
    st.stop()


# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------

st.sidebar.title("The Console")

st.sidebar.markdown(
    """
    Select a continuous feature and inspect how the model's average prediction
    changes when that feature is varied across its observed range.
    """
)

continuous_feature_names = [
    str(headers[idx]) for idx in continuous_idx
]

selected_feature_name = st.sidebar.selectbox(
    "Feature",
    continuous_feature_names
)

selected_feature_idx = int(
    continuous_idx[
        continuous_feature_names.index(selected_feature_name)
    ]
)

number_of_points = st.sidebar.slider(
    "Synthetic probe points",
    min_value=5,
    max_value=50,
    value=15,
    step=5,
    help=(
        "Controls how many diamond-shaped model probes are drawn across the "
        "selected feature's observed range. It changes sampling density, not "
        "the fitted linear relationship."
    )
)

show_raw_arrays = st.sidebar.checkbox(
    "Show raw model arrays",
    value=False
)


# ------------------------------------------------------------
# Model summary
# ------------------------------------------------------------

st.header("Model Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    metric_card(
        "RMSE",
        f"{model['rmse']:.4f}" if model["rmse"] is not None else "Not saved"
    )

with col2:
    metric_card(
        "MAE",
        f"{model['mae']:.4f}" if model["mae"] is not None else "Not saved"
    )

with col3:
    metric_card(
        "MSE",
        f"{model['mse']:.4f}" if model["mse"] is not None else "Not saved"
    )

with col4:
    metric_card(
        "R²",
        f"{model['r_squared']:.4f}" if model["r_squared"] is not None else "Not saved"
    )


st.markdown('<div class="section-card">', unsafe_allow_html=True)

st.subheader("Loaded artifact")

st.code(
    f"""
Weights shape: {w_learned.shape}
Bias: {b_learned}
Number of observations: {X.shape[0]}
Number of features: {X.shape[1]}
Continuous feature indices: {continuous_idx}
    """.strip()
)

st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------
# Partial effect plot
# ------------------------------------------------------------

st.header("Partial Effect")

st.markdown(
    """
    A **partial effect** asks a model-focused question: *how does the model's
    average predicted salary change as one feature changes, after accounting for
    the observed values of all the other features?* It describes the behaviour
    learned by this model; it does **not** establish that changing the feature
    causes salary to change.
    """
)

with st.expander("How the partial effect is computed", expanded=True):
    st.markdown(
        """
        1. Create evenly spaced probe values from the selected feature's minimum
           to maximum observed value.
        2. For each probe value, copy every observed employee profile and replace
           only the selected feature with that value.
        3. Standardize the replacement value with the training mean and standard
           deviation, then predict a salary for every copied profile.
        4. Average those predictions. The resulting average is one point on the
           partial-effect curve.

        In notation, for feature $j$ and probe value $x$:
        """
    )
    st.latex(
        r"\operatorname{PD}_j(x)=\frac{1}{n}\sum_{i=1}^{n} "
        r"\hat f\!\left(x,\mathbf{x}_{i,-j}\right)"
    )
    st.markdown(
        r"""
        Because this is an additive linear model without interaction terms, the
        result must be a straight line. In raw feature units its slope is
        $w_j/\sigma_j$; adding more probes therefore makes the diamonds denser
        without bending or otherwise changing the line.
        """
    )

x_values, y_partial = create_partial_effect_data(
    X=X,
    feature_idx=selected_feature_idx,
    continuous_idx=continuous_idx,
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

raw_unit_effect = (
    w_learned[selected_feature_idx]
    / std_deviation[selected_feature_idx]
)
observed_range_effect = y_partial[-1] - y_partial[0]

effect_col1, effect_col2, effect_col3 = st.columns(3)
with effect_col1:
    metric_card("Probe points shown", f"{number_of_points}")
with effect_col2:
    metric_card(
        "Prediction change per +1 raw unit",
        f"{raw_unit_effect:+,.2f}"
    )
with effect_col3:
    metric_card(
        "Change across observed range",
        f"{observed_range_effect:+,.2f}"
    )

st.markdown(
    """
    <p class="small-note">
        The green circles are observed employees and salaries. The diamonds are
        synthetic model probes averaged over the observed employee profiles. The
        vertical distance through the raw scatter is also affected by profession,
        equipment, season and the remaining continuous features, so the partial-effect
        trend is not expected to pass through every observation.
    </p>
    """,
    unsafe_allow_html=True
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

if show_raw_arrays:
    st.header("Raw Model Arrays")

    with st.expander("Inspect raw model arrays", expanded=True):
        st.write("Headers")
        st.write(headers)

        st.write("Weights")
        st.write(w_learned)

        st.write("Mean")
        st.write(mean)

        st.write("Standard deviation")
        st.write(std_deviation)
