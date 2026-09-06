import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score
from lime.lime_tabular import LimeTabularExplainer

st.set_page_config(page_title="Bot/Human Composite Trust Score", layout="wide")

BINARY_FEATURE_LABELS = {
    "bio_has_url": ("has a URL in their bio", "has no URL in their bio"),
    "has_description": ("has a bio", "has no bio"),
    "verified": ("is verified", "is not verified"),
    "default_profile": ("uses the default profile theme", "customized their profile theme"),
    "default_profile_image": ("uses the default profile picture", "has a custom profile picture"),
    "geo_enabled": ("has location sharing enabled", "has location sharing disabled"),
    "has_location": ("lists a location", "lists no location"),
    "has_lang": ("has a language set", "has no language set"),
    "high_volume_flag": ("posts at an unusually high volume", "posts at a normal volume"),
}


@st.cache_resource
def load_bundle():
    return joblib.load("models/bundle.joblib")


@st.cache_resource
def build_lime_explainers(_pillars, _train_df):
    """One LimeTabularExplainer per pillar, built once and reused across accounts."""
    explainers = {}
    for pillar, cols in _pillars.items():
        explainers[pillar] = LimeTabularExplainer(
            _train_df[cols].values,
            feature_names=cols,
            class_names=["legitimate", "malicious"],
            discretize_continuous=True,
            mode="classification",
            random_state=42,
        )
    return explainers


def lime_explain(explainer, model, cols, x_row, num_features=2, num_samples=500):
    def predict_fn(x, cols=cols, model=model):
        return model.predict_proba(pd.DataFrame(x, columns=cols))

    exp = explainer.explain_instance(
        x_row, predict_fn, num_features=num_features, num_samples=num_samples, labels=(1,)
    )
    return exp.as_list(label=1)  # [(condition_str, weight), ...] weight>0 -> pushes toward malicious


def simplify_lime_condition(condition, feature_medians):
    """Turn LIME's raw split condition (e.g. 'followers_count <= 36.00') into the same
    plain-English style used for the SHAP descriptions, so the two are easy to compare."""
    feat = next((f for f in feature_medians if condition.startswith(f) or f"< {f}" in condition or f" {f} " in f" {condition} "), None)
    if feat is None:
        return condition
    if feat in BINARY_FEATURE_LABELS:
        is_zero = "<= 0.00" in condition or "< 0.50" in condition
        return BINARY_FEATURE_LABELS[feat][1 if is_zero else 0]
    # extract the numeric boundary LIME split on, compare it to this feature's median
    import re
    nums = re.findall(r"-?\d+\.?\d*", condition)
    boundary = float(nums[-1]) if nums else feature_medians[feat]
    median = feature_medians[feat]
    direction = "low" if boundary <= median else "high"
    return f"{direction} {feat.replace('_', ' ')}"


@st.cache_data
def load_features():
    return pd.read_csv("features.csv")


def describe_feature(feature, value, feature_medians):
    if feature in BINARY_FEATURE_LABELS:
        return BINARY_FEATURE_LABELS[feature][0 if value == 1 else 1]
    median = feature_medians[feature]
    direction = "high" if value > median else "low"
    return f"{direction} {feature.replace('_', ' ')}"


def tier(score, low, high):
    if score < low:
        return "Legitimate"
    elif score < high:
        return "Suspicious"
    return "Malicious"


TIER_COLOR = {"Legitimate": "#2ecc71", "Suspicious": "#f39c12", "Malicious": "#e74c3c"}

bundle = load_bundle()
df = load_features()

pillar_models = bundle["pillar_models"]
meta_model = bundle["meta_model"]
PILLARS = bundle["pillars"]
low, high = bundle["thresholds"]["low"], bundle["thresholds"]["high"]
feature_medians = bundle["feature_medians"]
metrics = bundle["metrics"]
test_idx = bundle["test_idx"]
train_idx = bundle["train_idx"]

st.title("🔍 Composite Trust Score — Bot/Human Detector")
st.caption(
    "4 pillar Random Forests (Behavioral, Linguistic, Interaction, Metadata) "
    "fused by a Logistic Regression meta-model into one trust score."
)

with st.sidebar:
    st.header("Pick an account")
    st.caption(
        f"Choose from all {len(df):,} accounts in the dataset. Accounts are tagged as "
        "**Held-out** (never seen during training — a true test of generalization) or "
        "**Trained-on** (the model saw this exact account while learning, so its score "
        "reflects fit, not generalization)."
    )

    test_idx_set = set(test_idx)
    demo_df = df.copy().reset_index()  # keep original df index in a column
    demo_df["split"] = demo_df["index"].apply(lambda i: "Held-out" if i in test_idx_set else "Trained-on")
    demo_df["display_name"] = demo_df["screen_name"].astype(str) + "  ·  " + demo_df["split"]

    split_filter = st.radio("Show", ["All accounts", "Held-out only", "Trained-on only"], horizontal=True)
    if split_filter == "Held-out only":
        options_df = demo_df[demo_df["split"] == "Held-out"]
    elif split_filter == "Trained-on only":
        options_df = demo_df[demo_df["split"] == "Trained-on"]
    else:
        options_df = demo_df

    display_names = options_df["display_name"].tolist()
    choice = st.selectbox(f"Account ({len(display_names):,} available)", display_names, index=0)
    orig_idx = options_df.loc[options_df["display_name"] == choice, "index"].values[0]
    is_held_out = orig_idx in test_idx_set

    st.markdown("---")
    st.subheader("Model performance (held-out test set)")
    metrics_df = pd.DataFrame(metrics).T
    st.dataframe(metrics_df.style.format("{:.3f}"))

row = df.loc[[orig_idx]]
true_label = row["label"].values[0] if "label" in row.columns else None

# --- Compute pillar probs + fused trust score for this account ---
pillar_probs = {}
for pillar, cols in PILLARS.items():
    X = row[cols]
    pillar_probs[pillar] = pillar_models[pillar].predict_proba(X)[:, 1][0]

meta_X = pd.DataFrame([pillar_probs])[list(PILLARS.keys())]
trust_score = meta_model.predict_proba(meta_X)[:, 1][0]
account_tier = tier(trust_score, low, high)

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    st.metric("Composite Trust Score", f"{trust_score:.3f}", help="0 = trustworthy, 1 = malicious-leaning")
with col2:
    st.markdown(
        f"<div style='padding:12px;border-radius:8px;background:{TIER_COLOR[account_tier]};"
        f"color:white;text-align:center;font-size:20px;font-weight:bold;'>{account_tier}</div>",
        unsafe_allow_html=True,
    )
with col3:
    if true_label is not None:
        actual = "Bot" if true_label == 1 else "Human"
        if is_held_out:
            st.write(f"**Ground-truth label:** {actual} · held-out (true generalization test)")
        else:
            st.write(f"**Ground-truth label:** {actual} · ⚠️ trained-on (model saw this account)")
    st.write(f"**Thresholds:** Legitimate < {low:.3f} ≤ Suspicious < {high:.3f} ≤ Malicious")

st.markdown("### Pillar breakdown")
pillar_bar_df = pd.DataFrame({"pillar": list(pillar_probs.keys()), "probability": list(pillar_probs.values())})
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(pillar_bar_df["pillar"], pillar_bar_df["probability"], color="#3498db")
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
ax.set_ylim(0, 1)
ax.set_ylabel("P(malicious) per pillar")
plt.tight_layout()
st.pyplot(fig)

st.markdown("### Why this score? Dual-layer explainability (SHAP + LIME)")
st.caption(
    "SHAP (global, exact for tree models) and LIME (local surrogate — tests what-if tweaks around "
    "this one account) are computed independently. When they agree, that's a strong signal the "
    "explanation is real, not a quirk of one method. When they disagree, that's shown too — it's "
    "useful information, not a bug to hide."
)

lime_explainers = build_lime_explainers(PILLARS, df.loc[train_idx])

explain_cols = st.columns(len(PILLARS))
for col, (pillar, cols) in zip(explain_cols, PILLARS.items()):
    with col:
        st.markdown(f"**{pillar.capitalize()}**")
        X_row = row[cols]

        # --- SHAP (global, exact) ---
        shap_explainer = shap.TreeExplainer(pillar_models[pillar])
        sv = shap_explainer.shap_values(X_row, check_additivity=False)
        sv = sv[..., 1][0]  # "malicious" class
        shap_order = np.argsort(-np.abs(sv))[:2]
        shap_top_feat = cols[shap_order[0]]

        st.caption("SHAP says:")
        for i in shap_order:
            feat = cols[i]
            val = X_row[feat].values[0]
            desc = describe_feature(feat, val, feature_medians)
            push = "toward malicious" if sv[i] > 0 else "toward legitimate"
            st.write(f"- account {desc} → {push}")

        # --- LIME (local surrogate, second opinion) ---
        lime_results = lime_explain(
            lime_explainers[pillar], pillar_models[pillar], cols, X_row.values[0], num_features=2
        )
        lime_top_feat = next(
            (f for f in feature_medians if lime_results[0][0].startswith(f) or f" {f} " in f" {lime_results[0][0]} "),
            None,
        )

        st.caption("LIME says:")
        for condition, weight in lime_results:
            desc = simplify_lime_condition(condition, feature_medians)
            push = "toward malicious" if weight > 0 else "toward legitimate"
            st.write(f"- account has {desc} → {push}")

        agree = shap_top_feat == lime_top_feat
        if agree:
            st.success("✓ Agree on top feature", icon="✅")
        else:
            st.info("Differ on top feature — expected sometimes, methods work differently", icon="ℹ️")

st.markdown("---")
st.header("📊 Model Diagnostics")

test_labels = np.array(bundle["test_labels"])
test_scores = np.array(bundle["test_trust_scores"])
auc = roc_auc_score(test_labels, test_scores)

st.write(
    f"Evaluated on the held-out test set ({len(test_labels)} accounts, never seen during training). "
    f"ROC-AUC: **{auc:.3f}**"
)

st.subheader("Malicious-tier threshold (live trade-off)")
st.caption(
    "The Malicious cutoff was originally set at the 85th percentile of trust scores. "
    "Drag it to see the false-positive / false-negative trade-off in real time."
)
thresh = st.slider("Malicious classification threshold", 0.0, 1.0, float(high), 0.01)

preds_at_thresh = (test_scores >= thresh).astype(int)
cm = confusion_matrix(test_labels, preds_at_thresh)
tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn) if (fp + tn) else 0
fnr = fn / (fn + tp) if (fn + tp) else 0
precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0

d1, d2, d3, d4 = st.columns(4)
d1.metric("False Positive Rate", f"{fpr:.1%}", help="Legitimate accounts wrongly flagged as malicious")
d2.metric("False Negative Rate", f"{fnr:.1%}", help="Malicious accounts missed")
d3.metric("Precision", f"{precision:.1%}")
d4.metric("Recall", f"{recall:.1%}")

st.caption(
    f"At this threshold: TN={tn}, FP={fp}, FN={fn}, TP={tp}. "
    "Raising the threshold reduces false accusations of real users (lower FPR) but lets more bots through (higher FNR) — and vice versa."
)

with st.expander("ℹ️ Methodology notes (dataset checks, imbalance handling, validation)"):
    st.markdown(
        """
- **Class balance:** 66.8% human / 33.2% bot (~2:1) — mild imbalance, handled with `class_weight="balanced"`
  in both the pillar Random Forests and the fusion Logistic Regression, rather than synthetic oversampling
  (SMOTE/ADASYN), which is unnecessary at this ratio and risks introducing artifact samples.
- **Missing values / duplicates:** none found in `features.csv` (checked by row, by `id`, and by `screen_name`).
- **Validation strategy:** a single stratified 80/20 train/test split is used for the reported metrics above,
  cross-checked with 5-fold stratified cross-validation per pillar (F1 std ≤ 0.009 across folds) to confirm the
  split wasn't a lucky/unlucky draw. No separate validation set was needed since hyperparameters are fixed
  rather than grid-searched.
- **bio_sentiment_score:** a simple lexicon count (positive word matches − negative word matches from a small
  hardcoded word list), not a normalized VADER-style score — hence the small integer range.
- **Dual-layer explainability:** SHAP (`TreeExplainer`, exact for tree models) is the global explanation layer;
  LIME (`LimeTabularExplainer`, local surrogate) independently re-explains the same prediction by testing
  perturbed versions of the account. They're computed with completely different math, so agreement between
  them is meaningful evidence the explanation is real — and disagreement is shown transparently rather than
  hidden, since the two methods answering "why" in different ways is expected, not an error.
        """
    )

st.caption(
    "Demo app — scores any of the 37,438 labeled accounts in the dataset, tagged Held-out "
    "(true generalization test) or Trained-on (model saw this account while learning). "
    "Scoring a brand-new, unlabeled username from a live platform API is the next build step, "
    "not yet implemented."
)
