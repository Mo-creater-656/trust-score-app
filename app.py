import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score

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

st.title("🔍 Composite Trust Score — Bot/Human Detector")
st.caption(
    "4 pillar Random Forests (Behavioral, Linguistic, Interaction, Metadata) "
    "fused by a Logistic Regression meta-model into one trust score."
)

with st.sidebar:
    st.header("Pick an account")
    demo_df = df.loc[test_idx].reset_index(drop=True)
    screen_names = demo_df["screen_name"].astype(str).tolist()
    choice = st.selectbox("Held-out demo account", screen_names, index=0)
    row_idx = demo_df.index[demo_df["screen_name"].astype(str) == choice][0]
    orig_idx = test_idx[row_idx]

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
        st.write(f"**Ground-truth label (test set):** {actual}")
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

st.markdown("### Why this score? (top contributing features per pillar)")
explain_cols = st.columns(len(PILLARS))
for col, (pillar, cols) in zip(explain_cols, PILLARS.items()):
    with col:
        st.markdown(f"**{pillar.capitalize()}**")
        X_row = row[cols]
        explainer = shap.TreeExplainer(pillar_models[pillar])
        sv = explainer.shap_values(X_row, check_additivity=False)
        sv = sv[..., 1][0]  # "malicious" class
        order = np.argsort(-np.abs(sv))[:2]
        for i in order:
            feat = cols[i]
            val = X_row[feat].values[0]
            desc = describe_feature(feat, val, feature_medians)
            push = "toward malicious" if sv[i] > 0 else "toward legitimate"
            st.write(f"- account {desc} → {push}")

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
        """
    )

st.caption(
    "Demo app — accounts shown are from the held-out test split, so ground-truth labels are known "
    "and can be compared against the predicted trust score/tier."
)
