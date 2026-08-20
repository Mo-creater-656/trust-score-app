"""
Trains the 4 pillar Random Forests + the fusion meta-model exactly like
step3_train_classifiers.ipynb, then saves everything the Streamlit app needs
(models, thresholds, feature medians) into models/bundle.joblib.

Run once locally before deploying:
    python train_models.py
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

PILLARS = {
    "behavioral": ["avg_tweets_per_day", "posting_rate_consistency", "high_volume_flag"],
    "linguistic": [
        "bio_length", "bio_word_count", "bio_lexical_diversity", "bio_has_url",
        "bio_hashtag_count", "bio_mention_count", "bio_exclaim_ratio",
        "bio_sentiment_score", "has_description",
    ],
    "interaction": [
        "followers_count", "friends_count", "follower_friend_ratio",
        "favourites_count", "favourites_per_day",
    ],
    "metadata": [
        "account_age_days", "verified", "default_profile", "default_profile_image",
        "geo_enabled", "has_location", "has_lang",
    ],
}
LABEL_COL = "label"

print("Loading features.csv ...")
df = pd.read_csv("features.csv")
print(df.shape)

train_idx, test_idx = train_test_split(
    df.index, test_size=0.2, stratify=df[LABEL_COL], random_state=RANDOM_STATE
)
y_train = df.loc[train_idx, LABEL_COL]
y_test = df.loc[test_idx, LABEL_COL]

pillar_models = {}
pillar_train_probs = {}
pillar_test_probs = {}
pillar_metrics = {}

for pillar, cols in PILLARS.items():
    X_train = df.loc[train_idx, cols]
    X_test = df.loc[test_idx, cols]

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    train_probs = clf.predict_proba(X_train)[:, 1]
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_preds = clf.predict(X_test)

    pillar_models[pillar] = clf
    pillar_train_probs[pillar] = train_probs
    pillar_test_probs[pillar] = test_probs
    pillar_metrics[pillar] = {
        "accuracy": accuracy_score(y_test, test_preds),
        "f1": f1_score(y_test, test_preds),
    }
    print(f"{pillar:12s} acc={pillar_metrics[pillar]['accuracy']:.3f} f1={pillar_metrics[pillar]['f1']:.3f}")

meta_train = pd.DataFrame(pillar_train_probs, index=train_idx)
meta_test = pd.DataFrame(pillar_test_probs, index=test_idx)

meta_model = LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE)
meta_model.fit(meta_train, y_train)

trust_scores_test = meta_model.predict_proba(meta_test)[:, 1]
fused_preds = meta_model.predict(meta_test)
fused_acc = accuracy_score(y_test, fused_preds)
fused_f1 = f1_score(y_test, fused_preds)
pillar_metrics["FUSED"] = {"accuracy": fused_acc, "f1": fused_f1}
print(f"FUSED        acc={fused_acc:.3f} f1={fused_f1:.3f}")

# Risk tier thresholds, from score distribution (same as notebook)
low, high = np.percentile(trust_scores_test, [60, 85])
print(f"Thresholds -> Legitimate < {low:.3f} <= Suspicious < {high:.3f} <= Malicious")

# Feature medians, used by the app to describe "high"/"low" for continuous features
ALL_COLS = sum(PILLARS.values(), [])
feature_medians = df[ALL_COLS].median().to_dict()

bundle = {
    "pillar_models": pillar_models,
    "meta_model": meta_model,
    "pillars": PILLARS,
    "thresholds": {"low": float(low), "high": float(high)},
    "feature_medians": feature_medians,
    "metrics": pillar_metrics,
    "test_idx": list(test_idx),  # so the demo app can pick real held-out accounts
    # test-set ground truth + fused trust scores, so the app can recompute
    # confusion-matrix / FPR / FNR live for any threshold the user drags to
    "test_labels": y_test.values.tolist(),
    "test_trust_scores": trust_scores_test.tolist(),
}

joblib.dump(bundle, "models/bundle.joblib", compress=3)
print("\nSaved models/bundle.joblib (compressed)")
