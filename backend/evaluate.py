"""Evaluation: python -m backend.evaluate

Tickets have no timestamps, so there is no time ordering to preserve; we use repeated
stratified 5-fold cross-validation. Every model is fit only on the training folds.
"""
import json
from collections import Counter

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold

from backend.triage import (SEVERITY_ORDER, fit_thresholds, TriageAgent, build_classifier, keyword_category,
                            load_data, preprocess, severity_score, train_classifier)

SEEDS = 3


def _X(df):
    return [preprocess(d, a) for d, a in zip(df["description"], df["asset_type"])]


def evaluate():
    df = load_data().reset_index(drop=True)
    X = np.array(_X(df), dtype=object)
    y_cat = df["target_category"].values
    y_sev = df["target_severity"].values
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=SEEDS, random_state=42)

    res = {k: {"acc": [], "f1": []} for k in ["cat_majority", "cat_keyword_rules", "cat_tfidf_lr", "sev_majority", "sev_tfidf_lr", "sev_rule_scorer", "agent_cat", "agent_sev"]}
    res["cat_ml_only_agent_path"] = {"acc": [], "f1": []}
    ml_used = []
    agent_cat_pred, agent_cat_true, failures = [], [], []
    sev_off_by_one = []

    for tr, te in rskf.split(X, y_cat):
        # --- category
        dm = DummyClassifier(strategy="most_frequent").fit(X[tr], y_cat[tr])
        m = build_classifier().fit(X[tr], y_cat[tr])
        kw = [keyword_category(t)[0] or "data_gap" for t in df["description"].values[te]]
        for name, pred in [("cat_majority", dm.predict(X[te])), ("cat_keyword_rules", kw), ("cat_tfidf_lr", m.predict(X[te]))]:
            res[name]["acc"].append(accuracy_score(y_cat[te], pred))
            res[name]["f1"].append(f1_score(y_cat[te], pred, average="macro"))
        # --- severity
        ds = DummyClassifier(strategy="most_frequent").fit(X[tr], y_sev[tr])
        ms = build_classifier().fit(X[tr], y_sev[tr])
        tr_scores = [severity_score(df.loc[i, "description"], y_cat[i])[0] for i in tr]
        th = fit_thresholds(tr_scores, y_sev[tr])  # cut-offs learned on training folds only
        rule = [severity_score(t, c, th)[1] for t, c in zip(df["description"].values[te], y_cat[te])]
        for name, pred in [("sev_majority", ds.predict(X[te])), ("sev_tfidf_lr", ms.predict(X[te])), ("sev_rule_scorer", rule)]:
            res[name]["acc"].append(accuracy_score(y_sev[te], pred))
            res[name]["f1"].append(f1_score(y_sev[te], pred, average="macro"))
        # --- full agent trained on train fold only
        agent = TriageAgent(model=m, thresholds=th)
        outs = [agent.run(df.loc[i, "description"], df.loc[i, "asset_type"]) for i in te]
        pc, ps = [o["category"] for o in outs], [o["severity"] for o in outs]
        used = [o["category_source"] == "ml" for o in outs]
        ml_used += used
        res["cat_ml_only_agent_path"]["acc"].append(float(np.mean([p == t for p, t, u in zip(pc, y_cat[te], used) if u] or [np.nan])))
        res["cat_ml_only_agent_path"]["f1"].append(0.0)
        res["agent_cat"]["acc"].append(accuracy_score(y_cat[te], pc))
        res["agent_cat"]["f1"].append(f1_score(y_cat[te], pc, average="macro"))
        res["agent_sev"]["acc"].append(accuracy_score(y_sev[te], ps))
        res["agent_sev"]["f1"].append(f1_score(y_sev[te], ps, average="macro"))
        agent_cat_pred += pc
        agent_cat_true += list(y_cat[te])
        for i, a, b, o in zip(te, ps, y_sev[te], outs):
            sev_off_by_one.append(abs(SEVERITY_ORDER.index(a) - SEVERITY_ORDER.index(b)))
            if a != b or o["category"] != y_cat[i]:
                failures.append((df.loc[i, "ticket_id"], df.loc[i, "description"], f"cat {y_cat[i]}->{o['category']}", f"sev {b}->{a}"))

    summary = {k: {"accuracy_mean": round(float(np.nanmean(v["acc"])), 3), "accuracy_std": round(float(np.nanstd(v["acc"])), 3),
                   "macro_f1_mean": round(float(np.mean(v["f1"])), 3)} for k, v in res.items()}
    summary["agent_severity_within_one_band"] = round(float(np.mean(np.array(sev_off_by_one) <= 1)), 3)
    labels = sorted(set(y_cat))
    cm = confusion_matrix(agent_cat_true, agent_cat_pred, labels=labels)
    fail_counts = Counter(f[0] for f in failures)
    summary["share_of_tickets_handled_by_ml_without_fallback"] = round(float(np.mean(ml_used)), 3)
    summary["n_tickets"] = len(df)
    summary["cv"] = f"5-fold stratified x {SEEDS} repeats"
    summary["most_frequent_failures"] = [
        {"ticket_id": t, "times_wrong": n, "description": next(f[1] for f in failures if f[0] == t),
         "detail": next(f"{f[2]}; {f[3]}" for f in failures if f[0] == t)}
        for t, n in fail_counts.most_common(6)
    ]
    summary["category_confusion_labels"] = labels
    summary["category_confusion_matrix"] = cm.tolist()
    return summary


if __name__ == "__main__":
    out = evaluate()
    print(json.dumps(out, indent=2))
    with open("evaluation_results.json", "w") as f:
        json.dump(out, f, indent=2)
