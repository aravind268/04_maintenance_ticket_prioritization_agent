"""Core logic for the maintenance ticket triage agent (framework-free, so it is easy to test).

Flow:  raw ticket -> preprocess -> [tool] classify_category (TF-IDF + LogisticRegression)
       -> [tool] score_severity (transparent rule scoring) -> [tool] lookup_playbook
       -> [tool] check_escalation -> routing decision + reasoning trace.

No LLM and no RAG are used. The "agent" is a deterministic tool-calling workflow with
branching decisions (see TriageAgent.run). No API keys are needed.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "maintenance_tickets.csv"

SEVERITY_ORDER = ["low", "medium", "high", "critical"]

# ----------------------------------------------------------------------------- data


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["description"] = df["description"].fillna("").astype(str)
    df["asset_type"] = df["asset_type"].fillna("unknown").str.lower().str.strip()
    return df


def preprocess(text: str, asset_type: str = "") -> str:
    """Lowercase, normalise turbine/inverter IDs (WTG-12 -> wtgid), strip punctuation.
    Asset type is prepended as a token so the model can use it as a feature."""
    t = (text or "").lower()
    t = re.sub(r"\b(wtg|inv|inverter|block|unit|blade)[- ]?\d+\b", r"\1id", t)
    t = re.sub(r"\d+(\.\d+)?", " num ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return f"asset_{asset_type} {t}".strip()


# ----------------------------------------------------------------------------- model


def build_classifier() -> Pipeline:
    """TF-IDF unigrams+bigrams -> multinomial logistic regression.
    ngram_range=(1,2): captures phrases like 'oil temperature'.
    C=100: weak regularisation. C is the inverse regularisation strength: lower C = smoother,
    less overfit model but under-confident probabilities on tiny data (C=10 gave median
    confidence 0.42, which made the confidence threshold useless); higher C = sharper fit.
    C in {10,100,1000} gave similar CV accuracy (~0.84-0.86), so 100 is a middle choice."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            ("clf", LogisticRegression(C=100, max_iter=3000)),
        ]
    )


def train_classifier(df: pd.DataFrame | None = None) -> Pipeline:
    df = load_data() if df is None else df
    X = [preprocess(d, a) for d, a in zip(df["description"], df["asset_type"])]
    model = build_classifier()
    model.fit(X, df["target_category"])
    return model


# ----------------------------------------------------------------------------- rules

# keyword -> category, used as (a) a baseline model and (b) the agent's fallback tool.
KEYWORD_RULES: dict[str, list[str]] = {
    "safety_hazard": ["smoke", "fire", "gas leak", "spark", "exposed live", "evacuat", "injury", "slip hazard", "handrail", "signage", "burnt"],
    "transformer_fault": ["transformer", "buchholz", "tap changer", "dissolved gas"],
    "gearbox_vibration": ["gearbox", "vibration"],
    "pitch_system": ["pitch"],
    "inverter_thermal": ["inverter"],
    "grid_curtailment": ["curtail", "backdown", "ramp rate", "reactive power", "dispatch"],
    "data_gap": ["scada", "telemetry", "missing", "not updating", "not reporting", "data", "historian", "meter reading"],
    "inspection_compliance": ["inspection", "checklist", "certificate", "compliance", "overdue", "statutory", "audit", "test record"],
}


def keyword_category(text: str) -> tuple[str | None, list[str]]:
    t = (text or "").lower()
    best, best_hits = None, []
    for cat, kws in KEYWORD_RULES.items():  # dict order = priority order
        hits = [k for k in kws if k in t]
        if len(hits) > len(best_hits):
            best, best_hits = cat, hits
    return best, best_hits


# Severity: base score per category + keyword modifiers -> band. Weights are expert-style
# assumptions (not learned), chosen so the logic can be explained to an operations reader.
CATEGORY_BASE = {
    "safety_hazard": 60, "transformer_fault": 50, "gearbox_vibration": 45, "pitch_system": 40,
    "inverter_thermal": 40, "grid_curtailment": 30, "data_gap": 25, "inspection_compliance": 20,
}
UP = {  # phrase -> points added
    "critical_terms": (["smoke", "fire", "evacuat", "tripped", "will not restart", "stopped", "metal particles",
                        "acetylene", "buchholz", "gas leak", "injury", "sparking", "may not feather", "repeated"], 25),
    "high_terms": (["alarm", "abnormal", "above", "overdue", "regulator", "offline", "entire", "derat", "not applied",
                    "exporting above", "mismatch", "grinding", "stuck", "oil leak", "high"], 12),
}
DOWN = (["minor", "slight", "routine", "cleared by itself", "auto reset", "marginally", "within limits",
         "still normal", "scheduled", "not yet scheduled", "blank", "small margin", "outdated", "missing warning"], -18)


def severity_score(text: str, category: str, thresholds=None) -> tuple[int, str, list[str]]:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    t = (text or "").lower()
    score = CATEGORY_BASE.get(category, 30)
    reasons = [f"base {score} for category '{category}'"]
    crit, pts = UP["critical_terms"]
    hits = [k for k in crit if k in t]
    if hits:
        score += pts
        reasons.append(f"+{pts} critical terms {hits[:3]}")
    high, pts = UP["high_terms"]
    hits = [k for k in high if k in t]
    if hits:
        score += pts
        reasons.append(f"+{pts} elevated terms {hits[:3]}")
    down, pts = DOWN
    hits = [k for k in down if k in t]
    if hits:
        score += pts
        reasons.append(f"{pts} mitigating terms {hits[:3]}")
    score = max(0, min(100, score))
    return score, band_from_score(score, thresholds), reasons


DEFAULT_THRESHOLDS = (30, 45, 65)  # (medium, high, critical) cut-offs; overwritten by fit_thresholds


def band_from_score(score: int, thresholds=DEFAULT_THRESHOLDS) -> str:
    m, h, c = thresholds
    return "critical" if score >= c else "high" if score >= h else "medium" if score >= m else "low"


def fit_thresholds(scores, labels) -> tuple[int, int, int]:
    """Choose the 3 band cut-offs that maximise accuracy on the given (training) tickets."""
    import numpy as np
    sc = np.asarray(scores)
    y = np.array([SEVERITY_ORDER.index(l) for l in labels])
    best, best_acc = DEFAULT_THRESHOLDS, -1.0
    grid = range(10, 100, 3)
    for m in grid:
        for h in grid:
            if h <= m:
                continue
            for c in grid:
                if c <= h:
                    continue
                pred = (sc >= m).astype(int) + (sc >= h) + (sc >= c)
                acc = (pred == y).mean()
                if acc > best_acc:
                    best, best_acc = (m, h, c), acc
    return best


# ----------------------------------------------------------------------------- playbook ("knowledge" tool)

PLAYBOOK = {
    "inverter_thermal": dict(team="Solar O&M - Electrical", root_cause="Cooling failure (fan/filter), dust build-up, or ambient overload causing thermal derating.",
                             action="Inspect fans and filters, check ambient/heatsink temperature trend, clean filters; escalate if trips repeat."),
    "gearbox_vibration": dict(team="Wind O&M - Mechanical", root_cause="Bearing or gear wear, lubrication degradation, or misalignment.",
                              action="Run vibration spectrum analysis and oil sample test; consider turbine shutdown if above alarm limit."),
    "inspection_compliance": dict(team="Compliance & Asset Management", root_cause="Missed schedule or incomplete sign-off in the compliance process.",
                                  action="Assign owner, schedule inspection, and update the compliance register before the due date."),
    "data_gap": dict(team="SCADA / IT-OT", root_cause="Communication link, gateway/RTU, or sensor failure interrupting data flow.",
                     action="Check network link and gateway status, restart the data logger, and back-fill missing data."),
    "pitch_system": dict(team="Wind O&M - Controls", root_cause="Pitch motor, encoder, controller communication, or backup battery fault.",
                         action="Read pitch fault codes, test battery and encoder, keep turbine paused until safe-feather is verified."),
    "grid_curtailment": dict(team="Grid Operations / Scheduling", root_cause="Instruction not propagated between grid operator, plant controller and dashboard.",
                             action="Verify instruction receipt and setpoint, reconcile dashboard vs SCADA, and inform the grid operator."),
    "transformer_fault": dict(team="Substation / HV Maintenance", root_cause="Winding overheating, insulation degradation, oil loss, or protection operation.",
                              action="Review DGA and temperature trend, isolate the unit if protection operated, and start an HV inspection."),
    "safety_hazard": dict(team="HSE + Site Manager", root_cause="Electrical fault, fluid leak, or physical hazard posing risk to personnel.",
                          action="Secure the area now, isolate the energy source, notify HSE, and file an incident report."),
}
SLA_HOURS = {"critical": 2, "high": 8, "medium": 48, "low": 120}

# ----------------------------------------------------------------------------- agent

CONF_THRESHOLD = 0.45  # below this, the ML category is not trusted on its own


class TriageAgent:
    """Deterministic tool-calling agent. It decides which tool to call next based on the
    previous tool's output (e.g. falls back to keyword rules when ML confidence is low,
    escalates to HSE when safety terms are present, flags human review when unsure)."""

    def __init__(self, model: Pipeline | None = None, thresholds=None):
        df = load_data()
        self.model = model or train_classifier(df)
        if thresholds is None:  # fit severity cut-offs on the labelled tickets (using true categories)
            sc = [severity_score(d, c)[0] for d, c in zip(df["description"], df["target_category"])]
            thresholds = fit_thresholds(sc, df["target_severity"])
        self.thresholds = thresholds
        self.tools = {
            "classify_category": self.classify_category,
            "keyword_fallback": self.keyword_fallback,
            "score_severity": self.score_severity,
            "lookup_playbook": self.lookup_playbook,
            "check_escalation": self.check_escalation,
        }

    # --- tools
    def classify_category(self, text, asset_type):
        x = preprocess(text, asset_type)
        probs = self.model.predict_proba([x])[0]
        order = probs.argsort()[::-1]
        classes = self.model.classes_
        return {
            "category": str(classes[order[0]]),
            "confidence": round(float(probs[order[0]]), 3),
            "top3": [(str(classes[i]), round(float(probs[i]), 3)) for i in order[:3]],
        }

    def keyword_fallback(self, text):
        cat, hits = keyword_category(text)
        return {"category": cat, "matched_keywords": hits}

    def score_severity(self, text, category):
        score, band, reasons = severity_score(text, category, self.thresholds)
        return {"score": score, "severity": band, "reasons": reasons}

    def lookup_playbook(self, category):
        return PLAYBOOK.get(category, {})

    def check_escalation(self, text, severity, category):
        t = text.lower()
        triggers = [k for k in KEYWORD_RULES["safety_hazard"][:6] if k in t]
        escalate = bool(triggers) or category == "safety_hazard"
        out_sev = "critical" if (escalate and severity in ("high", "medium") and triggers) else severity
        return {"escalate_to_hse": escalate, "triggers": triggers, "severity": out_sev}

    # --- orchestration
    def run(self, text: str, asset_type: str = "unknown") -> dict:
        text = (text or "").strip()
        asset_type = (asset_type or "unknown").lower().strip()
        trace: list[dict] = []
        if not text:
            return {"error": "Empty ticket description.", "needs_human_review": True, "trace": trace}

        def call(tool, **kw):
            res = self.tools[tool](**kw)
            trace.append({"tool": tool, "input": {k: v for k, v in kw.items() if k != "text"}, "output": res})
            return res

        # step 1: ML category
        c = call("classify_category", text=text, asset_type=asset_type)
        category, conf, source = c["category"], c["confidence"], "ml"
        # step 2 (decision): low confidence -> try keyword rules
        review = False
        if conf < CONF_THRESHOLD:
            k = call("keyword_fallback", text=text)
            if k["category"]:
                category, source = k["category"], "keyword_fallback"
            else:
                review = True  # neither method is confident -> human triage
                category, source = "unclassified", "none"
        # step 3: severity
        s = call("score_severity", text=text, category=category)
        if category == "unclassified":
            s["severity"] = "medium"  # safe default; a human decides the real priority
        severity = s["severity"]
        # step 4 (decision): safety escalation
        e = call("check_escalation", text=text, severity=severity, category=category)
        severity = e["severity"]
        # step 5: playbook
        p = call("lookup_playbook", category=category)

        team = p.get("team", "Duty Engineer (manual triage)")
        if e["escalate_to_hse"]:
            team = "HSE + " + team if not team.startswith("HSE") else team
        return {
            "category": category,
            "category_source": source,
            "category_confidence": conf,
            "severity": severity,
            "severity_score": s["score"],
            "probable_root_cause": p.get("root_cause", "Unknown - needs manual diagnosis."),
            "recommended_next_action": p.get("action", "Route to duty engineer for manual triage."),
            "route_to": team,
            "sla_response_hours": SLA_HOURS[severity],
            "escalate_to_hse": e["escalate_to_hse"],
            "needs_human_review": review or conf < 0.30,
            "trace": trace,
            "disclaimer": "Root cause is a *probable* cause from a playbook for the predicted category, not a diagnosis.",
        }
