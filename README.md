# AI Agent for Maintenance Ticket Prioritization (Greenko Project Set 4)

Triage system for renewable-energy maintenance tickets (solar, wind, hydro, grid). Given free-text ticket + asset type it returns **category, severity (score + band), probable root cause, recommended next action, owning team, response SLA**, and a step-by-step **agent reasoning trace**.

**Intended user:** an O&M control-room engineer who receives many tickets and must decide what to look at first.
**Success =** correct category, severity within one band of the engineer's judgement, safety tickets never under-prioritised, and uncertain tickets handed to a human instead of guessed.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app:app --reload        # open http://localhost:8000
python -m backend.evaluate              # regenerates evaluation_results.json
python -c "import tests.test_triage"    # or: pip install pytest && pytest
```
API: `POST /api/triage {"text": "...", "asset_type": "wind"}` · `GET /api/samples` · `GET /api/evaluation` · `GET /health`

## Solution flow
```
ticket text + asset type
  -> preprocess (lowercase, normalise IDs like WTG-12, asset token)
  -> TOOL classify_category   TF-IDF(1-2 grams) + Logistic Regression  -> category + confidence
       |- confidence < 0.45 -> TOOL keyword_fallback (rules) ; still nothing -> "unclassified", human review
  -> TOOL score_severity      base score per category + keyword modifiers -> 0-100 -> band
  -> TOOL check_escalation    safety keywords -> escalate to HSE, raise severity to critical
  -> TOOL lookup_playbook     category -> probable root cause, next action, team
  -> route + SLA (critical 2h / high 8h / medium 48h / low 120h)
```
**What is an "agent" here:** a deterministic tool-calling workflow. It chooses the next tool from the previous tool's output (fallback, escalation, human-review branches) and records every call in the trace. **There is no LLM and no RAG** - the playbook is a small Python dictionary, not retrieval over documents. No API keys are needed.

## Data
`data/maintenance_tickets.csv` - 63 tickets: the 6 bundled rows (`source=bundled`) plus **57 synthetic tickets I wrote** (`source=synthetic_added`) to cover 8 categories and 4 severity levels. Severity labels are my own judgement, not from real engineers. Classes: inverter_thermal, gearbox_vibration, inspection_compliance, data_gap, pitch_system, grid_curtailment, transformer_fault, safety_hazard (7-8 tickets each).

## Model choices
* **Task:** multi-class text classification (category). Severity is an ordinal band, handled by a transparent scoring rule.
* **Model:** TF-IDF + Logistic Regression - fast, works on tiny data, gives probabilities (needed for the confidence-based fallback), and is explainable. A neural/transformer model would overfit 63 tickets and is hard to justify here.
* **Baselines:** majority class; keyword rules; for severity also a TF-IDF+LogReg severity classifier.
* **Key settings:** `ngram_range=(1,2)` (captures phrases like "oil temperature"); `C=100` (inverse regularisation: lower = smoother but under-confident on tiny data - C=10 made 54% of predictions fall below the confidence threshold; C 10/100/1000 gave similar accuracy 0.84-0.86); `CONF_THRESHOLD=0.45` (below this the ML answer is not trusted alone).
* **Severity score:** category base (e.g. safety 60, inspection 20) + keyword modifiers (+25 critical terms such as "smoke", "tripped"; +12 elevated terms such as "alarm"; -18 mitigating terms such as "minor", "routine"). The three band cut-offs are **learned from labelled tickets** (`fit_thresholds`).

## Evaluation (repeated stratified 5-fold CV x3, `python -m backend.evaluate`)
Tickets have no timestamps so no time-ordered split is possible. Models/cut-offs are fit on training folds only.

| Task | Model | Accuracy | Macro-F1 |
|---|---|---|---|
| Category | Majority baseline | 7.9% | 0.02 |
| Category | **TF-IDF + LogReg (ML only)** | **83.7%** (±9.8) | 0.81 |
| Category | Full agent (ML + keyword fallback) | 97.3% | 0.97 * |
| Severity | Majority baseline | 23.7% | 0.09 |
| Severity | TF-IDF + LogReg | 31.0% | 0.23 |
| Severity | **Full agent (rule scorer)** | **52.7%** (±13.8) | 0.50 |

Severity within one band of the label: **93.7%**. ML handled 69.8% of tickets without needing the fallback.

\* **Read this number carefully:** I wrote the keyword rules after looking at all 63 tickets, so 97% is *optimistic* (rules effectively saw the test data). The honest generalisation estimate for category is the ML-only 83.7%.

**Failure cases:** exact severity is only ~53% - labels are subjective and the scorer is crude, e.g. "Minor temperature alarm on inverter cleared by itself" scored too high, "Abnormal grinding noise from gearbox" scored too low (high vs critical). Errors are mostly one band off.

## Limitations / production-readiness gaps
* Tiny, mostly synthetic, self-labelled dataset; real tickets are longer, noisier and multilingual. Results will be lower on real data.
* Severity weights are hand-set assumptions; severity band cut-offs fitted on 63 tickets are unstable. Needs engineer-labelled history and ordinal models.
* Single label per ticket; real tickets can mix issues. Root cause is a playbook guess for the category, not a diagnosis. Asset IDs/criticality, weather, SCADA alarms and ticket history are not used.
* No auth, logging, monitoring, DB, or retraining pipeline. The model is retrained at startup.
* Next steps: collect real labelled tickets, add sentence embeddings, calibrate probabilities, RAG over O&M manuals for root-cause text, human feedback loop.

## Railway deployment
`Procfile` and `.python-version` are included. Steps: push this repo to GitHub -> railway.app -> New Project -> Deploy from GitHub repo -> after the build, Settings -> Networking -> Generate Domain -> open it and check `/health`. No environment variables are required. **Deployment link:** `<paste your Railway URL here>`

## AI-assistance disclosure
Built with AI assistance (Claude). I reviewed the pipeline, wrote/checked the evaluation, and can explain every component. Dataset additions were AI-drafted and then treated as synthetic.
