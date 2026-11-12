# Review cheat-sheet (Nov 13, 3:00 PM) - maps to the Case Study Presentation Guide

## 1-minute problem pitch (Part A.1)
"Control-room engineers get a stream of free-text maintenance tickets across solar, wind, hydro and grid assets. My system reads a ticket and returns category, severity, probable cause, next action and owning team, so safety and critical faults are looked at first. Success = right category, severity within one band, safety never under-prioritised, uncertain tickets sent to a human. Not done: real data, LLM, RAG, and Railway deploy unless you complete it (see below)."

## Flow to draw (A.2)  ticket -> preprocess -> classify (ML) -> [low confidence -> keyword fallback -> human review] -> severity score -> safety escalation -> playbook -> route + SLA

## Model (A.3)
* Classification task: category is a label from text. Severity = ordinal -> scoring rule.
* TF-IDF + LogReg vs majority baseline and keyword rules. Chosen: small data, probabilities, explainable.
* Params: ngram (1,2) = phrases; C=100 = less regularisation (C=10 was under-confident); confidence threshold 0.45.

## Evidence (B.4) - say the honest numbers
Category ML-only 83.7%; full agent 97.3% BUT rules written after seeing data -> optimistic. Severity exact 52.7%, within one band 93.7%, vs 23.7% majority. Split: stratified 5-fold CV (no timestamps, so no time split). Show one miss: T012 "Minor temperature alarm ... cleared by itself" scored too high.
Variance sign convention is not applicable to this case (not a forecasting problem) - say so if asked.

## GenAI honesty (B.5)
No LLM, no RAG. It is a tool-calling agent: tools = classify_category, keyword_fallback, score_severity, check_escalation, lookup_playbook; decisions = fallback / escalate / human review. Python calculates everything. Safeguards: empty input rejected, low confidence -> human review, no secrets needed.

## Demo (B.6) - test beforehand, keep a screenshot as backup
1. "Smoke observed from combiner box near block 4" (solar) -> critical, HSE escalation
2. "Gearbox vibration has increased for turbine WTG-12 with abnormal noise" (wind)
3. "hello lunch menu" -> unclassified, human review

## Repo walk (Part C)  README -> data/maintenance_tickets.csv -> backend/triage.py (preprocess, model, severity, agent) -> backend/evaluate.py -> backend/app.py -> frontend/index.html

## Likely questions - prepare one-line answers
* Why not an LLM? Small structured task, needs to be cheap, deterministic, auditable; an LLM could be added for explanations/RAG later.
* Why is 97% not the real accuracy? Rules were authored on the same tickets -> leakage; ML-only 84% is the fair figure.
* Why is severity weak? Subjective labels, hand-set weights, 63 tickets; mostly off by one band.
* Is it an agent? Yes in the workflow sense (tools + branching decisions), but not LLM-driven and no autonomous planning.
* Challenge: ML confidence was too low at C=10 so fallback did all the work (found by diagnosing the confidence distribution); fixed with C=100. Also fixed a bug where nonsense tickets were routed to HSE.
* Where did AI help? Drafting code and synthetic data; I verified via CV, tests and diagnosing failures.
