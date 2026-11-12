from backend.triage import TriageAgent, preprocess, severity_score

agent = TriageAgent()


def test_preprocess_normalises_ids():
    assert "wtgid" in preprocess("Gearbox on WTG-12 noisy", "wind")


def test_gearbox_ticket():
    r = agent.run("Gearbox vibration has increased for turbine WTG-12 with abnormal noise", "wind")
    assert r["category"] == "gearbox_vibration"
    assert r["severity"] in ("high", "critical")
    assert "Mechanical" in r["route_to"]


def test_safety_escalates_to_hse():
    r = agent.run("Smoke seen coming from combiner box, area cordoned off", "solar")
    assert r["escalate_to_hse"] and r["severity"] == "critical" and r["route_to"].startswith("HSE")


def test_empty_ticket_is_rejected_gracefully():
    r = agent.run("   ", "solar")
    assert r["needs_human_review"] and "error" in r


def test_gibberish_flags_human_review():
    r = agent.run("hello there lunch menu", "unknown")
    assert r["needs_human_review"]


def test_severity_ordering():
    hi = severity_score("Repeated trips, smoke observed", "safety_hazard")[0]
    lo = severity_score("Minor routine inspection tag outdated", "inspection_compliance")[0]
    assert hi > lo
