"""Review resolver: eligibility limits, vote features, promotion only above the threshold."""
import review_resolver as rr


def _plate(text="TN09BR6934", reasons=("CONFLICTING_READINGS",), weakest=0.97, corrected=False, valid=True):
    return {"status": "REVIEW", "text": text, "final_text": "", "weakest": weakest, "avg": 0.99,
            "decision": {"status": "REVIEW", "reasons": list(reasons), "corrected": corrected,
                         "final_text": "", "validation": {"is_valid_format": valid}}}


class FakeModel:
    def __init__(self, p): self.p = p
    def predict_proba(self, x):
        import numpy as np
        return np.array([[1 - self.p, self.p]])


def _resolver(p, thr=0.9):
    r = rr.ReviewResolver(FakeModel(p), thr)
    r.evidence = lambda plate, attempts, views: {}
    return r


def test_eligibility_hard_limits():
    assert rr.eligible(_plate())
    assert not rr.eligible(_plate(corrected=True))                         # character swaps
    assert not rr.eligible(_plate(valid=False))
    assert not rr.eligible(_plate(weakest=0.1))
    assert not rr.eligible(_plate(reasons=("UNUSUAL_PLATE_SHAPE",)))


def test_vote_counts_rivals():
    att = [{"raw": "TN09BR6934", "avg": 0.99, "variant": "original"}] * 5 + \
          [{"raw": "DN09BR6934", "avg": 0.98, "variant": "adaptive_threshold"}] * 3
    v = rr.vote(att, "TN09BR6934")
    assert v["v_support"] == 5 and v["v_rival_support"] == 3 and v["v_rival_dist"] == 1
    assert v["v_rival_warn"] == 1.0                     # DN is a legacy state code
    assert abs(v["v_share"] - 5 / 8) < 1e-9


def test_parseq_agreement():
    e = rr.parseq_evidence([{"raw": "TN09BR6934", "avg": 0.9}, {"raw": "TN09BR693", "avg": 0.5}], "TN09BR6934")
    assert e["p_exact"] == 1 and e["p_any_exact"] == 1 and e["p_best_sim"] == 1


def test_promotes_only_above_threshold_and_never_changes_text():
    p = _resolver(0.95).apply(_plate(), [], [])
    assert p["status"] == "SUCCESS" and p["final_text"] == "TN09BR6934" and p["text"] == "TN09BR6934"
    assert p["decision"]["resolved_by"] == rr.RESOLVED_BY
    assert p["decision"]["resolved_reasons"] == ["CONFLICTING_READINGS"]
    q = _resolver(0.5).apply(_plate(), [], [])
    assert q["status"] == "REVIEW" and q["final_text"] == "" and q["resolver_probability"] == 0.5
