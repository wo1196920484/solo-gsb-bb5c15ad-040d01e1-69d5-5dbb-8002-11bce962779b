"""Tests for unknown-isotope-cluster detection.

Pure-module tests (no FastAPI dependency) exercise the candidate
generation, scoring and weighted-interval selection; API tests exercise the
``POST /api/v1/isotope-clusters`` endpoint via the FastAPI test client.
"""

import pytest

from app.isotope_clusters import (
    MISSING_PENALTY,
    NEUTRON_MASS_STEP_DA,
    PeakListError,
    _Candidate,
    _select_candidates,
    detect_isotope_clusters,
    validate_cluster_peaks,
)

STEP = NEUTRON_MASS_STEP_DA
TOL = 0.005


def _zchain(base_mz, n, charge=1, intensities=None, start_intensity=100.0):
    """``n`` perfectly spaced peaks at ``base_mz + k * 1.003354835/charge``."""
    step = STEP / charge
    if intensities is None:
        intensities = [max(start_intensity - k * 20.0, 1.0) for k in range(n)]
    return [
        (base_mz + k * step, intensities[k])
        for k in range(n)
    ]


# --- pure module ------------------------------------------------------------

def test_perfect_z1_cluster_detected():
    peaks = _zchain(100.0, 4, charge=1)
    result = detect_isotope_clusters(peaks, 1, 1, TOL, min_peaks=3)
    assert len(result.clusters) == 1
    cluster = result.clusters[0]
    assert cluster.charge == 1
    assert cluster.monoisotopic_mz == 100.0
    assert cluster.member_indices == (0, 1, 2, 3)
    assert cluster.missing_positions == ()
    assert cluster.score == pytest.approx(100.0 + 80.0 + 60.0 + 40.0)
    assert result.unassigned_peaks == []


def test_z2_cluster_uses_half_neutron_spacing():
    peaks = _zchain(50.0, 3, charge=2)
    result = detect_isotope_clusters(peaks, 1, 2, TOL, min_peaks=3)
    assert len(result.clusters) == 1
    assert result.clusters[0].charge == 2
    assert result.clusters[0].member_indices == (0, 1, 2)
    # The same peaks are not a z=1 step chain, so no z=1 candidate exists.
    assert result.candidate_count >= 1


def test_single_missing_member_reports_missing_position():
    # M+1 is absent: members at nominal positions 0, 2, 3.
    peaks = [
        (200.0, 100.0),
        (200.0 + 2 * STEP, 60.0),
        (200.0 + 3 * STEP, 20.0),
    ]
    result = detect_isotope_clusters(peaks, 1, 1, TOL, min_peaks=3)
    assert len(result.clusters) == 1
    cluster = result.clusters[0]
    assert cluster.member_indices == (0, 1, 2)
    assert cluster.missing_positions == (1,)
    assert cluster.score == pytest.approx(180.0 - MISSING_PENALTY)


def test_two_missing_gaps_stop_the_chain():
    # Gaps after positions 0 and 3; the second gap is not allowed, so the
    # walk ends at the +3*step member.
    peaks = [
        (100.0, 100.0),
        (100.0 + 2 * STEP, 60.0),
        (100.0 + 3 * STEP, 20.0),
        (100.0 + 5 * STEP, 10.0),
    ]
    result = detect_isotope_clusters(peaks, 1, 1, TOL, min_peaks=3)
    # No candidate can contain all four peaks; the first three form one
    # cluster and the last peak is unassigned (a 2-peak chain starting at
    # index 2 is below min_peaks=3).
    assert len(result.clusters) == 1
    assert result.clusters[0].member_indices == (0, 1, 2)
    assert result.clusters[0].missing_positions == (1,)
    assert [p.index for p in result.unassigned_peaks] == [3]


def test_intensities_normalised_to_base_peak():
    peaks = _zchain(100.0, 3, charge=1, intensities=[500.0, 300.0, 100.0])
    result = detect_isotope_clusters(peaks, 1, 1, TOL, min_peaks=3)
    # Base peak becomes 100, so the member sum is 100 + 60 + 20.
    assert result.clusters[0].score == pytest.approx(180.0)


def test_two_disjoint_clusters_both_selected():
    peaks = _zchain(100.0, 3, charge=1, start_intensity=100.0)
    peaks += _zchain(300.0, 3, charge=2, start_intensity=100.0)
    result = detect_isotope_clusters(peaks, 1, 2, TOL, min_peaks=3)
    assert len(result.clusters) == 2
    first, second = result.clusters
    assert first.monoisotopic_mz == 100.0 and first.charge == 1
    assert first.member_indices == (0, 1, 2)
    assert second.monoisotopic_mz == 300.0 and second.charge == 2
    assert second.member_indices == (3, 4, 5)
    assert result.unassigned_peaks == []


def test_no_qualifying_cluster_returns_empty():
    peaks = [(100.0, 10.0), (150.0, 10.0), (200.0, 10.0)]
    result = detect_isotope_clusters(peaks, 1, 10, TOL, min_peaks=3)
    assert result.clusters == []
    assert result.candidate_count == 0
    assert [p.index for p in result.unassigned_peaks] == [0, 1, 2]


def test_min_peaks_filters_short_prefixes():
    peaks = _zchain(100.0, 2)
    result = detect_isotope_clusters(peaks, 1, 1, TOL, min_peaks=3)
    assert result.clusters == []
    assert result.candidate_count == 0


def test_peak_inside_selected_span_is_covered():
    # A noise peak sits inside the cluster span but is not a member; span
    # non-overlap (first .. last member) covers it.
    peaks = [
        (100.0, 100.0),
        (100.0 + STEP, 60.0),
        (100.0 + STEP + 0.3, 90.0),
        (100.0 + 2 * STEP, 20.0),
    ]
    result = detect_isotope_clusters(peaks, 1, 1, TOL, min_peaks=3)
    assert len(result.clusters) == 1
    assert result.clusters[0].member_indices == (0, 1, 3)
    # Index 2 lies within the selected span [0, 3], hence not unassigned.
    assert result.unassigned_peaks == []


def test_peak_outside_selected_span_is_unassigned():
    peaks = [
        (50.0, 5.0),
        (100.0, 100.0),
        (100.0 + STEP, 60.0),
        (100.0 + 2 * STEP, 20.0),
        (400.0, 7.0),
    ]
    result = detect_isotope_clusters(peaks, 1, 1, TOL, min_peaks=3)
    assert len(result.clusters) == 1
    assert result.clusters[0].member_indices == (1, 2, 3)
    assert [p.index for p in result.unassigned_peaks] == [0, 4]
    assert result.unassigned_peaks[0].mz == 50.0
    assert result.unassigned_peaks[0].intensity == 5.0


def test_dp_prefers_non_overlapping_high_score_pair():
    # Two separated 3-peak chains score more jointly than any span crossing
    # them; peaks 0,1,2 then 4,5,6 with an unrelated peak 3 in between.
    peaks = _zchain(100.0, 3, charge=1, start_intensity=100.0)
    peaks.append((100.0 + 3 * STEP + 5.0, 5.0))
    peaks += _zchain(100.0 + 3 * STEP + 50.0, 3, charge=1, start_intensity=100.0)
    result = detect_isotope_clusters(peaks, 1, 1, TOL, min_peaks=3)
    assert len(result.clusters) == 2
    covered = {idx for c in result.clusters for idx in c.member_indices}
    assert covered == {0, 1, 2, 4, 5, 6}
    assert [p.index for p in result.unassigned_peaks] == [3]


def test_dp_tie_break_more_members():
    # Equal score with overlapping spans: the 2-member and 3-member
    # candidates share index 2, so only one can be chosen -> more win.
    cands = [
        _Candidate(1, 0, 2, 2, (), 50.0),
        _Candidate(1, 2, 4, 3, (), 50.0),
    ]
    mzs = [float(i) for i in range(5)]
    state = _select_candidates(cands, mzs)
    assert state.count == 3
    assert state.picks[0].start == 2


def test_dp_tie_break_mono_mz_sequence():
    # Same score and member count; the lexicographically smaller mono m/z
    # sequence wins.
    cands = [
        _Candidate(1, 2, 3, 2, (), 50.0),
        _Candidate(1, 0, 1, 2, (), 50.0),
    ]
    mzs = [float(i) for i in range(5)]
    state = _select_candidates(cands, mzs)
    assert state.monos == (0.0, 2.0)
    assert [p.start for p in state.picks] == [0, 2]


def test_dp_tie_break_charge_sequence():
    # Identical spans and scores at different charges: smaller charge wins.
    cands = [
        _Candidate(3, 0, 1, 2, (), 50.0),
        _Candidate(1, 0, 1, 2, (), 50.0),
    ]
    mzs = [float(i) for i in range(3)]
    state = _select_candidates(cands, mzs)
    assert state.charges == (1,)


def test_spacing_outside_tolerance_does_not_match():
    peaks = [
        (100.0, 100.0),
        (100.0 + STEP + 0.02, 60.0),
        (100.0 + 2 * STEP + 0.04, 20.0),
    ]
    result = detect_isotope_clusters(peaks, 1, 1, 0.005, min_peaks=3)
    assert result.clusters == []
    assert [p.index for p in result.unassigned_peaks] == [0, 1, 2]


def test_validate_cluster_peaks_failures():
    cases = [
        ([], None),
        ([(1.0, 1.0), (1.0, 2.0)], 1),
        ([(2.0, 1.0), (1.0, 2.0)], 1),
        ([(1.0, -0.1)], 0),
        ([(1.0, float("nan"))], 0),
        ([(float("inf"), 1.0)], 0),
        ([(1.0, 0.0), (2.0, 0.0)], None),
    ]
    for peaks, index in cases:
        with pytest.raises(PeakListError) as exc:
            validate_cluster_peaks(peaks)
        assert exc.value.index == index


def test_validate_cluster_peaks_accepts_zero_intensity_peaks():
    # Non-negative allows zero as long as some other peak is positive.
    validate_cluster_peaks([(1.0, 0.0), (2.0, 5.0)])


# --- API --------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.settings.db_path", str(tmp_path / "test.db"))
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        yield c


def _payload(**overrides):
    peaks = [
        {"mz": 100.0 + k * STEP, "intensity": 100.0 - 20 * k}
        for k in range(4)
    ]
    payload = {
        "peaks": peaks,
        "charge_min": 1,
        "charge_max": 3,
        "spacing_tolerance": 0.005,
        "min_peaks": 3,
    }
    payload.update(overrides)
    return payload


def test_isotope_clusters_endpoint_detects_cluster(client):
    resp = client.post("/api/v1/isotope-clusters", json=_payload())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["charge_min"] == 1
    assert body["charge_max"] == 3
    assert body["spacing_tolerance"] == 0.005
    assert body["min_peaks"] == 3
    assert body["peak_count"] == 4
    assert body["candidate_count"] >= 1
    assert body["cluster_count"] == 1
    cluster = body["clusters"][0]
    assert cluster["charge"] == 1
    assert cluster["monoisotopic_mz"] == 100.0
    assert cluster["member_indices"] == [0, 1, 2, 3]
    assert cluster["missing_positions"] == []
    assert cluster["score"] == pytest.approx(100.0 + 80.0 + 60.0 + 40.0)
    assert body["unassigned_peaks"] == []


def test_isotope_clusters_endpoint_missing_member(client):
    peaks = [
        {"mz": 200.0, "intensity": 100.0},
        {"mz": 200.0 + 2 * STEP, "intensity": 60.0},
        {"mz": 200.0 + 3 * STEP, "intensity": 20.0},
    ]
    resp = client.post(
        "/api/v1/isotope-clusters",
        json=_payload(peaks=peaks, charge_min=1, charge_max=1),
    )
    assert resp.status_code == 200, resp.text
    cluster = resp.json()["clusters"][0]
    assert cluster["missing_positions"] == [1]
    assert cluster["member_indices"] == [0, 1, 2]


def test_isotope_clusters_endpoint_empty_clusters(client):
    payload = _payload(
        peaks=[
            {"mz": 100.0, "intensity": 1.0},
            {"mz": 150.0, "intensity": 1.0},
            {"mz": 200.0, "intensity": 1.0},
        ]
    )
    resp = client.post("/api/v1/isotope-clusters", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["clusters"] == []
    assert body["cluster_count"] == 0
    assert body["candidate_count"] == 0
    assert [p["index"] for p in body["unassigned_peaks"]] == [0, 1, 2]


def test_isotope_clusters_endpoint_unassigned_echo(client):
    payload = _payload(
        peaks=[
            {"mz": 50.0, "intensity": 5.0},
            {"mz": 100.0, "intensity": 100.0},
            {"mz": 100.0 + STEP, "intensity": 60.0},
            {"mz": 100.0 + 2 * STEP, "intensity": 20.0},
        ]
    )
    resp = client.post("/api/v1/isotope-clusters", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["clusters"][0]["member_indices"] == [1, 2, 3]
    assert body["unassigned_peaks"] == [
        {"index": 0, "mz": 50.0, "intensity": 5.0}
    ]


def test_isotope_clusters_endpoint_unsorted_peaks_422(client):
    payload = _payload(
        peaks=[
            {"mz": 101.0, "intensity": 1.0},
            {"mz": 100.0, "intensity": 1.0},
        ]
    )
    resp = client.post("/api/v1/isotope-clusters", json=payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_peaks"
    assert detail["index"] == 1


def test_isotope_clusters_endpoint_duplicate_mz_422(client):
    payload = _payload(
        peaks=[
            {"mz": 100.0, "intensity": 1.0},
            {"mz": 100.0, "intensity": 2.0},
        ]
    )
    resp = client.post("/api/v1/isotope-clusters", json=payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_peaks"
    assert detail["index"] == 1


def test_isotope_clusters_endpoint_negative_intensity_422(client):
    resp = client.post(
        "/api/v1/isotope-clusters",
        json=_payload(peaks=[{"mz": 100.0, "intensity": -1.0}]),
    )
    # Negative intensity is rejected by the model (standard 422 detail list).
    assert resp.status_code == 422


def test_isotope_clusters_endpoint_nonfinite_422(client):
    # httpx refuses to encode NaN/Infinity itself, so send raw JSON bodies.
    for body in (
        '{"peaks":[{"mz":100.0,"intensity":NaN}],"charge_min":1,'
        '"charge_max":1,"spacing_tolerance":0.005,"min_peaks":2}',
        '{"peaks":[{"mz":Infinity,"intensity":1.0}],"charge_min":1,'
        '"charge_max":1,"spacing_tolerance":0.005,"min_peaks":2}',
    ):
        resp = client.post(
            "/api/v1/isotope-clusters",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422


def test_isotope_clusters_endpoint_all_zero_intensities_422(client):
    payload = _payload(
        peaks=[
            {"mz": 100.0, "intensity": 0.0},
            {"mz": 101.0, "intensity": 0.0},
        ]
    )
    resp = client.post("/api/v1/isotope-clusters", json=payload)
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "invalid_peaks"


def test_isotope_clusters_endpoint_parameter_bounds_422(client):
    good = _payload()

    resp = client.post(
        "/api/v1/isotope-clusters", json={**good, "charge_min": 0}
    )
    assert resp.status_code == 422
    resp = client.post(
        "/api/v1/isotope-clusters", json={**good, "charge_max": 11}
    )
    assert resp.status_code == 422
    resp = client.post(
        "/api/v1/isotope-clusters",
        json={**good, "charge_min": 3, "charge_max": 1},
    )
    assert resp.status_code == 422
    resp = client.post(
        "/api/v1/isotope-clusters", json={**good, "spacing_tolerance": 0.0}
    )
    assert resp.status_code == 422
    resp = client.post(
        "/api/v1/isotope-clusters", json={**good, "min_peaks": 1}
    )
    assert resp.status_code == 422
    resp = client.post(
        "/api/v1/isotope-clusters", json={**good, "min_peaks": 201}
    )
    assert resp.status_code == 422


def test_isotope_clusters_endpoint_peak_count_bounds_422(client):
    resp = client.post(
        "/api/v1/isotope-clusters", json=_payload(peaks=[])
    )
    assert resp.status_code == 422
    peaks = [{"mz": float(i), "intensity": 1.0} for i in range(201)]
    resp = client.post(
        "/api/v1/isotope-clusters", json=_payload(peaks=peaks)
    )
    assert resp.status_code == 422


def test_isotope_clusters_endpoint_missing_fields_422(client):
    resp = client.post("/api/v1/isotope-clusters", json={"peaks": []})
    assert resp.status_code == 422
    resp = client.post("/api/v1/isotope-clusters", json={})
    assert resp.status_code == 422
