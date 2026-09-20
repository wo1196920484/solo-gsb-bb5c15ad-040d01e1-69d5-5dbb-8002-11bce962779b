"""Unknown-isotope-cluster detection in centroid m/z peak lists.

Given nothing but centroid peaks (strictly increasing m/z, non-negative
finite intensities) and a charge range 1..10, candidate isotope clusters are
generated *without* any molecular formula: for a charge ``z`` the nominal
spacing between consecutive isotope peaks is

    step(z) = 1.003354835 / z

(the 13C-12C neutron mass difference per charge). A candidate rooted at peak
``i`` walks to the observed peak whose spacing from the current member is
within ``spacing_tolerance`` of one integer step (k = 1, normally) or two
integer steps (k = 2, i.e. exactly one missing isotope position; at most one
such gap per cluster). Every prefix of such a walk with at least
``min_peaks`` members is emitted as a candidate.

Intensities are base-peak normalised (base peak = 100). A candidate's score
is the sum of its member intensities minus 10 per missing position.
Candidates whose index spans (first member .. last member, inclusive) do not
overlap are chosen by weighted-interval dynamic programming so the total
score is maximal. Score ties resolve, in order, to: more members total, the
lexicographically smaller sequence of monoisotopic m/z values, then the
lexicographically smaller charge sequence. Peaks outside every chosen
cluster span are reported as unassigned.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass

from .enrichment import PeakListError
from .envelope import Peak

# 13C - 12C mass difference (Da); the nominal M+1 neutron spacing.
NEUTRON_MASS_STEP_DA = 1.003354835

# Accepted request bounds (mirrored by the endpoint's Pydantic model).
MIN_CHARGE = 1
MAX_CHARGE = 10
MAX_PEAKS = 200
MIN_MIN_PEAKS = 2

# Score penalty for each missing isotope position inside a candidate.
MISSING_PENALTY = 10.0


def validate_cluster_peaks(peaks: list[Peak]) -> None:
    """Validate centroid peaks for cluster detection.

    Requires a non-empty list of peaks with strictly increasing finite m/z
    and non-negative finite intensities, at least one of them positive (the
    base-peak normalisation is otherwise undefined). Failures raise
    :class:`PeakListError` (the same ``invalid_peaks`` error used by the
    other observed-peak endpoints).
    """
    if not peaks:
        raise PeakListError("Peak list must contain at least one peak")
    for i, (mz, intensity) in enumerate(peaks):
        if not math.isfinite(mz):
            raise PeakListError("mz must be a finite number", i)
        if not math.isfinite(intensity):
            raise PeakListError("intensity must be a finite number", i)
        if intensity < 0.0:
            raise PeakListError(
                f"Peak intensity must be non-negative; got {intensity}", i
            )
    for i in range(1, len(peaks)):
        prev, cur = peaks[i - 1][0], peaks[i][0]
        if cur == prev:
            raise PeakListError(
                f"Duplicate peak m/z {cur}; m/z values must be strictly increasing",
                i,
            )
        if cur < prev:
            raise PeakListError(
                f"Peaks must be sorted by strictly increasing m/z; "
                f"m/z {cur} at index {i} follows {prev}",
                i,
            )
    if max(intensity for _, intensity in peaks) <= 0.0:
        raise PeakListError("At least one peak must have a positive intensity")


@dataclass(frozen=True)
class _Candidate:
    """One candidate cluster: every member lies inside [start, end]."""

    charge: int
    start: int  # index of the monoisotopic (first) member
    end: int  # index of the last member
    member_count: int
    missing_positions: tuple[int, ...]  # 1-based nominal isotope positions
    score: float


@dataclass(frozen=True)
class DetectedCluster:
    """One selected cluster as returned to the caller."""

    charge: int
    monoisotopic_mz: float
    score: float
    member_indices: tuple[int, ...]
    missing_positions: tuple[int, ...]


@dataclass(frozen=True)
class UnassignedPeak:
    """A peak covered by no selected cluster span."""

    index: int
    mz: float
    intensity: float


@dataclass
class ClusterDetectionResult:
    clusters: list[DetectedCluster]
    unassigned_peaks: list[UnassignedPeak]
    candidate_count: int


def _step_match(
    mzs: list[float], cur: int, k: int, step: float, tolerance: float
) -> int | None:
    """Index of the best peak ``k`` nominal steps after ``mzs[cur]``.

    The match must be strictly after ``cur`` and satisfy
    ``|mz[j] - (mzs[cur] + k*step)| <= tolerance``; among within-tolerance
    peaks the one nearest the ideal position wins, ties going to the lower
    m/z. A sorted-array neighbour of the ideal position is the global
    nearest, so this is a logarithmic lookup.
    """
    center = mzs[cur] + k * step
    j = bisect_left(mzs, center, lo=cur + 1)
    best: int | None = None
    for cand in (j - 1, j):
        if cand <= cur or cand >= len(mzs):
            continue
        error = abs(mzs[cand] - center)
        if error <= tolerance and (
            best is None or error < abs(mzs[best] - center)
        ):
            best = cand
    return best


def _walk_chain(
    mzs: list[float], start: int, step: float, tolerance: float
) -> tuple[list[int], list[int], list[int]]:
    """Greedy maximal cluster walk rooted at ``start``.

    Each extension first tries one nominal step; only when no within-
    tolerance 1-step peak exists does it allow a single 2-step gap (one
    missing isotope position per cluster). Returns the member indices, each
    member's nominal isotope position (0-based; the root is position 0) and
    the nominal positions of missing members (length 0 or 1).
    """
    members = [start]
    nominal = [0]
    missing: list[int] = []
    gap_used = False
    cur = start
    while True:
        nxt = _step_match(mzs, cur, 1, step, tolerance)
        if nxt is None and not gap_used:
            nxt = _step_match(mzs, cur, 2, step, tolerance)
            if nxt is not None:
                missing.append(nominal[-1] + 1)
                nominal.append(nominal[-1] + 2)
                gap_used = True
        if nxt is None:
            break
        if len(nominal) == len(members):
            # 1-step extension (the 2-step branch already pushed its nominal).
            nominal.append(nominal[-1] + 1)
        members.append(nxt)
        cur = nxt
    return members, nominal, missing


def _build_candidates(
    mzs: list[float],
    norm_intensities: list[float],
    charge_min: int,
    charge_max: int,
    tolerance: float,
    min_peaks: int,
) -> list[_Candidate]:
    """Generate every qualifying candidate prefix for every (root, charge).

    A prefix score sums its *member* intensities (peaks skipped over by a
    2-step gap sit inside the span but contribute nothing) and subtracts the
    missing-position penalty.
    """
    candidates: list[_Candidate] = []
    n = len(mzs)
    for charge in range(charge_min, charge_max + 1):
        step = NEUTRON_MASS_STEP_DA / charge
        for start in range(n):
            members, nominal, missing = _walk_chain(mzs, start, step, tolerance)
            member_sum = 0.0
            for length, member_idx in enumerate(members, start=1):
                member_sum += norm_intensities[member_idx]
                if length < min_peaks:
                    continue
                end = members[length - 1]
                prefix_missing = tuple(
                    pos for pos in missing if pos <= nominal[length - 1]
                )
                candidates.append(
                    _Candidate(
                        charge=charge,
                        start=start,
                        end=end,
                        member_count=length,
                        missing_positions=prefix_missing,
                        score=member_sum - MISSING_PENALTY * len(prefix_missing),
                    )
                )
    return candidates


class _State:
    """Best weighted-interval-scheduling solution on a DP prefix.

    Picks are candidates in ascending cluster order (ascending m/z). The
    tie-break keys mirror the requested ranking: total score, then member
    count, then lexicographic monoisotopic m/z sequence, then charges.
    """

    __slots__ = ("score", "count", "monos", "charges", "picks")

    def __init__(
        self,
        score: float = 0.0,
        count: int = 0,
        monos: tuple[float, ...] = (),
        charges: tuple[int, ...] = (),
        picks: tuple[_Candidate, ...] = (),
    ) -> None:
        self.score = score
        self.count = count
        self.monos = monos
        self.charges = charges
        self.picks = picks


def _prefer(candidate_state: _State, current: _State) -> bool:
    """Whether ``candidate_state`` beats ``current`` under the ranking rules."""
    if candidate_state.score != current.score:
        return candidate_state.score > current.score
    if candidate_state.count != current.count:
        return candidate_state.count > current.count
    if candidate_state.monos != current.monos:
        return candidate_state.monos < current.monos
    return candidate_state.charges < current.charges


def _select_candidates(candidates: list[_Candidate], mzs: list[float]) -> _State:
    """Weighted interval DP: maximum total score with non-overlapping spans."""
    n = len(mzs)
    ending: list[list[_Candidate]] = [[] for _ in range(n)]
    for cand in candidates:
        ending[cand.end].append(cand)

    dp: list[_State] = []
    for end in range(n):
        # Skip every candidate ending here (and hence peak ``end`` itself).
        best = dp[end - 1] if end > 0 else _State()
        for cand in ending[end]:
            prev = dp[cand.start - 1] if cand.start > 0 else _State()
            trial = _State(
                score=prev.score + cand.score,
                count=prev.count + cand.member_count,
                monos=prev.monos + (mzs[cand.start],),
                charges=prev.charges + (cand.charge,),
                picks=prev.picks + (cand,),
            )
            if _prefer(trial, best):
                best = trial
        dp.append(best)
    return dp[-1]


def _describe_members(
    mzs: list[float],
    start: int,
    member_count: int,
    step: float,
    tolerance: float,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Re-walk a selected candidate and slice its first ``member_count``
    members; the walk itself reports the missing nominal positions."""
    members, nominal, missing = _walk_chain(mzs, start, step, tolerance)
    members = tuple(members[:member_count])
    prefix_missing = tuple(pos for pos in missing if pos <= nominal[member_count - 1])
    return members, prefix_missing


def detect_isotope_clusters(
    peaks: list[Peak],
    charge_min: int,
    charge_max: int,
    spacing_tolerance: float,
    min_peaks: int,
) -> ClusterDetectionResult:
    """Detect non-overlapping unknown isotope clusters.

    ``peaks`` must already pass :func:`validate_cluster_peaks`; ``charge_min``
    and ``charge_max`` lie in 1..10 with ``charge_min <= charge_max``.
    """
    mzs = [mz for mz, _ in peaks]
    base = max(intensity for _, intensity in peaks)
    norm_intensities = [100.0 * intensity / base for _, intensity in peaks]

    candidates = _build_candidates(
        mzs, norm_intensities, charge_min, charge_max, spacing_tolerance, min_peaks
    )
    state = _select_candidates(candidates, mzs)

    covered = [False] * len(peaks)
    clusters: list[DetectedCluster] = []
    for cand in state.picks:
        step = NEUTRON_MASS_STEP_DA / cand.charge
        member_indices, missing = _describe_members(
            mzs, cand.start, cand.member_count, step, spacing_tolerance
        )
        for idx in range(cand.start, cand.end + 1):
            covered[idx] = True
        clusters.append(
            DetectedCluster(
                charge=cand.charge,
                monoisotopic_mz=mzs[cand.start],
                score=cand.score,
                member_indices=member_indices,
                missing_positions=missing,
            )
        )

    unassigned = [
        UnassignedPeak(index=idx, mz=mzs[idx], intensity=peaks[idx][1])
        for idx in range(len(peaks))
        if not covered[idx]
    ]
    return ClusterDetectionResult(
        clusters=clusters,
        unassigned_peaks=unassigned,
        candidate_count=len(candidates),
    )
