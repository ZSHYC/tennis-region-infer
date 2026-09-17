from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from dataset import Event


@dataclass(frozen=True)
class ScoreRow:
    sample_id: str
    frame_number: int
    hit_score: float
    bounce_score: float


def scores_to_events(scores: Iterable[ScoreRow], thresholds: dict[str, float], nms_radius: int) -> list[Event]:
    if nms_radius < 0 or set(thresholds) != {"hit", "bounce"}:
        raise ValueError("thresholds 必须只含 hit/bounce，且 nms_radius 非负")
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in scores:
        grouped[(row.sample_id, "hit")].append((row.frame_number, row.hit_score))
        grouped[(row.sample_id, "bounce")].append((row.frame_number, row.bounce_score))
    events: list[Event] = []
    for (sample_id, event_type), candidates in grouped.items():
        kept: list[int] = []
        for frame_number, score in sorted(candidates, key=lambda item: (-item[1], item[0])):
            if score < thresholds[event_type] or any(abs(frame_number - frame) <= nms_radius for frame in kept):
                continue
            kept.append(frame_number)
            events.append(Event(sample_id, frame_number, event_type, float(score)))
    return sorted(events, key=lambda event: (event.sample_id, event.frame_number, event.event_type))


def _best(
    candidates: list[tuple[int, float, tuple[tuple[Event, Event, float], ...]]],
) -> tuple[int, float, tuple[tuple[Event, Event, float], ...]]:
    return max(candidates, key=lambda item: (item[0], -item[1]))


def _match_group(
    ground_truth: list[Event],
    predictions: list[Event],
    tolerance: float,
    distance: Callable[[Event, Event], float],
) -> list[tuple[Event, Event, float]]:
    rows = len(ground_truth) + 1
    columns = len(predictions) + 1
    empty: tuple[int, float, tuple[tuple[Event, Event, float], ...]] = (0, 0.0, ())
    table = [[empty for _column in range(columns)] for _row in range(rows)]
    for row in range(1, rows):
        for column in range(1, columns):
            candidates = [table[row - 1][column], table[row][column - 1]]
            error = distance(ground_truth[row - 1], predictions[column - 1])
            if error <= tolerance:
                previous = table[row - 1][column - 1]
                candidates.append(
                    (
                        previous[0] + 1,
                        previous[1] + error,
                        previous[2] + ((ground_truth[row - 1], predictions[column - 1], error),),
                    )
                )
            table[row][column] = _best(candidates)
    return list(table[-1][-1][2])


def match_events(
    ground_truth: Iterable[Event],
    predictions: Iterable[Event],
    tolerance: float,
    distance: Callable[[Event, Event], float] | None = None,
) -> list[tuple[Event, Event, float]]:
    if tolerance < 0:
        raise ValueError("tolerance 必须非负")
    distance = distance or (lambda gt, pred: float(abs(gt.frame_number - pred.frame_number)))
    gt_groups: dict[tuple[str, str], list[Event]] = defaultdict(list)
    pred_groups: dict[tuple[str, str], list[Event]] = defaultdict(list)
    for event in ground_truth:
        gt_groups[(event.sample_id, event.event_type)].append(event)
    for event in predictions:
        pred_groups[(event.sample_id, event.event_type)].append(event)
    matches = []
    for key in sorted(gt_groups.keys() | pred_groups.keys()):
        gt = sorted(gt_groups[key], key=lambda event: event.frame_number)
        pred = sorted(pred_groups[key], key=lambda event: event.frame_number)
        matches.extend(_match_group(gt, pred, tolerance, distance))
    return matches


def _report_counts(tp: int, fp: int, fn: int, minutes: float) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_per_minute": fp / minutes if minutes > 0 else 0.0,
    }


def _evaluate_tolerance(
    ground_truth: list[Event],
    predictions: list[Event],
    tolerance: float,
    distance: Callable[[Event, Event], float],
    minutes: float,
) -> dict[str, dict[str, float | int]]:
    report = {}
    totals = [0, 0, 0]
    for event_type in ("hit", "bounce"):
        gt = [event for event in ground_truth if event.event_type == event_type]
        pred = [event for event in predictions if event.event_type == event_type]
        tp = len(match_events(gt, pred, tolerance, distance))
        counts = (tp, len(pred) - tp, len(gt) - tp)
        report[event_type] = _report_counts(*counts, minutes)
        totals = [left + right for left, right in zip(totals, counts, strict=True)]
    report["overall"] = _report_counts(*totals, minutes)
    return report


def evaluate_all(
    ground_truth: Iterable[Event],
    predictions: Iterable[Event],
    frame_times: dict[str, np.ndarray],
    *,
    frame_tolerances: Iterable[int] = (1, 2, 5),
    ms_tolerances: Iterable[float] = (50.0, 100.0),
) -> dict[str, dict[str, dict[str, dict[str, float | int]]]]:
    gt = [event for event in ground_truth if event.event_type in {"hit", "bounce"}]
    pred = [event for event in predictions if event.event_type in {"hit", "bounce"}]
    minutes = 0.0
    for sample_id, values in frame_times.items():
        times = np.asarray(values, dtype=np.float64)
        if len(times) > 1 and (not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
            raise ValueError(f"frame_times 非法: {sample_id}")
        if len(times) > 1:
            minutes += float(times[-1] - times[0]) / 60.0

    def time_distance(left: Event, right: Event) -> float:
        times = frame_times[left.sample_id]
        if not (0 <= left.frame_number < len(times) and 0 <= right.frame_number < len(times)):
            raise ValueError("事件帧号超出 frame_times")
        return abs(float(times[left.frame_number]) - float(times[right.frame_number])) * 1000.0

    return {
        "frames": {
            str(int(tolerance)): _evaluate_tolerance(
                gt,
                pred,
                float(tolerance),
                lambda left, right: float(abs(left.frame_number - right.frame_number)),
                minutes,
            )
            for tolerance in frame_tolerances
        },
        "milliseconds": {
            f"{float(tolerance):g}": _evaluate_tolerance(gt, pred, float(tolerance), time_distance, minutes)
            for tolerance in ms_tolerances
        },
    }

