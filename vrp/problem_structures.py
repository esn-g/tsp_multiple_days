"""Utilities for representing Vehicle Routing Problem (VRP) data.

The module focuses on translating the domain specific ``Jobclass`` and
``Workerclass`` objects used elsewhere in the repository into a common
structure that can be consumed by the optimisation pipeline.  The goal is to
make it easy to gradually implement the remaining pieces (time windows, travel
matrix, worker constraints) without changing the surrounding code yet again.

The classes defined here are intentionally lightweight data containers.  They do
not implement heavy logic themselves – that will be added in follow up commits.
Instead they provide a consistent API and sensible defaults so other modules can
already start depending on the structure while the details are filled in.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Iterable, List, Optional, Sequence
import re

from classes.Jobclass import Jobclass
from classes.Workerclass import Workerclass
from pyhelpers.geocode import forward_geocode
from pyhelpers.osrm import get_osrm_time_matrix
from pyhelpers.day_indexing import day_to_index

import json

_TIME_SLOT_PATTERN = re.compile(r"^(?P<start>\d{1,2}:\d{2})\s*-\s*(?P<end>\d{1,2}:\d{2})$")

def _minutes_from_midnight(t: time) -> int:
    return t.hour * 60 + t.minute  # seconds ignored by design

def _parse_time_hhmm(s: str) -> time:
    h, m = map(int, s.split(":"))
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError("Hour/minute out of range")
    return time(hour=h, minute=m)

def _normalise_date(d) -> Optional[datetime.date]:
    # Your existing normaliser; stubbed minimal here:
    if d is None:
        return None
    if isinstance(d, datetime):
        return d.date()
    return d  # assume it's already a date

@dataclass
class TimeWindow:
    """Same-day window using minutes-from-midnight. `end = start + duration_min`."""
    day: Optional[int]          # Monday=0 ... Sunday=6
    start: Optional[int]        # minutes from midnight
    end: Optional[int]          # minutes from midnight (start + duration)

    @classmethod
    def from_job(
        cls,
        job: "Jobclass",
        *,
        start_day: Optional[str] =None,
        allow_diff: Optional[bool] =False,
        default_day_start: Optional[time] = None,
        default_day_end: Optional[time] = None,
        default_start_min: Optional[int] = None,
        duration_min: int = 5,
    ) -> "TimeWindow":
        base_date = _normalise_date(job.date)
        day_idx = day_to_index(target_day=job.day, start_day=start_day) if base_date else None
        start_min: Optional[int] = None

        # Prefer explicit time_slot → take only the START time (HH:MM) as the anchor
        if getattr(job, "time_slot", None):
            m = _TIME_SLOT_PATTERN.match(str(job.time_slot).strip())
            if m:
                start_min = _minutes_from_midnight(_parse_time_hhmm(m.group("start")))

        # Fallback to provided default start (already in minutes)
        if start_min is None and default_start_min is not None:
            start_min = int(default_start_min)

        # If still unknown or no date to anchor day, return empty window
        if start_min is None or day_idx is None:
            return cls(day=day_idx, start=None, end=None)
        
        end_min = start_min + int(duration_min)

        if allow_diff:
            day_start_min = (
                _minutes_from_midnight(default_day_start)
                if isinstance(default_day_start, time)
                else start_min
            )
            day_end_min = (
                _minutes_from_midnight(default_day_end)
                if isinstance(default_day_end, time)
                else end_min
            )

            # Clamp within same-day bounds
            day_start_min = max(0, min(day_start_min, 1439))
            day_end_min = max(day_start_min + 1, min(day_end_min, 1440))

            service_type = str(getattr(job, "service_type", "") or "").lower()
            if "hemstäd" in service_type:
                midday = 12 * 60
                morning_start = max(day_start_min, 6 * 60)
                morning_end = min(day_end_min, midday)
                afternoon_start = max(day_start_min, midday)
                afternoon_end = min(day_end_min, 15 * 60)

                if start_min < midday and morning_end > morning_start:
                    start_min, end_min = morning_start, morning_end
                elif afternoon_end > afternoon_start:
                    start_min, end_min = afternoon_start, afternoon_end
                else:
                    start_min, end_min = day_start_min, day_end_min
            else:
                start_min, end_min = day_start_min, day_end_min

            if end_min <= start_min:
                end_min = min(1440, start_min + int(duration_min))

        # Enforce same-day (no cross-midnight)
        if not (0 <= start_min <= 1439) or end_min > 1440:
            # invalid (e.g., start=23:58 with duration 5 → 1443)
            return cls(day=day_idx, start=None, end=None)

        return cls(day=day_idx, start=start_min, end=end_min)

    @classmethod
    def from_datetime(
        cls,
        dt: datetime,
        *,
        duration_min: int = 5,
    ) -> "TimeWindow":
        day_idx = dt.weekday()
        start_min = dt.hour * 60 + dt.minute
        end_min = start_min + int(duration_min)
        
        if end_min > 1440:
            return cls(day=day_idx, start=None, end=None)
        return cls(day=day_idx, start=start_min, end=end_min)

@dataclass
class VRPNode:
    """Representation of a job within the VRP model."""

    node_id: int
    job: Jobclass
    address: str
    service_duration: timedelta
    time_window: TimeWindow

    @property
    def label(self) -> str:
        """Human friendly label combining job id and description."""

        return f"{self.job.job_id}: {self.job.description}".strip()


@dataclass
class WorkerConstraints:
    """Captures the scheduling constraints derived from a worker."""

    worker: Workerclass
    can_drive: bool
    max_daily_minutes: Optional[int] = None
    max_weekly_minutes: Optional[int] = None

    @classmethod
    def from_worker(cls, worker: Workerclass) -> "WorkerConstraints":
        max_weekly_minutes: Optional[int] = None
        if worker.hours_per_week:
            try:
                max_weekly_minutes = int(float(worker.hours_per_week) * 60)
            except (TypeError, ValueError):
                max_weekly_minutes = None

        max_daily_minutes: Optional[int] = None
        if max_weekly_minutes and worker.days_per_week:
            try:
                days = float(worker.days_per_week)
                if days > 0:
                    max_daily_minutes = int(max_weekly_minutes / days)
            except (TypeError, ValueError):
                max_daily_minutes = None

        return cls(
            worker=worker,
            can_drive=bool(worker.has_driver_license),
            max_daily_minutes=max_daily_minutes,
            max_weekly_minutes=max_weekly_minutes,
        )


@dataclass
class VRPProblemDefinition:
    """Container gathering all data required by the solver."""

    nodes: Sequence[VRPNode]
    depot_address: str
    time_matrix: List[List[int]]
    worker_constraints: Sequence[WorkerConstraints]


class VRPProblemBuilder:
    """Assemble :class:`VRPProblemDefinition` objects from domain classes."""

    def __init__(
        self,
        *,
        default_day_start: time = time(hour=6),
        default_day_end: time = time(hour=18),
        
    ) -> None:
        self._default_day_start = default_day_start
        self._default_day_end = default_day_end

    def build_nodes(self, start_day, allow_diff, jobs: Iterable[Jobclass]) -> List[VRPNode]:
        nodes: List[VRPNode] = []
        for idx, job in enumerate(jobs, start=1):  # node_id starts at 1 because 0 is usually the depot
            time_window = TimeWindow.from_job(
                job,
                default_day_start=self._default_day_start,
                default_day_end=self._default_day_end,
                start_day = start_day,
                allow_diff= allow_diff
                
            )
            service_duration = _hours_to_timedelta(job.service_time)
            address = job.get_job_adress() if hasattr(job, "get_job_adress") else job.adress
            nodes.append(
                VRPNode(
                    node_id=idx,
                    job=job,
                    address=address,
                    service_duration=service_duration,
                    time_window=time_window,
                )
            )
        return nodes

    def build_time_matrix(self, addresses: Sequence[str]) -> List[List[int]]:
        """Return a placeholder time matrix for the provided addresses.

        The method currently returns a square matrix filled with zeros.  It acts
        as a scaffold so that callers can already depend on the structure while
        the actual travel time computation (distance API, historical data, ...)
        is implemented in future commits.
        """
        coords = [forward_geocode(adr) for adr in addresses]
        return (get_osrm_time_matrix(coords))

    def build_worker_constraints(self, workers: Iterable[Workerclass]) -> List[WorkerConstraints]:
        return [WorkerConstraints.from_worker(worker) for worker in workers]

    def build_problem(
        self,
        *,
        jobs: Iterable[Jobclass],
        workers: Iterable[Workerclass],
        depot_address: str,
        start_weekday: str,
        allow_diff: bool,
    ) -> VRPProblemDefinition:
        nodes = self.build_nodes(jobs=jobs,start_day=start_weekday,allow_diff=allow_diff)
        addresses = [depot_address] + [node.address for node in nodes]
        time_matrix = self.build_time_matrix(addresses)
        worker_constraints = self.build_worker_constraints(workers)
        return VRPProblemDefinition(
            nodes=nodes,
            depot_address=depot_address,
            time_matrix=time_matrix,
            worker_constraints=worker_constraints,
        )


def _normalise_date(value) -> Optional[datetime.date]:
    if isinstance(value, datetime):
        return value.date()
    try:
        # pandas.Timestamp is duck-typed to have ``to_pydatetime``
        to_datetime = getattr(value, "to_pydatetime", None)
        if callable(to_datetime):
            return to_datetime().date()
        if isinstance(value, str) and value:
            return datetime.fromisoformat(value).date()
    except (ValueError, TypeError):
        return None
    return value.date() if hasattr(value, "date") else None


def _parse_time(value: str) -> time:
    hour, minute = value.split(":")
    return time(hour=int(hour), minute=int(minute))


def _hours_to_timedelta(hours: Optional[float]) -> timedelta:
    try:
        return timedelta(hours=float(hours))
    except (TypeError, ValueError):
        return timedelta()
