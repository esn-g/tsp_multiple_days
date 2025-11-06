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

import json

_TIME_SLOT_PATTERN = re.compile(r"^(?P<start>\d{1,2}:\d{2})\s*-\s*(?P<end>\d{1,2}:\d{2})$")


@dataclass
class TimeWindow:
    """Represents the time interval a job may start.

    The start and end attributes are stored as ``datetime`` instances.  Using
    concrete datetimes (instead of just hour/minute pairs) simplifies further
    calculations down the road because the solver typically expects absolute
    times.  When the job does not provide enough information to build a
    datetime, the fields are set to ``None`` and callers can fill in sensible
    defaults later.
    """

    start: Optional[datetime]
    end: Optional[datetime]

    @classmethod
    def from_job(
        cls, job: Jobclass, *, default_day_start: Optional[time] = None, default_day_end: Optional[time] = None
    ) -> "TimeWindow":
        """Create a :class:`TimeWindow` from a :class:`Jobclass` instance.

        Parameters
        ----------
        job:
            The job whose ``time_slot`` and ``date`` fields are used.
        default_day_start / default_day_end:
            Optional fallbacks when a job does not specify a time slot.  When
            provided the returned window will be anchored to the job's day
            (or ``todays_date`` fallback) using these defaults.
        """

        if job.time_slot:
            match = _TIME_SLOT_PATTERN.match(str(job.time_slot).strip())
            if match:
                start_str = match.group("start")
                end_str = match.group("end")
                try:
                    base_date = _normalise_date(job.date)
                    start_dt = datetime.combine(base_date, _parse_time(start_str)) if base_date else None
                    end_dt = datetime.combine(base_date, _parse_time(end_str)) if base_date else None
                    if start_dt and end_dt and end_dt < start_dt:
                        # Some schedules cross midnight; add 24h to the end in this scenario.
                        end_dt += timedelta(days=1)
                    return cls(start=start_dt, end=end_dt)
                except ValueError:
                    # Fall back to default handling below.
                    pass

        base_date = _normalise_date(job.date)
        start_dt = datetime.combine(base_date, default_day_start) if base_date and default_day_start else None
        end_dt = datetime.combine(base_date, default_day_end) if base_date and default_day_end else None
        return cls(start=start_dt, end=end_dt)


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

    def build_nodes(self, jobs: Iterable[Jobclass]) -> List[VRPNode]:
        nodes: List[VRPNode] = []
        for idx, job in enumerate(jobs, start=1):  # node_id starts at 1 because 0 is usually the depot
            time_window = TimeWindow.from_job(
                job,
                default_day_start=self._default_day_start,
                default_day_end=self._default_day_end,
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
        print(coords)
        return (get_osrm_time_matrix(coords))
        size = len(addresses)
        ret
        #return [[0 for _ in range(size)] for _ in range(size)]

    def build_worker_constraints(self, workers: Iterable[Workerclass]) -> List[WorkerConstraints]:
        return [WorkerConstraints.from_worker(worker) for worker in workers]

    def build_problem(
        self,
        *,
        jobs: Iterable[Jobclass],
        workers: Iterable[Workerclass],
        depot_address: str,
    ) -> VRPProblemDefinition:
        nodes = self.build_nodes(jobs)
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
