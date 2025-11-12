"""Interactive Streamlit application for multi-day scheduling optimisation.

The UI exposes two complementary modes:

* **Schedule review** – mirror of the existing ``setup_problem`` workflow where
  the user selects job and staff Excel files, chooses the scheduling horizon
  and runs the optimiser on the imported data.
* **Scenario builder** – allows users to create additional jobs and workers
  directly in the browser before running the optimisation.  This is useful for
  what-if analysis without having to edit the source spreadsheets.

Both modes present the original schedule as well as the optimised assignment on
worker/time axes so the impact of the optimisation can be inspected visually.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import altair as alt
import pydeck as pdk
import pandas as pd
import streamlit as st
import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


from classes.Jobclass import Jobclass
from classes.Workerclass import Workerclass
from pyhelpers.day_indexing import weekday_name_sv
from pyhelpers.geocode import MAPBOX_TOKEN as _GEOCODE_TOKEN, forward_geocode
from pyhelpers.osrm import get_osrm_time_matrix
from setup_problem import format_jobs, format_workers
from vrp.problem_structures import VRPProblemBuilder, VRPProblemDefinition
from vrp_multiple_days import solve_vrp_problem_definition


_MAPBOX_TOKEN = os.environ.get("MAPBOX_API_KEY") or os.environ.get("MAPBOX_TOKEN") or _GEOCODE_TOKEN
if _MAPBOX_TOKEN:
    pdk.settings.mapbox_api_key = _MAPBOX_TOKEN

_COLOR_PALETTE: List[Tuple[int, int, int]] = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]

_EMPLOYMENT_TERMS = [
    "(VT) – Allmän visstidsanställning",
    "Full-time",
    "Part-time",
    "Hourly",
    "Seasonal",
    "Other",
]


_OPTIMISED_STATE_KEY = "optimised_schedule_state"




# ---------------------------------------------------------------------------
# Helper data structures
# ---------------------------------------------------------------------------

@dataclass
class ScheduleAssignment:
    """Representation of a solved job assignment for plotting purposes."""

    worker_name: str
    day_offset: int
    start_minutes: int
    end_minutes: int
    job: Jobclass


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


_DEFAULT_JOBS_PATH = Path(__file__).resolve().parents[1] / "Kund1" / "september_clean" / "september_arbete.xlsx"
_DEFAULT_STAFF_PATH = Path(__file__).resolve().parents[1] / "Kund1" / "anställda" / "September_staff_details.xlsx"


class SafeVRPProblemBuilder(VRPProblemBuilder):
    """Problem builder that gracefully falls back when travel data is missing."""

    def build_time_matrix(self, addresses: Sequence[str]) -> List[List[int]]:
        try:
            matrix = super().build_time_matrix(addresses)
        except Exception:
            matrix = None

        if not matrix:
            size = len(addresses)
            # Use a simple constant travel time to keep the solver feasible.
            return [[0 if i == j else 15 for j in range(size)] for i in range(size)]

        # ``get_osrm_time_matrix`` returns minutes already – ensure structure.
        if isinstance(matrix, list) and all(isinstance(row, list) for row in matrix):
            return matrix

        size = len(addresses)
        return [[0 if i == j else 15 for j in range(size)] for i in range(size)]


def _ensure_job_id(job: Jobclass, index: int) -> str:
    return str(job.job_id if getattr(job, "job_id", None) not in (None, "") else f"job-{index}")


def _parse_time_slot(job: Jobclass) -> Optional[tuple[datetime, datetime]]:
    if not job.date:
        return None

    job_date = pd.to_datetime(job.date).to_pydatetime().date()
    time_slot = getattr(job, "time_slot", None)
    if isinstance(time_slot, str) and "-" in time_slot:
        start_raw, end_raw = [part.strip() for part in time_slot.split("-", 1)]
        try:
            start_dt = datetime.combine(job_date, datetime.strptime(start_raw, "%H:%M").time())
            end_dt = datetime.combine(job_date, datetime.strptime(end_raw, "%H:%M").time())
            if end_dt <= start_dt:
                end_dt = start_dt + timedelta(hours=float(getattr(job, "service_time", 1) or 1))
            return start_dt, end_dt
        except ValueError:
            pass

    service_hours = float(getattr(job, "service_time", 0) or 0)
    if service_hours <= 0:
        return None

    default_start = datetime.combine(job_date, time(hour=8))
    default_end = default_start + timedelta(hours=service_hours)
    return default_start, default_end


def _jobs_dataframe(jobs: Sequence[Jobclass]) -> pd.DataFrame:
    rows = []
    for idx, job in enumerate(jobs):
        parsed = _parse_time_slot(job)
        if not parsed:
            continue
        start_dt, end_dt = parsed
        rows.append(
            {
                "job_id": _ensure_job_id(job, idx),
                "worker": getattr(job, "staff_name", "Unassigned") or "Unassigned",
                "start": start_dt,
                "end": end_dt,
                "address": getattr(job, "adress", ""),
                "service": getattr(job, "service_type", ""),
                "description": getattr(job, "description", ""),
            }
        )
    return pd.DataFrame(rows)


def _assignments_to_df(assignments: Iterable[ScheduleAssignment], base_date: date) -> pd.DataFrame:
    rows = []
    for idx, item in enumerate(assignments):
        current_date = base_date + timedelta(days=item.day_offset)
        start_dt = datetime.combine(current_date, time()) + timedelta(minutes=item.start_minutes)
        end_dt = datetime.combine(current_date, time()) + timedelta(minutes=item.end_minutes)
        rows.append(
            {
                "job_id": _ensure_job_id(item.job, idx),
                "worker": item.worker_name,
                "start": start_dt,
                "end": end_dt,
                "address": getattr(item.job, "adress", ""),
                "service": getattr(item.job, "service_type", ""),
                "description": getattr(item.job, "description", ""),
            }
        )
    return pd.DataFrame(rows)


def _format_event_label(row: pd.Series) -> str:
    description = str(row.get("description") or "").strip()
    start = row.get("start")
    end = row.get("end")
    window = ""
    if isinstance(start, datetime) and isinstance(end, datetime):
        window = f"{start.strftime('%H:%M')}–{end.strftime('%H:%M')}"
    worker = str(row.get("worker") or "").strip()
    parts = [part for part in (description, window, worker) if part]
    return "\n".join(parts)


@st.cache_data(show_spinner=False)
def _geocode_address(address: str) -> Optional[Tuple[float, float]]:
    if not address or not str(address).strip():
        return None
    try:
        return forward_geocode(address)
    except Exception:
        return None


def _worker_color(worker: str) -> Tuple[int, int, int]:
    if not worker:
        return (120, 120, 120)
    index = abs(hash(worker)) % len(_COLOR_PALETTE)
    return _COLOR_PALETTE[index]


def _prepare_map_artifacts(
    df: pd.DataFrame, *, depot_address: Optional[str] = None
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    points: List[Dict[str, object]] = []
    lines: List[Dict[str, object]] = []

    depot_coords: Optional[Tuple[float, float]] = None
    if depot_address:
        depot_coords = _geocode_address(depot_address)
        if depot_coords:
            depot_lon, depot_lat = depot_coords
            points.append(
                {
                    "lon": depot_lon,
                    "lat": depot_lat,
                    "worker": "Depot",
                    "date": "",
                    "job_id": "Depot",
                    "address": depot_address,
                    "label": "Depot",
                    "color": [40, 40, 40, 220],
                }
            )

    if df.empty:
        return points, lines

    working_df = df.sort_values(["worker", "start"]).copy()
    working_df["date_only"] = working_df["start"].dt.date

    for (worker, day), group in working_df.groupby(["worker", "date_only"], sort=True):
        previous_job: Optional[Tuple[float, float]] = None
        color = list(_worker_color(worker))
        for record in group.itertuples(index=False):
            coords = _geocode_address(getattr(record, "address", ""))
            if not coords:
                continue
            lon, lat = coords
            label = _format_event_label(pd.Series(record._asdict()))
            point = {
                "lon": lon,
                "lat": lat,
                "worker": worker,
                "date": str(day),
                "job_id": getattr(record, "job_id", ""),
                "address": getattr(record, "address", ""),
                "label": label.replace("\n", "<br/>") if label else "",
                "color": color + [200],
            }
            points.append(point)
            if previous_job:
                lines.append(
                    {
                        "source_lon": previous_job[0],
                        "source_lat": previous_job[1],
                        "target_lon": lon,
                        "target_lat": lat,
                        "color": color + [160],
                        "worker": worker,
                        "date": str(day),
                    }
                )
            elif depot_coords:
                lines.append(
                    {
                        "source_lon": depot_coords[0],
                        "source_lat": depot_coords[1],
                        "target_lon": lon,
                        "target_lat": lat,
                        "color": color + [160],
                        "worker": worker,
                        "date": str(day),
                    }
                )
            previous_job = (lon, lat)

        if previous_job and depot_coords:
            lines.append(
                {
                    "source_lon": previous_job[0],
                    "source_lat": previous_job[1],
                    "target_lon": depot_coords[0],
                    "target_lat": depot_coords[1],
                    "color": color + [160],
                    "worker": worker,
                    "date": str(day),
                }
            )

    return points, lines


def _map_chart(
    df: pd.DataFrame,
    title: str,
    *,
    key_prefix: str,
    depot_address: Optional[str] = None,
) -> None:
    points, lines = _prepare_map_artifacts(df, depot_address=depot_address)
    if not points:
        st.info(f"No geocoded locations available for '{title}'.")
        return

    avg_lon = sum(point["lon"] for point in points) / len(points)
    avg_lat = sum(point["lat"] for point in points) / len(points)

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=points,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=120,
        pickable=True,
    )

    line_layer = pdk.Layer(
        "LineLayer",
        data=lines,
        get_source_position="[source_lon, source_lat]",
        get_target_position="[target_lon, target_lat]",
        get_color="color",
        get_width=4,
        pickable=False,
    )

    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/streets-v11",
        initial_view_state=pdk.ViewState(latitude=avg_lat, longitude=avg_lon, zoom=10, pitch=0, bearing=0),
        layers=[line_layer, scatter_layer],
        tooltip={
            "html": "<b>{worker}</b><br/>{label}<br/><small>{address}</small>",
            "style": {"fontSize": "12px"},
        },
    )

    st.pydeck_chart(deck, width='stretch', key=f"map_{key_prefix}")


def _schedule_day_view(
    df: pd.DataFrame,
    title: str,
    *,
    key_prefix: str,
    depot_address: Optional[str] = None,
) -> None:
    if df.empty:
        st.info(f"No schedulable entries available for '{title}'.")
        return

    working_df = df.copy()
    working_df["date_only"] = working_df["start"].dt.date
    unique_days = sorted(day for day in working_df["date_only"].unique() if pd.notna(day))

    if not unique_days:
        st.info(f"No dated entries available for '{title}'.")
        return

    selected_day = st.selectbox(
        "Select day",
        options=unique_days,
        format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
        key=f"{key_prefix}_day",
    )

    day_df = working_df[working_df["date_only"] == selected_day]
    worker_options = sorted(day_df["worker"].dropna().unique())

    selected_workers = st.multiselect(
        "Select workers to display",
        options=worker_options,
        default=worker_options,
        key=f"{key_prefix}_workers",
    )

    if selected_workers:
        filtered = day_df[day_df["worker"].isin(selected_workers)].copy()
    else:
        filtered = day_df.iloc[0:0].copy()

    subtitle = f"{title} – {pd.Timestamp(selected_day).strftime('%Y-%m-%d')}"

    show_map = st.checkbox(
        "Show map",
        value=True,
        key=f"{key_prefix}_{selected_day}_show_map",
        help="Toggle to display the geographic route next to the schedule",
    )

    if show_map:
        chart_col, map_col = st.columns((3, 2))
        with chart_col:
            _timeline_chart(filtered, subtitle, key_prefix=f"{key_prefix}_{selected_day}")
        with map_col:
            _map_chart(
                filtered,
                subtitle,
                key_prefix=f"{key_prefix}_{selected_day}",
                depot_address=depot_address,
            )
    else:
        _timeline_chart(filtered, subtitle, key_prefix=f"{key_prefix}_{selected_day}")


def _calculate_travel_time_summary(
    df: pd.DataFrame, depot_address: Optional[str]
) -> Optional[Dict[str, object]]:
    """Return travel time aggregates for the provided schedule dataframe.

    The dataframe is expected to contain ``worker``, ``start`` and ``address``
    columns.  All routes are assumed to start and end at ``depot_address`` on
    the same day.  Travel durations are summed following the chronological
    order of jobs per worker per day.
    """

    if df is None or df.empty or not depot_address:
        return None

    depot_coords = _geocode_address(depot_address)
    if not depot_coords:
        return None

    working_df = df.copy()
    if "start" not in working_df.columns or "worker" not in working_df.columns:
        return None

    if not pd.api.types.is_datetime64_any_dtype(working_df["start"]):
        working_df["start"] = pd.to_datetime(working_df["start"], errors="coerce")

    working_df = working_df.dropna(subset=["start", "worker", "address"])
    if working_df.empty:
        return None

    working_df["date_only"] = working_df["start"].dt.date
    matrix_cache: Dict[Tuple[Tuple[float, float], ...], Optional[List[List[float]]]] = {}

    per_worker_rows: List[Dict[str, object]] = []
    per_day_totals: Dict[date, float] = defaultdict(float)
    overall_total = 0.0

    grouped = working_df.groupby(["worker", "date_only"], sort=True)

    for (worker, day), group in grouped:
        if not worker:
            continue

        sorted_group = group.sort_values("start")

        route_coords: List[Tuple[float, float]] = [depot_coords]
        for record in sorted_group.itertuples(index=False):
            coords = _geocode_address(getattr(record, "address", ""))
            if not coords:
                continue
            route_coords.append(coords)
        route_coords.append(depot_coords)

        if len(route_coords) <= 2:
            continue

        coords_key = tuple(route_coords)
        matrix = matrix_cache.get(coords_key)
        if matrix is None and coords_key not in matrix_cache:
            try:
                matrix = get_osrm_time_matrix([[lon, lat] for lon, lat in route_coords])
            except Exception:
                matrix = None
            matrix_cache[coords_key] = matrix

        if not matrix:
            continue

        travel_minutes = 0.0
        for idx in range(len(route_coords) - 1):
            try:
                leg = matrix[idx][idx + 1]
            except (IndexError, TypeError):
                leg = None
            if leg is None:
                continue
            try:
                travel_minutes += float(leg)
            except (TypeError, ValueError):
                continue

        if travel_minutes <= 0:
            continue

        date_label = pd.Timestamp(day).strftime("%Y-%m-%d")
        per_worker_rows.append(
            {
                "Worker": worker,
                "Date": date_label,
                "Travel minutes": round(travel_minutes, 2),
            }
        )
        per_day_totals[day] += travel_minutes
        overall_total += travel_minutes

    if not per_worker_rows:
        return None

    per_worker_df = pd.DataFrame(per_worker_rows).sort_values(["Date", "Worker"]).reset_index(drop=True)
    per_day_df = (
        pd.DataFrame(
            [
                {
                    "Date": pd.Timestamp(day).strftime("%Y-%m-%d"),
                    "Total travel minutes": round(total, 2),
                }
                for day, total in sorted(per_day_totals.items())
            ]
        )
        if per_day_totals
        else pd.DataFrame(columns=["Date", "Total travel minutes"])
    )

    return {
        "per_worker": per_worker_df,
        "per_day": per_day_df,
        "overall_total": round(overall_total, 2),
    }


def _render_optimised_results(
    state: Optional[dict],
    *,
    depot_address: Optional[str],
    todays_date: date,
) -> None:
    if not state:
        return

    result_df = state.get("result_df")
    if result_df is None or result_df.empty:
        return

    st.subheader("Optimisation results")

    message = state.get("message")
    if message:
        st.success(message)

    generated_at = state.get("generated_at")
    if generated_at:
        st.caption(f"Solved at {generated_at}")

    stored_base = state.get("base_date")
    base_date: Optional[date]
    if isinstance(stored_base, date):
        base_date = stored_base
    elif isinstance(stored_base, str):
        try:
            base_date = datetime.fromisoformat(stored_base).date()
        except ValueError:
            base_date = None
    else:
        base_date = None

    if base_date and base_date != todays_date:
        st.info(
            "The displayed optimisation was generated for a different reference date "
            f"({base_date.isoformat()}). Rerun the solver to refresh the results."
        )

    show_result_table = st.checkbox(
        "Show optimised schedule table",
        value=False,
        key="show_result_table",
        help="Toggle to display the detailed table of optimised assignments.",
    )
    if show_result_table:
        st.dataframe(result_df, width='stretch')

    _schedule_day_view(
        result_df,
        "Optimised schedule",
        key_prefix="optimised",
        depot_address=depot_address,
    )


def _timeline_chart(df: pd.DataFrame, title: str, *, key_prefix: str) -> None:
    if df.empty:
        st.info(f"No schedulable entries available for '{title}'.")
        return

    df = df.copy()
    df["event_label"] = df.apply(_format_event_label, axis=1)

    worker_count = max(1, df["worker"].nunique())
    chart_width = max(400, 180 * worker_count)

    bars = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("worker:N", title="Worker", sort=list(df["worker"].unique())),
            y=alt.Y("start:T", title="Start time", scale=alt.Scale(reverse=True)),
            y2=alt.Y2("end:T", title="End time"),
            color=alt.Color("job_id:N", title="Job", legend=None),
            tooltip=[
                "job_id",
                "worker",
                alt.Tooltip("start:T", title="Start"),
                alt.Tooltip("end:T", title="End"),
                alt.Tooltip("description:N", title="Description"),
                "service",
                "address",
            ],
        )
    )

    text = (
        alt.Chart(df)
        .mark_text(
            align="left",
            baseline="top",
            dx=4,          # small left padding inside the rect
            dy=2,          # small top padding inside the rect
            fontSize=9,
            lineBreak="\n",     # ← this makes "\n" create a new line
            lineHeight=11,      # ← tweak spacing between lines
            color="black",
        )
        .encode(
            # <— key change: pin to the LEFT edge of the worker band
            x=alt.X("worker:N", sort=list(df["worker"].unique()), bandPosition=0),
            # y is the start of the block; with reverse scale this is the visual “top”
            y=alt.Y("start:T", scale=alt.Scale(reverse=True)),
            # your label already contains '\n' so this renders multi-line
            text=alt.Text("event_label:N"),
        )
    )

    chart = (
        alt.layer(bars, text)
        .properties(title=title, width=chart_width, height=520)
        .configure_axis(labelAngle=0)
    )

    st.altair_chart(chart, width='stretch', key=f"timeline_{key_prefix}")


def _load_jobs(file: Optional[bytes], selected_date: date, horizon: int) -> List[Jobclass]:
    if file is None:
        if _DEFAULT_JOBS_PATH.exists():
            df = pd.read_excel(_DEFAULT_JOBS_PATH, engine="openpyxl")
        else:
            return []
    else:
        df = pd.read_excel(file, engine="openpyxl")

    df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce").dt.date
    start = selected_date
    end = selected_date + timedelta(days=horizon)
    mask = (df["Datum"] >= start) & (df["Datum"] <= end)
    filtered = df.loc[mask].copy()
    return format_jobs(filtered)


def _load_workers(file: Optional[bytes]) -> List[Workerclass]:
    if file is None:
        if _DEFAULT_STAFF_PATH.exists():
            df = pd.read_excel(_DEFAULT_STAFF_PATH, engine="openpyxl")
        else:
            return []
    else:
        df = pd.read_excel(file, engine="openpyxl")
    return format_workers(df)


def _build_problem(jobs: Sequence[Jobclass], workers: Sequence[Workerclass], *, depot: str, start_day: str, allow_diff: bool) -> Optional[VRPProblemDefinition]:
    if not jobs or not workers:
        return None
    builder = SafeVRPProblemBuilder()
    return builder.build_problem(jobs=jobs, workers=workers, depot_address=depot, start_weekday=start_day, allow_diff=allow_diff)


def _collect_assignments(solution: Optional[dict]) -> List[ScheduleAssignment]:
    assignments: List[ScheduleAssignment] = []
    if not solution or not solution.get("assignments"):
        return assignments

    for entry in solution["assignments"]:
        worker = entry.get("worker")
        worker_name = getattr(worker, "name", None) if worker else None
        if not worker_name:
            worker_name = f"Vehicle {entry.get('vehicle_id', '-') }"
        job_node = entry.get("job_node")
        if not job_node:
            continue
        assignments.append(
            ScheduleAssignment(
                worker_name=worker_name,
                day_offset=int(entry.get("day", 0)),
                start_minutes=int(entry.get("start_min", 0)),
                end_minutes=int(entry.get("end_min", entry.get("start_min", 0))),
                job=job_node.job,
            )
        )
    return assignments


def _select_jobs_ui(jobs: Sequence[Jobclass]) -> List[Jobclass]:
    if not jobs:
        return []

    labels = [f"{_ensure_job_id(job, idx)} · {getattr(job, 'description', '')}" for idx, job in enumerate(jobs)]
    label_to_job = dict(zip(labels, jobs))
    selected_labels = st.multiselect("Select jobs to include", labels, default=labels)
    return [label_to_job[label] for label in selected_labels]


def _select_workers_ui(workers: Sequence[Workerclass]) -> List[Workerclass]:
    if not workers:
        return []

    labels = [getattr(worker, "name", f"Worker {idx}") or f"Worker {idx}" for idx, worker in enumerate(workers)]
    label_to_worker = dict(zip(labels, workers))
    selected_labels = st.multiselect("Select workers to include", labels, default=labels)
    return [label_to_worker[label] for label in selected_labels]


def _configure_vt_workers(workers: Sequence[Workerclass], *, key_prefix: str = "vt") -> None:
    vt_workers = [
        worker
        for worker in workers
        if isinstance(getattr(worker, "employment_terms", None), str)
        and "Allmän visstidsanställning" in worker.employment_terms
    ]

    if not vt_workers:
        return

    with st.expander("Visstidsanställning (VT) weekly limits", expanded=False):
        st.write("Justera veckovisa parametrar för arbetare med (VT) – Allmän visstidsanställning.")
        for idx, worker in enumerate(vt_workers):
            days_default = worker.days_per_week if pd.notna(getattr(worker, "days_per_week", None)) else 3
            hours_default = worker.hours_per_week if pd.notna(getattr(worker, "hours_per_week", None)) else 20

            worker.days_per_week = st.number_input(
                f"{worker.name} · Dagar per vecka",
                min_value=0.0,
                value=float(days_default or 5.0),
                step=1.0,
                key=f"{key_prefix}_days_{idx}_{worker.name}",
            )
            worker.hours_per_week = st.number_input(
                f"{worker.name} · Timmar per vecka",
                min_value=0.0,
                value=float(hours_default or 40.0),
                step=5.0,
                key=f"{key_prefix}_hours_{idx}_{worker.name}",
            )


def _scenario_builder_forms() -> tuple[List[Jobclass], List[Workerclass]]:
    jobs: List[Jobclass] = st.session_state.setdefault("manual_jobs", [])
    workers: List[Workerclass] = st.session_state.setdefault("manual_workers", [])

    with st.expander("Add a worker"):
        with st.form("add_worker_form"):
            name = st.text_input("Name")
            has_license = st.checkbox("Has driving licence", value=True)
            employee_number = st.text_input("Employee number")
            employment_percentage = st.number_input("Employment %", min_value=0.0, value=100.0, step=5.0)
            employment_choice = st.selectbox(
                "Employment terms",
                options=_EMPLOYMENT_TERMS,
                index=1,
                key="manual_worker_employment_choice",
            )
            if employment_choice == "Other":
                employment_terms = st.text_input(
                    "Describe employment terms",
                    value="",
                    key="manual_worker_employment_custom",
                )
            else:
                employment_terms = employment_choice

            days_key = "manual_worker_days"
            hours_key = "manual_worker_hours"
            last_term_key = "manual_worker_last_term"

            if days_key not in st.session_state:
                st.session_state[days_key] = 5.0
            if hours_key not in st.session_state:
                st.session_state[hours_key] = 40.0

            vt_selected = employment_terms == "(VT) – Allmän visstidsanställning"
            last_term = st.session_state.get(last_term_key)
            if vt_selected and last_term != employment_terms:
                st.session_state[days_key] = 3.0
                st.session_state[hours_key] = 20.0
            elif last_term == "(VT) – Allmän visstidsanställning" and employment_terms != last_term:
                st.session_state[days_key] = 5.0
                st.session_state[hours_key] = 40.0
            st.session_state[last_term_key] = employment_terms

            days_per_week = st.number_input(
                "Days per week",
                min_value=0.0,
                step=0.5,
                key=days_key,
            )
            hours_per_week = st.number_input(
                "Hours per week",
                min_value=0.0,
                step=1.0,
                key=hours_key,
            )
            salary_type = st.text_input("Salary type", value="Monthly")
            submitted = st.form_submit_button("Add worker")
            if submitted and name:
                worker = Workerclass(
                    name=name,
                    has_driver_license=has_license,
                    employee_number=employee_number,
                    days_per_week=days_per_week,
                    hours_per_week=hours_per_week,
                    employment_percentage=employment_percentage,
                    employment_terms=employment_terms,
                    salary_type=salary_type,
                )
                workers.append(worker)
                st.success(f"Added worker {name}")
                for key in (days_key, hours_key, last_term_key, "manual_worker_employment_choice", "manual_worker_employment_custom"):
                    st.session_state.pop(key, None)

    with st.expander("Add a job"):
        with st.form("add_job_form"):
            job_id = st.text_input("Job identifier")
            service_type = st.text_input("Service type", value="Hemstäd")
            job_date = st.date_input("Date", value=date.today())
            start_time = st.time_input("Start time", value=time(hour=8, minute=0))
            duration_hours = st.number_input("Duration (hours)", min_value=0.25, value=2.0, step=0.25)
            break_minutes = st.number_input("Break (minutes)", min_value=0, value=0, step=5)
            staff_name = st.text_input("Preferred worker", help="Optional name of the worker currently assigned")
            address = st.text_input("Address", value="Storgatan 1")
            postcode = st.text_input("Postcode", value="523 31")
            project_number = st.text_input("Project number", value="P-001")
            project_name = st.text_input("Project name", value="Cleaning")
            customer_number = st.text_input("Customer number", value="C-001")
            customer_name = st.text_input("Customer name", value="Customer")
            billing_reference = st.text_input("Billing reference", value="")
            job_type = st.text_input("Job type", value="Standard")
            status = st.text_input("Status", value="Planned")
            booking_frequency = st.selectbox("Booking frequency", options=["Engångsjobb", "Varje vecka", "Varannan vecka", "Var tredje vecka", "Var fjärde vecka"], index=0)
            submitted = st.form_submit_button("Add job")

            if submitted:
                start_dt = datetime.combine(job_date, start_time)
                end_dt = start_dt + timedelta(hours=float(duration_hours))
                time_slot = f"{start_dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')}"
                job = Jobclass(
                    job_id=job_id or f"manual-{len(jobs)+1}",
                    day=weekday_name_sv(job_date),
                    date=job_date,
                    time_slot=time_slot,
                    break_minutes=break_minutes,
                    staff_name=staff_name,
                    service_type=service_type,
                    work_address=address,
                    work_postcode=postcode,
                    project_number=project_number,
                    project_name=project_name,
                    customer_number=customer_number,
                    customer_name=customer_name,
                    billing_reference=billing_reference,
                    job_type=job_type,
                    status=status,
                    booking_frequency={
                        "Engångsjobb": 0,
                        "Varje vecka": 1,
                        "Varannan vecka": 2,
                        "Var tredje vecka": 3,
                        "Var fjärde vecka": 4,
                    }[booking_frequency],
                    service_time=float(duration_hours),
                )
                jobs.append(job)
                st.success(f"Added job {_ensure_job_id(job, len(jobs))}")

    return jobs, workers


# ---------------------------------------------------------------------------
# Streamlit layout
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Show and Optimize Schedule", layout="wide")
    st.title("Multi-day schedule optimiser")

    mode = st.sidebar.radio("Mode", options=["Schedule review", "Optimal insertion"], index=0)

    todays_date = st.sidebar.date_input("Today's date", value=date(2025,9,1))
    horizon = st.sidebar.number_input("Horizon (days)", min_value=0, value=0, step=1)
    allow_diff = st.sidebar.checkbox("Allow flexible day windows", value=False)
    scheduling_days = st.sidebar.number_input("Optimisation days", min_value=1, value=max(1, horizon + 1))
    start_hour = st.sidebar.slider("Day start (hour)", min_value=0, max_value=23, value=6)
    end_hour = st.sidebar.slider("Day end (hour)", min_value=1, max_value=23, value=18)
    time_limit = st.sidebar.number_input("Solver time limit (s)", min_value=1, value=10)

    depot_address = st.sidebar.text_input("Depot address", value="Storgatan 69 ,523 31 ULRICEHAMN, Sweden")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data sources")

    jobs_file = None
    staff_file = None

    jobs_file = st.sidebar.file_uploader("Jobs Excel (.xlsx)", type=["xlsx"])
    staff_file = st.sidebar.file_uploader("Staff Excel (.xlsx)", type=["xlsx"])

    jobs = _load_jobs(jobs_file, todays_date, horizon)
    workers = _load_workers(staff_file)

    manual_jobs, manual_workers = _scenario_builder_forms() if mode == "Optimal insertion" else ([], [])

    jobs = list(jobs) + list(manual_jobs)
    workers = list(workers) + list(manual_workers)

    st.subheader("Imported schedule")
    if jobs:
        original_df = _jobs_dataframe(jobs)
        show_original_table = st.checkbox(
            "Show imported Excel rows",
            value=False,
            key="show_original_table",
            help="Toggle to inspect the raw data that was loaded from Excel.",
        )
        if show_original_table:
            st.dataframe(original_df, width='stretch')
        _schedule_day_view(
            original_df,
            "Original schedule",
            key_prefix="original",
            depot_address=depot_address,
        )

        travel_summary = _calculate_travel_time_summary(original_df, depot_address)
        if travel_summary:
            st.markdown("#### Travel time summary (minutes)")
            st.dataframe(travel_summary["per_worker"], width='stretch', key="travel_per_worker")
            if not travel_summary["per_day"].empty:
                st.dataframe(travel_summary["per_day"], width='stretch', key="travel_per_day")
            st.metric(
                "Total travel minutes across all workers and days",
                f"{travel_summary['overall_total']:.2f}",
            )
        else:
            st.caption(
                "Travel time summary is unavailable – ensure addresses can be geocoded "
                "and that OSRM responses are accessible."
            )
    else:
        st.info("No jobs loaded yet. Add jobs in the scenario builder or provide an Excel file.")

    st.subheader("Optimisation setup")
    setup_cols = st.columns([2, 1])

    with setup_cols[1]:
        st.markdown("#### Selection")
        selected_jobs = _select_jobs_ui(jobs)
        selected_workers = _select_workers_ui(workers)
        _configure_vt_workers(selected_workers)

    with setup_cols[0]:
        st.markdown("#### Run solver")
        metrics_cols = st.columns(2)
        with metrics_cols[0]:
            st.metric("Jobs selected", len(selected_jobs))
        with metrics_cols[1]:
            st.metric("Workers selected", len(selected_workers))
        run_clicked = st.button(
            "Run optimisation",
            disabled=not (selected_jobs and selected_workers),
        )

    if run_clicked:
        start_weekday = weekday_name_sv(todays_date)
        problem = _build_problem(selected_jobs, selected_workers, depot=depot_address, start_day=start_weekday, allow_diff=allow_diff)
        if not problem:
            st.error("Unable to build optimisation problem – please ensure jobs and workers are selected.")
            st.session_state.pop(_OPTIMISED_STATE_KEY, None)
        else:
            solution = solve_vrp_problem_definition(
                problem,
                days=scheduling_days,
                start_hour=start_hour,
                end_hour=end_hour,
                timelimit_seconds=time_limit,
                debug=False,
            )

            assignments = _collect_assignments(solution)
            if not assignments:
                st.warning("Solver did not produce any assignments.")
                st.session_state.pop(_OPTIMISED_STATE_KEY, None)
            else:
                result_df = _assignments_to_df(assignments, todays_date)
                objective = solution.get("objective")
                message = (
                    f"Optimisation complete. Objective value: {objective}"
                    if objective is not None
                    else "Optimisation complete."
                )
                st.session_state[_OPTIMISED_STATE_KEY] = {
                    "result_df": result_df.copy(),
                    "message": message,
                    "objective": objective,
                    "base_date": todays_date.isoformat(),
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                }

    _render_optimised_results(
        st.session_state.get(_OPTIMISED_STATE_KEY),
        depot_address=depot_address,
        todays_date=todays_date,
    )


if __name__ == "__main__":
    main()
