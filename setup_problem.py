import pandas as pd
from datetime import datetime
from pathlib import Path

from classes.Jobclass import Jobclass
from classes.Workerclass import Workerclass
from vrp.problem_structures import VRPProblemBuilder
from pyhelpers.day_indexing import weekday_name_sv

# sets up the optimization problem for multiple days for real data
# prefer the data that ships with the repository but still keep backward
# compatibility with the historical desktop path that the original scripts
# referenced.
_REPO_INPUT_DIR = Path(__file__).resolve().parent / "Kund1" / "september_clean"
_LEGACY_INPUT_DIR = (
    Path.home() / "Desktop" / "OptimalRoutePlanning" / "Code" / "Kund1" / "september_clean"
)
_REPO_STAFF_DIR = Path(__file__).resolve().parent / "Kund1" / "anställda"
_LEGACY_STAFF_DIR = (
    Path.home()
    / "Desktop"
    / "OptimalRoutePlanning"
    / "Code"
    / "Kund1"
    / "anställda"
)
# adress of pilot customer 1 depot
depot_adr = "Storgatan 69 ,523 31 ULRICEHAMN, Sweden"
# dummy todays date
todays_date = datetime(2025, 9, 1)
scheduling_horizon = 1 

bokningsfrekvens_dict ={"Engångsjobb" : 0,
                        "Varje vecka" :1,
                        "Varannan vecka":2,
                        "Var tredje vecka":3,
                        "Var fjärde vecka":4}




def retrieve_jobs(todays_date, horizon_days=scheduling_horizon):
    """
    Read the Excel file in the data directory and return rows where the 'Datum' column
    falls between `todays_date` and `todays_date + horizon_days` (inclusive).

    Parameters:
        todays_date (datetime or str or pandas.Timestamp): start date (will be normalized to date)
        horizon_days (int): number of days after todays_date to include (inclusive)
    Returns:
        pandas.DataFrame: filtered rows (copy). If the Excel file or column is missing,
        raises FileNotFoundError or KeyError respectively.
    """
    # resolve path fallback in case the default repo directory doesn't exist
    for candidate in (_REPO_INPUT_DIR, _LEGACY_INPUT_DIR):
        if candidate.exists():
            path = candidate
            break
    else:
        raise FileNotFoundError(
            "Could not locate the input directory. Checked: "
            f"{_REPO_INPUT_DIR} and {_LEGACY_INPUT_DIR}"
        )

    excel_path = path / "september_arbete.xlsx"
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found at {excel_path}")

    df = pd.read_excel(excel_path, engine="openpyxl")
    if "Datum" not in df.columns:
        raise KeyError("Column 'Datum' not found in the Excel file")

    # normalize to dates (remove time component)
    df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce").dt.normalize()
    
    start = pd.to_datetime(todays_date).normalize()
    print(start)
    end = (start + pd.Timedelta(days=horizon_days)).normalize()
    print(end)

    mask = (df["Datum"] >= start) & (df["Datum"] <= end)
    print (f"Retrieved {mask.sum()} jobs between {start.date()} and {end.date()}.")
    return df.loc[mask].copy()
    
def format_jobs(jobs_df):
    """Convert DataFrame rows to Jobclass instances with calculated service times."""

    first_sixteen_columns = [
        "Dag",
        "Datum",
        "Tid",
        "Rast(min)",
        "Personal",
        "Tjänst",
        "Arb.adress",
        "Arb.adress postnr.",
        "Proj.nr",
        "Projekt",
        "Kundnr",
        "Kund",
        "Fakt.bes.",
        "Typ",
        "Status",
        "Bokningsfrekvens",
    ]

    job_list = []
    for index, row in jobs_df.iterrows():
        job_id = row.get("Job ID", index)  # Fallback to the DataFrame index

        # Calculate service time from Tid column (e.g. "06:00-07:15")
        service_time = 0  # default
        try:
            if pd.notna(row.get("Tid")) and "-" in str(row["Tid"]):
                start_str, end_str = row["Tid"].split("-")
                base_date = pd.to_datetime(row["Datum"]).strftime("%Y-%m-%d")
                start_time = pd.to_datetime(f"{base_date} {start_str.strip()}")
                end_time = pd.to_datetime(f"{base_date} {end_str.strip()}")
                service_time = (end_time - start_time).total_seconds() / 3600  # hours
                # job_start =start_time.hour * 60 + start_time.minute # start time for use with time window in minutes from midnight
        except (ValueError, AttributeError) as e:
            print(f"Warning: Could not parse time for job {job_id}: {e}")
            service_time = 1  # fallback to 1 hour if parsing fails

        column_values = {column: row.get(column) for column in first_sixteen_columns}
        
        freq_num = bokningsfrekvens_dict[column_values.get("Bokningsfrekvens")]
        

        job = Jobclass(
            job_id=job_id,
            day=column_values.get("Dag"),
            date=column_values.get("Datum"),
            time_slot=column_values.get("Tid"),
            break_minutes=column_values.get("Rast(min)"),
            staff_name=column_values.get("Personal"),
            service_type=column_values.get("Tjänst"),
            work_address=column_values.get("Arb.adress"),
            work_postcode=column_values.get("Arb.adress postnr."),
            project_number=column_values.get("Proj.nr"),
            project_name=column_values.get("Projekt"),
            customer_number=column_values.get("Kundnr"),
            customer_name=column_values.get("Kund"),
            billing_reference=column_values.get("Fakt.bes."),
            job_type=column_values.get("Typ"),
            status=column_values.get("Status"),
            booking_frequency=freq_num,
            service_time=service_time,
            
        )
        job_list.append(job)

    return job_list


def retrieve_workers():
    """Read the September staff details Excel file and return a DataFrame."""

    for candidate in (_REPO_STAFF_DIR, _LEGACY_STAFF_DIR):
        if candidate.exists():
            staff_dir = candidate
            break
    else:
        raise FileNotFoundError(
            "Could not locate the staff directory. Checked: "
            f"{_REPO_STAFF_DIR} and {_LEGACY_STAFF_DIR}"
        )

    staff_path = staff_dir / "September_staff_details.xlsx"
    if not staff_path.exists():
        raise FileNotFoundError(f"Staff Excel file not found at {staff_path}")

    return pd.read_excel(staff_path, engine="openpyxl")


def format_workers(workers_df):
    """Convert DataFrame rows to Workerclass instances."""

    worker_list = []
    for _, row in workers_df.iterrows():
        worker = Workerclass(
            name=row.get("Namn"),
            has_driver_license=str(row.get("Har körkort", "")).strip().lower() in {"ja", "yes", "y", "1", "true"},
            employee_number=row.get("Anst.nr"),
            days_per_week=row.get("Dagar per vecka"),
            hours_per_week=row.get("Timmar per vecka"),
            employment_percentage=row.get("Anställningsgrad (%)"),
            employment_terms=row.get("Anst.villkor"),
            salary_type=row.get("Lönetyp"),
        )
        worker_list.append(worker)

    return worker_list


def build_vrp_problem(job_list, worker_list, start_weekday, allow_diff):
    """Create a :class:`VRPProblemDefinition` for the current data set."""

    builder = VRPProblemBuilder()
    return builder.build_problem(jobs=job_list, workers=worker_list, depot_address=depot_adr, start_weekday=start_weekday, allow_diff=allow_diff)

if __name__ == "__main__":
    start_weekday = weekday_name_sv(todays_date)
    jobs_df = retrieve_jobs(todays_date, horizon_days=0)
    job_class_list = format_jobs(jobs_df)
    workers_df = retrieve_workers()
    worker_class_list = format_workers(workers_df)
    a_job = job_class_list[2]
    print(a_job.time_slot)

    vrp_problem = build_vrp_problem(job_class_list, worker_class_list, start_weekday=start_weekday, allow_diff=True)
    print(vrp_problem.nodes[1])
    print(
        "Prepared {jobs} jobs, {workers} workers and a {matrix}x{matrix} time matrix".format(
            jobs=len(vrp_problem.nodes),
            workers=len(vrp_problem.worker_constraints),
            matrix=len(vrp_problem.time_matrix),
        )
    )
    
