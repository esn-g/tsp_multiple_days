import pandas as pd
from datetime import datetime
from pathlib import Path
from code.tsp_multiple_days.classes.Jobclass import Jobclass
from code import data_harvest

# sets up the optimization problem for multiple days for real data
input_dir = Path.home() / "Desktop" / "OptimalRoutePlanning" / "Code" / "Kund1" / "september_clean"
# adress of pilot customer 1 depot
depot_adr = "Storgatan 69 ,523 31 ULRICEHAMN, Sweden"
# dummy todays date
todays_date = datetime(2025, 9, 1)


def retrieve_jobs(todays_date, horizon_days=4):
    """
    Read the Excel file in `input_dir` and return rows where the 'Datum' column
    falls between `todays_date` and `todays_date + horizon_days` (inclusive).

    Parameters:
        todays_date (datetime or str or pandas.Timestamp): start date (will be normalized to date)
        horizon_days (int): number of days after todays_date to include (inclusive)
    Returns:
        pandas.DataFrame: filtered rows (copy). If the Excel file or column is missing,
        raises FileNotFoundError or KeyError respectively.
    """
    # resolve path fallback in case input_dir doesn't exist (common dev vs prod path differences)
    path = input_dir
    if not path.exists():
        alt = Path(__file__).resolve().parents[1] / "Kund1" / "september_clean"
        if alt.exists():
            path = alt

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

    mask = (df["Datum"] >= start) & (df["Datum"] <= end)
    print (f"Retrieved {mask.sum()} jobs between {start.date()} and {end.date()}.")
    return df.loc[mask].copy()

def format_jobs(jobs_df):
    """Convert DataFrame rows to Jobclass instances with calculated service times."""
    job_list = []
    for index, row in jobs_df.iterrows():
        # Basic job info
        job_id = row.get("Job ID", index)  # Use index as fallback ID
        description = row.get("Tjänst", "No Description") + row.get("Proj.nr ", "No Description") +
        zip_code = row.get("Arb.adress postnr.", "")
        adress = f"{row.get('Arb.adress', '')} {zip_code}".strip()
        day = row.get("Dag" )
        
        # Calculate service time from Tid column (e.g. "06:00-07:15")
        service_time = 0  # default
        try:
            if pd.notna(row.get('Tid')) and '-' in str(row['Tid']):
                start_str, end_str = row['Tid'].split('-')
                # Create datetime objects for the same date to calculate duration
                base_date = pd.to_datetime(row['Datum']).strftime('%Y-%m-%d')
                start_time = pd.to_datetime(f"{base_date} {start_str.strip()}")
                end_time = pd.to_datetime(f"{base_date} {end_str.strip()}")
                service_time = (end_time - start_time).total_seconds() / 3600  # hours
        except (ValueError, AttributeError) as e:
            print(f"Warning: Could not parse time for job {job_id}: {e}")
            service_time = 1  # fallback to 1 hour if parsing fails
        
        # Default values for other required parameters
        
        # day_required = pd.to_datetime(row['Datum']).date()  # we know Datum exists from retrieve_jobs
        # time_slot_required = row.get('Tid', "")  # original time slot string
        # periodicity = row.get('Periodicity', 'once')  # default to one-time job
        
        job = Jobclass(
            job_id=job_id,
            description=description,
            adress=adress,
            service_time=service_time,
        )
        job_list.append(job)
    
    return job_list
def make_time_matrix(job_list):
    """Create a time matrix based on job addresses including depot."""
    job_addresses = [depot_adr]
    for job in job_list:
        job.get_job_adress()
        job_addresses.append(job.get_job_adress())
    pass

if __name__ == "__main__":
    jobs_df = retrieve_jobs(todays_date, horizon_days=4)
    job_list = format_jobs(jobs_df)
    print(job_list)
    
    print(f"Found {len(jobs_df)} jobs between {pd.to_datetime(todays_date).date()} and {(pd.to_datetime(todays_date) + pd.Timedelta(days=5)).date()}")
    
    