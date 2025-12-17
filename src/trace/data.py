"""
Data handling module for TRACE.

This module provides functions to fetch and prepare conflict event data from ACLED,
as well as utilities for loading hospital and mortality data.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests


def fetch_acled_data(
    country: str,
    start_date: str,
    end_date: str,
    api_token: Optional[str] = None,
    fields: Optional[list[str]] = None,
    api_email: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch ACLED conflict event data for a given country and date range.

    ACLED (Armed Conflict Location & Event Data Project) provides detailed information
    on political violence and protest events. This function queries the ACLED API
    and returns a pandas DataFrame with event data.

    Parameters
    ----------
    country : str
        Country name (e.g., "Palestine", "Ukraine", "Syria")
    start_date : str
        Start date in format "YYYY-MM-DD"
    end_date : str
        End date in format "YYYY-MM-DD"
    api_token : str, optional
        ACLED API access token. Required for full API access.
        Register at https://developer.acleddata.com/
    fields : list of str, optional
        Specific fields to retrieve. If None, returns all available fields.
    api_email : str, optional
        Email associated with ACLED API account

    Returns
    -------
    pd.DataFrame
        DataFrame with columns including:
        - event_date: Date of the event
        - latitude: Latitude coordinate
        - longitude: Longitude coordinate
        - fatalities: Number of reported fatalities
        - event_type: Type of conflict event
        - notes: Description of the event

    Examples
    --------
    >>> events_df = fetch_acled_data(
    ...     country="Palestine",
    ...     start_date="2023-01-01",
    ...     end_date="2023-12-31",
    ...     api_token="YOUR_TOKEN",
    ...     api_email="your@email.com"
    ... )
    >>> print(events_df.head())

    Notes
    -----
    - API access requires registration at https://developer.acleddata.com/
    - Free tier has rate limits; check ACLED documentation for details
    - The API returns JSON data which is converted to a DataFrame
    """
    base_url = "https://api.acleddata.com/acled/read"

    # Build query parameters
    params = {
        "country": country,
        "event_date": f"{start_date}|{end_date}",
        "event_date_where": "BETWEEN",
        "limit": 0,  # 0 means no limit (retrieve all records in range)
        "format": "json",
    }

    if fields:
        params["fields"] = ",".join(fields)

    if api_token:
        params["key"] = api_token

    if api_email:
        params["email"] = api_email

    # Acknowledge terms of use
    params["terms"] = "accept"

    # Make the GET request to ACLED API
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch ACLED data: {e}")

    data = response.json()

    # The JSON structure typically contains a list of event records under 'data' key
    events = data.get("data", [])

    if not events:
        print(f"Warning: No events found for {country} between {start_date} and {end_date}")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(events)

    # Ensure date is in datetime format
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"])

    # Convert numeric fields from string to numeric types if needed
    num_fields = ["fatalities", "latitude", "longitude"]
    for field in num_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors="coerce")

    return df


def prepare_acled_events(
    events_df: pd.DataFrame, start_date: str, end_date: str
) -> tuple[list[int], list[int], np.ndarray, pd.DatetimeIndex]:
    """
    Prepare ACLED events DataFrame for modeling.

    This function transforms raw ACLED event data into the format required by
    the casualty model: daily event counts, event-day mappings, and coordinates.

    Parameters
    ----------
    events_df : pd.DataFrame
        DataFrame from fetch_acled_data with event information
    start_date : str
        Start date of analysis period in format "YYYY-MM-DD"
    end_date : str
        End date of analysis period in format "YYYY-MM-DD"

    Returns
    -------
    events_by_day : list of int
        Array of length T with number of events per day
    event_days : list of int
        Array of day indices (0..T-1) for each event in the dataset
    event_coords : np.ndarray
        Array of shape (E, 2) with (latitude, longitude) for each event
    dates : pd.DatetimeIndex
        DatetimeIndex of actual dates corresponding to each time index

    Examples
    --------
    >>> events_by_day, event_days, event_coords, dates = prepare_acled_events(
    ...     events_df, "2023-01-01", "2023-12-31"
    ... )
    >>> print(f"Total days: {len(events_by_day)}")
    >>> print(f"Total events: {len(event_days)}")
    """
    # Create a date range for the analysis period
    dates = pd.date_range(start_date, end_date, freq="D")
    date_to_index = {d: idx for idx, d in enumerate(dates)}
    n_days = len(dates)

    # Filter events to within the specified range
    mask = (events_df["event_date"] >= pd.to_datetime(start_date)) & (
        events_df["event_date"] <= pd.to_datetime(end_date)
    )
    events_df = events_df.loc[mask].copy()
    events_df.sort_values("event_date", inplace=True)

    # Map each event's date to an index in 0..T-1
    event_days = (
        events_df["event_date"].map(lambda d: date_to_index.get(d.normalize(), None)).tolist()
    )

    # Remove any events that fall outside the known date range (None indices)
    valid_idx = [(i, day) for i, day in enumerate(event_days) if day is not None]

    if not valid_idx:
        # If no valid events in range, return empty structures
        return [0] * n_days, [], np.array([]).reshape(0, 2), dates

    indices, event_days = zip(*valid_idx)
    event_days = list(event_days)

    # Extract coordinates for each valid event
    if "latitude" in events_df.columns and "longitude" in events_df.columns:
        coords = events_df.loc[events_df.index[list(indices)], ["latitude", "longitude"]].values
    else:
        coords = np.array([]).reshape(0, 2)

    # Compute daily event counts
    events_by_day = [0] * n_days
    for day in event_days:
        events_by_day[day] += 1

    return events_by_day, event_days, coords, dates


def load_hospital_data(filepath: str) -> pd.DataFrame:
    """
    Load daily hospital casualty incidence data from a CSV file.

    The file is expected to have columns: 'date', 'hospital_id', 'count'.
    Returns a pandas DataFrame pivoted to shape (date x hospital).

    Parameters
    ----------
    filepath : str
        Path to CSV file containing hospital data

    Returns
    -------
    pd.DataFrame
        DataFrame with dates as index and hospital IDs as columns,
        containing daily casualty counts

    Examples
    --------
    >>> hospital_df = load_hospital_data("hospital_admissions.csv")
    >>> print(hospital_df.shape)  # (num_days, num_hospitals)
    """
    df = pd.read_csv(filepath, parse_dates=["date"])

    # Pivot to get each hospital as a column, dates as index
    df_pivot = df.pivot_table(index="date", columns="hospital_id", values="count", fill_value=0)

    # Ensure sorted by date
    df_pivot = df_pivot.sort_index()

    return df_pivot


def load_national_deaths(filepath: str) -> pd.Series:
    """
    Load daily national death counts from a CSV file.

    The file should have columns: 'date', 'deaths'.

    Parameters
    ----------
    filepath : str
        Path to CSV file containing mortality data

    Returns
    -------
    pd.Series
        Series indexed by date with daily death counts

    Examples
    --------
    >>> deaths_series = load_national_deaths("mortality_data.csv")
    >>> print(deaths_series.head())
    """
    df = pd.read_csv(filepath, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    deaths_series = pd.Series(df["deaths"].values, index=df["date"])
    return deaths_series


def create_hospital_coordinates(
    hospital_ids: list[str], locations: Optional[dict[str, tuple[float, float]]] = None
) -> np.ndarray:
    """
    Create hospital coordinate array for spatial modeling.

    Parameters
    ----------
    hospital_ids : list of str
        List of hospital identifiers
    locations : dict, optional
        Dictionary mapping hospital_id to (latitude, longitude) tuples.
        If None, generates random locations for testing.

    Returns
    -------
    np.ndarray
        Array of shape (H, 2) with (latitude, longitude) for each hospital

    Examples
    --------
    >>> coords = create_hospital_coordinates(
    ...     ["H1", "H2", "H3"],
    ...     {"H1": (31.5, 34.5), "H2": (31.6, 34.6), "H3": (31.4, 34.4)}
    ... )
    """
    if locations is None:
        # Generate random locations for testing
        print("Warning: No locations provided, generating random coordinates")
        return np.random.uniform(30, 35, size=(len(hospital_ids), 2))

    coords = np.array([locations[hid] for hid in hospital_ids])
    return coords


def load_example_acled_data() -> pd.DataFrame:
    """
    Load example ACLED data included with the package.

    This function loads a sample ACLED dataset for Palestine that is
    included with the package for demonstration and testing purposes.

    Returns
    -------
    pd.DataFrame
        DataFrame with ACLED event data

    Examples
    --------
    >>> events_df = load_example_acled_data()
    >>> print(f"Loaded {len(events_df)} events")
    >>> print(events_df.columns.tolist())
    """
    # Get the path to the data file
    package_dir = Path(__file__).parent
    data_file = package_dir / "data_files" / "acled_example.csv"

    if not data_file.exists():
        raise FileNotFoundError(
            f"Example ACLED data file not found at {data_file}. "
            "Please ensure the package is properly installed."
        )

    # Load the CSV
    df = pd.read_csv(data_file)

    # Ensure date is in datetime format
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"])

    # Convert numeric fields
    num_fields = ["fatalities", "latitude", "longitude"]
    for field in num_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors="coerce")

    return df


def fetch_palestine_mortality_data(
    start_date: Optional[str] = None, end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch daily mortality data for Gaza from Tech for Palestine.

    This function retrieves daily casualty reports from the Tech for Palestine
    dataset, which aggregates data from Gaza's Ministry of Health and Government
    Media Office.

    Parameters
    ----------
    start_date : str, optional
        Start date in format "YYYY-MM-DD". If None, returns all available data.
    end_date : str, optional
        End date in format "YYYY-MM-DD". If None, returns all available data.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - report_date: Date of the report
        - killed: Daily deaths (if available)
        - killed_cum: Cumulative deaths
        - injured: Daily injuries (if available)
        - injured_cum: Cumulative injuries
        - ext_killed: Extrapolated daily deaths
        - ext_injured: Extrapolated daily injuries

    Examples
    --------
    >>> mortality_df = fetch_palestine_mortality_data(
    ...     start_date="2023-10-07",
    ...     end_date="2023-12-31"
    ... )
    >>> print(f"Loaded {len(mortality_df)} days of mortality data")

    Notes
    -----
    - Data is from https://data.techforpalestine.org/
    - Numbers include only direct casualties of war
    - Extrapolated fields (ext_*) fill gaps in official reporting
    - Data is updated daily (morning Eastern time)

    References
    ----------
    - Dataset documentation: https://data.techforpalestine.org/docs/casualties-daily/
    - Ministry of Health methodology: https://www.dropsitenews.com/p/how-gaza-health-ministry-counts-dead
    """
    url = "https://data.techforpalestine.org/api/v2/casualties_daily.csv"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch Palestine mortality data: {e}")

    # Read CSV from response content
    from io import StringIO

    df = pd.read_csv(StringIO(response.text))

    # Convert report_date to datetime
    if "report_date" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date"])
    elif "date" in df.columns:
        df = df.rename(columns={"date": "report_date"})
        df["report_date"] = pd.to_datetime(df["report_date"])

    # Filter by date range if specified
    if start_date is not None:
        df = df[df["report_date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["report_date"] <= pd.to_datetime(end_date)]

    # Sort by date
    df = df.sort_values("report_date").reset_index(drop=True)

    return df


def prepare_mortality_data(
    mortality_df: pd.DataFrame, start_date: str, end_date: str, use_extrapolated: bool = True
) -> pd.Series:
    """
    Prepare mortality data for modeling.

    Converts a mortality DataFrame into a daily time series aligned with
    the analysis period.

    Parameters
    ----------
    mortality_df : pd.DataFrame
        DataFrame from fetch_palestine_mortality_data()
    start_date : str
        Start date of analysis period in format "YYYY-MM-DD"
    end_date : str
        End date of analysis period in format "YYYY-MM-DD"
    use_extrapolated : bool, default=True
        If True, uses extrapolated daily deaths (ext_killed) to fill gaps.
        If False, uses only official daily deaths (killed).

    Returns
    -------
    pd.Series
        Series indexed by date with daily death counts

    Examples
    --------
    >>> mortality_df = fetch_palestine_mortality_data()
    >>> deaths_series = prepare_mortality_data(
    ...     mortality_df, "2023-10-07", "2023-12-31"
    ... )
    """
    # Create date range for analysis period
    dates = pd.date_range(start_date, end_date, freq="D")

    # Choose which field to use
    if use_extrapolated and "ext_killed" in mortality_df.columns:
        death_field = "ext_killed"
    elif "killed" in mortality_df.columns:
        death_field = "killed"
    else:
        raise ValueError("No suitable death count field found in mortality data")

    # Create series indexed by date
    mortality_df = mortality_df.set_index("report_date")

    # Reindex to analysis period, filling missing values with 0
    deaths_series = mortality_df[death_field].reindex(dates, fill_value=0)

    return deaths_series
