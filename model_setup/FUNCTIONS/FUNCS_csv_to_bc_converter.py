#%%
"""
Functions to convert CSV discharge files to Delft3D-FM boundary condition (.bc) files.
"""

import re
import pandas as pd
from datetime import datetime
from pathlib import Path

#%%
def date_to_seconds_since_reference(date_str: str, reference_date: datetime = datetime(2001, 1, 1)) -> int:
    """
    Convert a date string (YYYY-MM-DD) to seconds since a reference date.
    
    Parameters
    ----------
    date_str : str
        Date string in YYYY-MM-DD format
    reference_date : datetime
        Reference date (default: 2001-01-01 00:00:00)
        
    Returns
    -------
    int
        Seconds since reference date
    """
    date = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
    delta = date - reference_date
    return int(delta.total_seconds())


def csv_to_bc(
    csv_path: str,
    output_path: str,
    boundary_number: int,
    boundary_suffix: str = "0001",
    reference_date: datetime = datetime(2001, 1, 1),
    quantity: str = "dischargebnd",
    unit: str = "m3/s"
) -> None:
    """
    Convert a CSV file with timestamp and discharge data to a Delft3D-FM .bc file.
    
    Parameters
    ----------
    csv_path : str
        Path to the input CSV file with columns 'timestamp' and 'discharge_m3s'
    output_path : str
        Path for the output .bc file
    boundary_number : int
        Boundary number (e.g., 1 for Boundary01_0001)
    boundary_suffix : str
        Suffix for the boundary name (default: "0001")
    reference_date : datetime
        Reference date for time calculation (default: 2001-01-01 00:00:00)
    quantity : str
        Quantity name for the .bc file (default: "dischargebnd")
    unit : str
        Unit for the quantity (default: "m3/s")
    """
    df = pd.read_csv(csv_path)
    
    if 'timestamp' not in df.columns or 'discharge_m3s' not in df.columns:
        raise ValueError("CSV file must have 'timestamp' and 'discharge_m3s' columns")
    
    boundary_name = f"Boundary{boundary_number:02d}_{boundary_suffix}"
    ref_date_str = reference_date.strftime("%Y-%m-%d %H:%M:%S")
    
    header = f"""[forcing]
name              = {boundary_name}
function          = timeseries
timeInterpolation = linear
quantity          = time
unit              = seconds since {ref_date_str}
quantity          = {quantity}
unit              = {unit}
"""
    
    data_lines = []
    for _, row in df.iterrows():
        seconds = date_to_seconds_since_reference(row['timestamp'], reference_date)
        discharge = row['discharge_m3s']
        data_lines.append(f"{seconds}   {discharge}")
    
    with open(output_path, 'w') as f:
        f.write(header)
        f.write('\n'.join(data_lines))
    
    print(f"  Created: {output_path}")


def convert_csv_folder_to_bc(
    csv_dir: str,
    output_prefix: str = "Qr500_inflow_sinuous",
    reference_date: datetime = datetime(2001, 1, 1),
    output_dir: str = None
) -> None:
    """
    Convert all river_section_*.csv files in a folder to .bc files.
    
    Parameters
    ----------
    csv_dir : str
        Directory containing river_section_*.csv files
    output_prefix : str
        Prefix for output filenames (default: "Qr500_inflow_sinuous")
    reference_date : datetime
        Reference date for time calculation (default: 2001-01-01 00:00:00)
    output_dir : str, optional
        Directory to write .bc files into. Defaults to csv_dir.
    """
    csv_dir = Path(csv_dir)
    out_dir = Path(output_dir) if output_dir is not None else csv_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(csv_dir.glob("river_section_*.csv"))
    
    for csv_path in csv_files:
        # Extract boundary number from filename (e.g., river_section_1.csv -> 1)
        match = re.search(r'river_section_(\d+)\.csv', csv_path.name)
        if match:
            boundary_number = int(match.group(1))
            output_filename = f"{boundary_number:02d}_{output_prefix}.bc"
            output_path = out_dir / output_filename
            
            csv_to_bc(
                csv_path=str(csv_path),
                output_path=str(output_path),
                boundary_number=boundary_number,
                reference_date=reference_date
            )
