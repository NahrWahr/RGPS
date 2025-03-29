import georinex as gr
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

# --- Configuration ---
# Specify the paths to your RINEX v3 observation files
rinex_file_base = 'path/to/your/base_station.obs' # REPLACE with actual path
rinex_file_rover = 'path/to/your/rover_drone.obs' # REPLACE with actual path

# Specify the GNSS system(s) to use (e.g., 'G' for GPS, 'E' for Galileo)
SYSTEM = 'G' 

# Specify the observation types to use for L1 frequency
# Common choices: C1C (C/A code), L1C (Carrier Phase)
# Check your RINEX header ('SYS / # / OBS TYPES') to see what's available
CODE_OBS = 'C1C' # Pseudorange observable
PHASE_OBS = 'L1C' # Carrier phase observable

# Minimum number of common satellites required for DD calculation (need at least 1 ref + 1 other)
MIN_COMMON_SATS = 4 # Ideally 5+ for robust geometry

# --- Constants ---
SPEED_OF_LIGHT = 299792458.0 # m/s
# GPS L1 Frequency (check precise value if needed, e.g., based on RINEX header or standards)
# For L1C/A:
GPS_L1_FREQ = 1575.42e6 # Hz 
GPS_L1_WAVELENGTH = SPEED_OF_LIGHT / GPS_L1_FREQ

# --- Helper Functions ---
def select_obs(ds, system, code_obs, phase_obs):
    """Selects specific observation types for a given system."""
    try:
        # Select only the specified system
        ds_sys = ds.sel(sv=ds.sv.str.startswith(system))
        
        # Select the code and phase observables
        # Handle cases where one or both might be missing for some sats/times
        selected_obs = {}
        if code_obs in ds_sys:
            selected_obs[code_obs] = ds_sys[code_obs]
        if phase_obs in ds_sys:
             # Convert phase from cycles to meters
            selected_obs[phase_obs] = ds_sys[phase_obs] * GPS_L1_WAVELENGTH 
           
        if not selected_obs:
             raise ValueError(f"Neither {code_obs} nor {phase_obs} found for system {system}")

        # Combine selected observations into a new Dataset
        # Use coords from the first available observable
        first_key = next(iter(selected_obs))
        ds_sel = xr.Dataset(selected_obs, coords=ds_sys[first_key].coords)
        ds_sel['sv'] = ds_sys['sv'] # Ensure sv coordinate is preserved correctly
        return ds_sel
        
    except KeyError as e:
        print(f"Error selecting observations: {e}. Check RINEX header for available types.")
        print(f"Available observables in dataset: {list(ds.data_vars.keys())}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during observation selection: {e}")
        return None

# --- Main Processing ---
print(f"Loading BASE station data: {rinex_file_base}")
try:
    # Use use='G' or similar to potentially speed up loading if only GPS is needed
    # cache=True can speed up subsequent loads of the same file
    obs_base = gr.load(rinex_file_base, use=SYSTEM, cache=True) 
except FileNotFoundError:
    print(f"ERROR: Base RINEX file not found at {rinex_file_base}")
    exit()
except Exception as e:
    print(f"ERROR loading base file: {e}")
    exit()

print(f"Loading ROVER data: {rinex_file_rover}")
try:
    obs_rover = gr.load(rinex_file_rover, use=SYSTEM, cache=True)
except FileNotFoundError:
    print(f"ERROR: Rover RINEX file not found at {rinex_file_rover}")
    exit()
except Exception as e:
    print(f"ERROR loading rover file: {e}")
    exit()

print("Selecting observations...")
base_sel = select_obs(obs_base, SYSTEM, CODE_OBS, PHASE_OBS)
rover_sel = select_obs(obs_rover, SYSTEM, CODE_OBS, PHASE_OBS)

if base_sel is None or rover_sel is None:
    print("ERROR: Could not select required observations from one or both files. Exiting.")
    exit()

# --- Time Alignment and Double Differencing ---
print("Aligning data and calculating double differences...")

# Align datasets based on time - keep only common epochs
# Use 'inner' join to only keep times present in both
aligned_base, aligned_rover = xr.align(base_sel, rover_sel, join='inner', copy=False)

# Check if alignment resulted in any data
if aligned_base.time.size == 0:
    print("ERROR: No common observation epochs found between the two files.")
    exit()

print(f"Found {aligned_base.time.size} common epochs.")

results = [] # To store DD results per epoch

# Iterate through each common timestamp
for epoch_time in aligned_base.time:
    # Select data for the current epoch
    base_epoch = aligned_base.sel(time=epoch_time).dropna(dim='sv', how='any') # Drop sats with NaN for *any* selected obs
    rover_epoch = aligned_rover.sel(time=epoch_time).dropna(dim='sv', how='any')

    # Find common satellites observed at this epoch
    common_sats = np.intersect1d(base_epoch.sv.values, rover_epoch.sv.values)
    
    if len(common_sats) < MIN_COMMON_SATS:
        # print(f"Skipping epoch {epoch_time.values}: Not enough common satellites ({len(common_sats)} found, need {MIN_COMMON_SATS})")
        continue

    # --- Select Reference Satellite ---
    # Simple strategy: choose the first common satellite alphabetically/numerically
    # Better strategy: choose the satellite with the highest elevation (requires NAV data)
    ref_sat = common_sats[0] 
    other_sats = common_sats[1:]

    # Extract data for common satellites only
    base_common = base_epoch.sel(sv=common_sats)
    rover_common = rover_epoch.sel(sv=common_sats)

    # --- Calculate Single Differences (Rover - Base) for each satellite ---
    # Ensure data is aligned by satellite SV before differencing
    base_common_sorted, rover_common_sorted = xr.align(base_common, rover_common, join='inner', copy=False)

    single_diff_code = None
    single_diff_phase = None

    if CODE_OBS in rover_common_sorted and CODE_OBS in base_common_sorted:
         single_diff_code = rover_common_sorted[CODE_OBS] - base_common_sorted[CODE_OBS]
    if PHASE_OBS in rover_common_sorted and PHASE_OBS in base_common_sorted:
         single_diff_phase = rover_common_sorted[PHASE_OBS] - base_common_sorted[PHASE_OBS] # Phase is already in meters

    # --- Calculate Double Differences (Sat_k - Sat_ref) ---
    # Get the single differences for the reference satellite
    sd_code_ref = single_diff_code.sel(sv=ref_sat) if single_diff_code is not None else None
    sd_phase_ref = single_diff_phase.sel(sv=ref_sat) if single_diff_phase is not None else None

    epoch_results = {
        'time': pd.to_datetime(epoch_time.values), 
        'ref_sat': ref_sat, 
        'dd': []
    }

    for sat_k in other_sats:
        dd_entry = {'sat_k': sat_k}
        
        # Calculate DD for code
        if single_diff_code is not None and sd_code_ref is not None:
             sd_code_k = single_diff_code.sel(sv=sat_k)
             # Check if both values are valid before differencing
             if not np.isnan(sd_code_k.item()) and not np.isnan(sd_code_ref.item()):
                 dd_entry['dd_code'] = (sd_code_k - sd_code_ref).item() # .item() extracts scalar value
             else:
                 dd_entry['dd_code'] = np.nan

        # Calculate DD for phase
        if single_diff_phase is not None and sd_phase_ref is not None:
            sd_phase_k = single_diff_phase.sel(sv=sat_k)
            if not np.isnan(sd_phase_k.item()) and not np.isnan(sd_phase_ref.item()):
                 dd_entry['dd_phase_m'] = (sd_phase_k - sd_phase_ref).item() # Result in meters
            else:
                 dd_entry['dd_phase_m'] = np.nan

        epoch_results['dd'].append(dd_entry)
        
    results.append(epoch_results)
    # Optional: Print progress
    if len(results) % 50 == 0:
         print(f"Processed epoch {len(results)} / {aligned_base.time.size}...")


# --- Output Results (Example: Print first few epochs) ---
print("\n--- Double Difference Results (Example) ---")
for i, epoch_res in enumerate(results[:5]): # Print first 5 epochs
    print(f"\nEpoch: {epoch_res['time']}")
    print(f"Reference Satellite: {epoch_res['ref_sat']}")
    print("Double Differences:")
    for dd in epoch_res['dd']:
        sat_k = dd['sat_k']
        dd_c = dd.get('dd_code', 'N/A') 
        dd_p = dd.get('dd_phase_m', 'N/A')
        dd_c_str = f"{dd_c:.3f}" if isinstance(dd_c, float) else dd_c
        dd_p_str = f"{dd_p:.4f}" if isinstance(dd_p, float) else dd_p
        print(f"  Sat {sat_k}: DD_Code = {dd_c_str} m, DD_Phase = {dd_p_str} m")

if not results:
    print("\nNo double differences could be calculated. Check input files, common satellites, and observation types.")
