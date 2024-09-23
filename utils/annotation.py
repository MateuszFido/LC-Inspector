from pyteomics import mzml, auxiliary
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import re
import sys
from alive_progress import alive_bar
from scipy.signal import find_peaks, peak_widths
from pathlib import Path

def load_ms1_data(path: str) -> tuple[list, np.ndarray, str]:
    """
    Using the pyteomics library, load the data from the .mzML file into a pandas DataFrame.
    
    Parameters
    ----------
    path : str
        The path to the .mzML file.
    
    Returns
    -------
    data : List of Scan objects
        The list of Scan objects containing the MS data.
    mz_axis : np.ndarray
        The m/z axis for the intensity values.
    filename : str
        The filename without the extension.
    """
    # Load the data
    data = mzml.MzML(path)
    
    # Take only the scans where ms level is 1
    print("Loading MS data...")
    data = [scan for scan in data if scan['ms level'] == 1]

    # Look up the necessary fields from the first scan in the file for m/z axis determination
    low_mass = auxiliary.cvquery(data[0], 'MS:1000501')
    high_mass = auxiliary.cvquery(data[0], 'MS:1000500')
    # Calculate the resolution of the m/z axis
    resolution = int((high_mass - low_mass) / 0.01)
    # Create the m/z axis
    mz_axis = np.linspace(low_mass, high_mass, resolution, dtype=np.float64)
    # Get the filename without the extension, optional 
    filename = Path(path).stem

    # Insert the path as the first index 
    data.insert(0, path)

    return data, mz_axis, filename


def calculate_mz_axis(data):
    """
    Calculate the m/z axis from a list of Scan objects.

    Parameters
    ----------
    data : List of Scan objects
        The list of Scan objects containing the MS data.

    Returns
    -------
    mz_axis : np.ndarray
        The m/z axis for the intensity values.
    """
    # Look up the necessary fields from the first scan in the file for m/z axis determination
    low_mass = auxiliary.cvquery(data[0], 'MS:1000501')
    high_mass = auxiliary.cvquery(data[0], 'MS:1000500')
    # Calculate the resolution of the m/z axis
    resolution = int((high_mass - low_mass) / 0.01)
    # Create the m/z axis
    mz_axis = np.linspace(low_mass, high_mass, resolution, dtype=np.float64)

    return mz_axis




def average_intensity(path: str) -> pd.DataFrame:
    """
    Averages the intensity across all scans.

    Parameters
    ----------
    path : str
        The path to the .mzML file.

    Returns
    -------
    data_matrix : pd.DataFrame
        A DataFrame containing the m/z and intensity values.
    """
    
    # Load the data
    data = mzml.MzML(path)
    
    # Take only the scans where ms level is 1
    data = [scan for scan in data if scan['ms level'] == 1]
    
    # Calculate the m/z axis
    mz_axis = calculate_mz_axis(data)
    
    # Initialize the average intensity array
    avg_int = np.zeros(len(mz_axis))
    
    # Grab the filename from the first scan 
    bar_title = f"Averaging the spectra for {path.split('/')[-1].replace('.mzml', '')}."
    
    # Iterate over the scans, calculate the average intensity and store it in avg_int
    counter = 0
    with alive_bar(len(data), title=bar_title, calibrate=2) as bar:
        for scan in data:
            counter += 1
            # Get m/z values and their intensities from the MzML path
            mz_array = np.ndarray.transpose(scan['m/z array'])
            intensity_array = np.ndarray.transpose(scan['intensity array'])
            
            # Interpolate continuous intensity signal from discrete m/z
            int_interp = np.interp(mz_axis, mz_array, intensity_array, left=0, right=0)
            avg_int += int_interp              
            bar()
    
    # Calculate the average intensity
    int_axis = np.round((avg_int / counter))
    
    # Store the averaged intensity values in a DataFrame
    data_matrix = pd.DataFrame({'m/z': np.round(mz_axis, 4), 'intensity / a.u.': int_axis }, dtype=np.float64)
    
    # Save to a separate .csv file
    data_matrix.to_csv(path.replace('.mzml', '.csv'), index=False)
    
    return data_matrix

class Peak():
    '''Support class for Peak objects. \n
    
    Properties:
    -----------
    mz: np.ndarray
        Numpy array of mz values computed from indices of intensity axis, returned by scipy.find_peaks().
    index: int
        Index of the peak centroid. 
    width: list 
        Peak boundaries calculated at 0.9 of peak amplitude.
        '''

    def __init__(self, index: int, mz: np.ndarray, width: list):
        self.index = index
        self.mz = mz
        self.width = width

    def __str__(self):
        return f"Feature with m/z range: {self.mz} and intensity range: {self.int_range}"

        
def pick_peaks(path, mz_axis):
    '''
    Peak-picking function. Uses SciPy's find_peaks() to perform peak-picking on a given chromatogram. 

    Parameters
    ----------
    path: str
        Path to the .csv file containing the chromatogram data.

    Returns
    -------
    peaklist: list
        List of Peak objects.
    '''

    data = pd.read_csv(path.replace('.mzml', '.csv'))
    peaklist = []

    # Find peaks
    peaks = find_peaks(data['intensity / a.u.'], height=1000)

    # Calculate peak widths at 0.9 of peak amplitude
    widths, width_heights, left, right = peak_widths(data['intensity / a.u.'], peaks[0], rel_height=0.9)
    
    # For each peak, extract their properties and append the Peak to peaklist
    counter = 0
    for peak_idx in peaks[0]:
        mz = mz_axis[int(np.floor(left[counter])):int(np.ceil(right[counter]))]   # m/z range
        width = [int(np.floor(left[counter])), int(np.ceil(right[counter]))]      # left and right base, rounded down and up respectively
        peak = Peak(peak_idx, mz, width) # create the Peak object
        counter += 1
        peaklist.append(peak)
    
    return peaklist

def construct_xic(path):
    """
    Construct the XICs from the chromatogram data.

    Parameters
    ----------
    path : str
        The path to the .mzML file.

    Returns
    -------
    trc : pd.DataFrame
        The XICs for the given peaks.
    """
    
    file = mzml.MzML(path) # Get the MzML
    scans = [scan for scan in file if scan['ms level'] == 1]

    mz_axis = calculate_mz_axis(scans)

    peaks = pick_peaks(path=path.replace('.mzml', '.csv'), mz_axis = mz_axis)

    # Initialize empty arrays to store the TIC, scan times, and XICs
    tic = []
    scan_times = []
    data = np.empty((len(peaks)+1, len(scans)))

    # Construct the XICs
    bar_title = f"Constructing XICs for {path.split('/')[-1].replace('.mzml', '')}"
    with alive_bar(len(scans), title=bar_title, calibrate=2) as bar:
        for j, scan in enumerate(scans):
            scan_times.append(auxiliary.cvquery(scan, 'MS:1000016'))
            tic.append(scan['total ion current'])
            mz_array = np.ndarray.tolist(scan['m/z array'])
            intensity_array = np.ndarray.tolist(scan['intensity array'])
            # Interpolate intensity linearly for each scan from mz_array and intensity_array onto MZ_AXIS
            int_interp = np.interp(mz_axis, mz_array, intensity_array) 
            data[0][j] = scan['index']
            i = 1
            for peak in peaks:
                if i < len(peaks)+2:
                    data[i][0] = round(np.median(peak.mz), 4)
                feature_int = int_interp[peak.width[0]:peak.width[1]]
                time_trace = np.round(np.trapz(feature_int))
                data[i][j] = time_trace
                i += 1
            bar()
    trc = np.ndarray.tolist(data)
    
    trc.insert(1, scan_times)
    trc.insert(1, tic)
    
    # Write the XICs to a .csv file
    # Use list comprehension for column titles
    mzs = [f'pos{round(np.median(peak.mz), 4)}' for peak in peaks] if 'pos' in path.split('/')[-1].replace('.mzml', '') else [f'neg{round(np.median(peak.mz), 4)}' for peak in peaks]
    print(path.split('/')[-1].replace('.mzml', ''))
    print(mzs)
    columns = ['MS1 scan ID', 'TIC (a.u.)', 'Scan time (min)']
    columns.extend(mzs)
    print(columns)
    trc = pd.DataFrame(trc).T

    print(trc)
    trc.to_csv(path.replace('.mzml', '_XIC.csv'), header=columns, index=False)

    return trc
  



def quantify_targeted(data, compounds, path):
    '''
    Perform peak-picking, find the closest value to the given targeted list of ions and report their averaged intensity.
    '''
    peaks, properties = find_peaks(data['intensity / a.u.'], height=1000)

    #TODO: Implement 


    return 


def annotate_lc_data(path, compounds):
    
    # Load the XIC data, skip MS1 scan index and TIC rows
    """
    Annotate the LC data with the given targeted list of ions.

    Parameters
    ----------
    path : str
        The path to the .mzML file.
    compounds : dict
        A dictionary of targeted compounds with their respective m/z values
        as the keys and the ion names as the values.

    Returns
    -------
    None
    """
    
    data = pd.read_csv(path.replace('.mzml', '_XIC.csv'), header=0)
    
    # Prepare the plotting folder
    os.makedirs(Path(path).parent.parent / 'plots' / 'XICs', exist_ok=True)
    plot_path = Path(path).parent.parent / 'plots' / 'XICs'

    fig, axes = plt.subplots(nrows=len(compounds), ncols=1, figsize=(8, int(2*len(compounds))))
    fig.suptitle(f'{os.path.basename(path)}')
    for i, (compound, ions) in enumerate(compounds.items()):
        for j, ion in enumerate(ions.keys()):
            # Look for the closest m/z value in the first row of data 
            closest = np.abs(data.iloc[0] - ion).idxmin()

            # Get the respective time value for the highest intensity of this m/z
            scan_time = data['Scan time (min)'].iloc[data[closest].idxmax()]

            print(f"Highest intensity of m/z={ion} ({compound}) was at {round(scan_time, 2)} mins.")
            compounds[compound][ion] = scan_time
            # Plot every XIC as a separate graph
            axes[i].plot(data['Scan time (min)'], data[closest])
            axes[i].plot(scan_time, data[closest].iloc[data[closest].idxmax()], "o")
            axes[i].text(x=scan_time, y=data[closest].iloc[data[closest].idxmax()], s=f"{compound}, {ion}")
            axes[i].set_xlabel('Scan time (min)')
            axes[i].set_ylabel('intensity / a.u.')

    # Save in the folder plots/XICs
    plt.savefig(os.path.join(plot_path, path.split('/')[-1].replace('.mzml', '_XICs.png')), dpi=300)
    plt.close()

    # Save compounds to a .txt file
    compounds = pd.DataFrame.from_dict(compounds)
    print(compounds)

    compounds.to_csv(os.path.join(path, path.split('/')[-1].replace('.mzml', '_compounds.csv')))

    return compounds

            



# XXX: currently not used - delete? 
def plot_mass_spectra(path, data, compounds):
    '''
    Helper function for plotting the averaged mass spectra for debugging purposes. 
    '''
    print(data)
    spectra_dir = Path(path).parent.parent / 'plots' / 'ms_spectra'
    os.makedirs(spectra_dir, exist_ok=True)

    # Plotting for debugging purposes
    plt.figure(figsize=(15, 10))
    # Plot the found peaks on the data 
    plt.plot(data['m/z'], data['intensity / a.u.'])
    # From the ions list, find the closest m/z value in data['m/z] 
    # to each ion, and plot it on top of the data
    for compound, ions in compounds.items():
        for ion in ions:
            closest = data['m/z'].iloc[(np.abs(data['m/z'] - ion)).argmin()]
            label = f"{compound}, {ion}"
            plt.plot(closest, data['intensity / a.u.'].iloc[(np.abs(data['m/z'] - ion)).argmin()], "o")
            plt.text(x=closest, y=data['intensity / a.u.'].iloc[(np.abs(data['m/z'] - ion)).argmin()], s=label)
    plt.title(f'{os.path.basename(path)} average mass spectrum')
    plt.xlabel('m/z (Da)')
    plt.ylabel('intensity / a.u.')
    plt.savefig(os.path.join(spectra_dir, f'{os.path.basename(path)}_spectrum.png'), dpi=300)
    plt.close()


# TODO: Add all the ions and the possible neutral losses 
compounds = {
    'Asp': dict.fromkeys([304.1028, 258.0611]),
    'Glu': dict.fromkeys([318.1185, 282.0768]), 
    'IntStd': dict.fromkeys([332.1344, 296.0927]),
    'Asn': dict.fromkeys([303.1188, 257.0771]),
    'Ser': dict.fromkeys([276.1080, 230.0663]),
    'Gln': dict.fromkeys([317.1346, 281.0928])
}

path = '/Users/mateuszfido/Library/CloudStorage/OneDrive-ETHZurich/Mice/UPLC code/hplc/data/ms/STMIX5mM-pos.mzml'
# data, mz_axis, filename = load_ms1_data(path + 'STMIX5mM-pos.mzml')
#average_intensity(path)
# construct_xic(path)
#plot_mass_spectra(averaged, compounds, path)
#quantify_targeted(averaged, compounds, path)
annotate_lc_data(path, compounds)