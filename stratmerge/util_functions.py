import numpy as np
import xarray as xr
from typing import Dict, Union, List
from pathlib import Path
def read_ascii_grid(file_path: str) -> tuple:
    """
    Read data from an ASCII grid file and extract metadata and values.

    Parameters:
    ----------
    file_path : str
        Path to the ASCII grid file.

    Returns:
    -------
    data : numpy.ndarray
        A 2D NumPy array containing the data from the ASCII grid.
    metadata : dict
        A dictionary containing the metadata extracted from the file.
        It includes the following keys:
        - 'nrows': Number of rows in the grid.
        - 'ncols': Number of columns in the grid.
        - 'xllcorner': X-coordinate of the lower-left corner.
        - 'yllcorner': Y-coordinate of the lower-left corner.
        - 'cellsize': Size of grid cells.
        - 'nodata_value': Value indicating no data.
    x_coords : numpy.ndarray
        1D array of x-coordinate values based on cell size and lower left corner.
    y_coords : numpy.ndarray
        1D array of y-coordinate values based on cell size and lower left corner.
    """
    with open(file_path, 'r') as file:
        # Read the header information (number of columns, rows, and cell size)
        metadata = {}
        for _ in range(6):
            key, value = file.readline().split()
            metadata[key] = float(value)
        
        metadata['nrows'] = int(metadata['nrows'])
        metadata['ncols'] = int(metadata['ncols'])
        # Read the data into a NumPy array
        data = np.loadtxt(file, dtype=float, comments='NODATA')


    # Reshape the data to a 2D array
    data = data.reshape(metadata['nrows'], metadata['ncols'])

    # Create the x and y coordinate arrays based on the cell size and lower left corner
    x_coords = np.linspace(metadata['xllcorner'], metadata['xllcorner'] + metadata['cellsize'] * (metadata['ncols'] - 1), metadata['ncols'])
    y_coords = np.linspace(metadata['yllcorner'], metadata['yllcorner'] + metadata['cellsize'] * (metadata['nrows'] - 1), metadata['nrows'])

    return data,metadata, x_coords, y_coords

def asciigrid_to_datarray(filepath: str, 
                          name: str = 'data',
                          flip_ud = True,
                          nodata_is_nan: bool = True):
    """
    Convert ASCII grid file to an Xarray DataArray.
    
    Parameters:
    ----------
    filepath : str
        Path to the ASCII grid file.
    name : str, optional
        Name of the resulting DataArray. Default is 'data'.
    nodata_is_nan : bool, optional
        If True, replace nodata values with NaN. Default is True.
    
    Returns:
    -------
    data_array : xarray.core.dataarray.DataArray
        Xarray DataArray containing the data from the ASCII grid file.
    """
    data, metadata,_,_ = read_ascii_grid(filepath)
    if nodata_is_nan:
        data[data == metadata['NODATA_value']] = np.nan
        metadata['NODATA_value'] = np.nan
    
    coords={'x':np.arange(data.shape[0]),
            'y':np.arange(data.shape[1]),
            }
    if flip_ud:
        data = np.flipud(data)
    data_array = xr.DataArray(data, 
                         dims=list(coords.keys()), 
                         coords=coords, 
                         name=name
                         )
    data_array .attrs = metadata
    
    return data_array 

def write_ascii_grid(array: np.ndarray, meta_data: Dict[str, Union[int, float]], file_path: Union[str, Path]) -> None:
    """
    Write a 2D array as an ASCII grid file.
    
    Parameters:
        array (numpy.ndarray): The 2D array to be saved.
        meta_data (dict): Metadata for the ASCII grid containing the following keys:
            - 'ncols' (int): Number of columns in the grid.
            - 'nrows' (int): Number of rows in the grid.
            - 'xllcorner' (float): x-coordinate of the lower-left corner.
            - 'yllcorner' (float): y-coordinate of the lower-left corner.
            - 'cellsize' (float): Size of grid cells.
            - 'nodata_value' (float): Value indicating missing or nodata cells.
        file_path (str or Path): The file path where the ASCII grid will be saved.
    
    Returns:
        None
    """
    # Flatten the 2D array to a 1D array
    #flat_array = array.flatten()

    # Get the number of rows and columns from the original 2D array
    nrows, ncols = array.shape

    # Define the header for the ASCII grid
    header = f"ncols {meta_data['ncols']}\n" \
             f"nrows {meta_data['nrows']}\n" \
             f"xllcorner {meta_data['xllcorner']}\n" \
             f"yllcorner {meta_data['yllcorner']}\n" \
             f"cellsize {meta_data['cellsize']}\n" \
             f"NODATA_value {meta_data['nodata_value']}\n"

    # Save the flattened array as an ASCII grid
    np.savetxt(file_path, array, header=header, comments='', fmt='%.3e', delimiter=' ')
    
def harmonic_weight(property_array: np.ndarray, thickness_array: np.ndarray) -> np.ndarray:
    """
    Calculate harmonic weights for a property array and thickness array.

    Parameters:
        property_array (numpy.ndarray): Array of property values.
        thickness_array (numpy.ndarray): Array of layer thicknesses.

    Returns:
        numpy.ndarray: Array of harmonic weights.
    """
    return thickness_array / property_array

def arithmetic_weight(property_array: np.ndarray, thickness_array: np.ndarray) -> np.ndarray:
    """
    Calculate arithmetic weights for a property array and thickness array.
    
    Parameters:
        property_array (numpy.ndarray): Array of property values.
        thickness_array (numpy.ndarray): Array of layer thicknesses.
    
    Returns:
        numpy.ndarray: Array of arithmetic weights.
    """
    return property_array * thickness_array

def has_duplicates(lst: List) -> bool:
    """
    Check if a list contains duplicate elements.
    
    Parameters:
        lst (list): List to be checked for duplicates.
    
    Returns:
        bool: True if the list contains duplicates, False otherwise.
    """
    seen = set()
    for item in lst:
        if item in seen:
            return True
        seen.add(item)
    return False

def are_values_equal(dict1: Dict, dict2: Dict) -> bool:
    """
    Test whether all values of the same key are equal in two dictionaries.

    Parameters:
        dict1 (dict): First dictionary for comparison.
        dict2 (dict): Second dictionary for comparison.

    Returns:
        bool: True if all values of the same key are equal, False otherwise.
    """
    keys_in_common = set(dict1).intersection(dict2.keys())
    
    # Iterate through the keys and check if values are equal
    for key in keys_in_common:
        if dict1[key] != dict2[key]:
            #both can be nan
            if not np.isnan(dict1[key]) and np.isnan(dict2[key]):
                return False  # If any values are not equal, return False

    # If all values are equal for the same keys, return True
    return True