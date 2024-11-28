import matplotlib.pyplot as plt
import numpy as np

# pointing error calculation
def calculate_pointing_error(set_azimuth, set_elevation, azimuth, elevation):
    """
    Calculate the pointing error given reference (set) and measured azimuth and elevation.
    
    Parameters:
        set_azimuth (array-like): Reference azimuth values in degrees.
        set_elevation (array-like): Reference elevation values in degrees.
        azimuth (array-like): Measured azimuth values in degrees.
        elevation (array-like): Measured elevation values in degrees.
    
    Returns:
        np.ndarray: Pointing error for each pair of reference and measured angles in degrees.
    """
    # Convert degrees to radians
    set_azimuth_rad = np.radians(set_azimuth)
    set_elevation_rad = np.radians(set_elevation)
    azimuth_rad = np.radians(azimuth)
    elevation_rad = np.radians(elevation)
    
    # Compute the pointing error using the great-circle distance formula
    pointing_error_rad = np.arccos(
        np.sin(set_elevation_rad) * np.sin(elevation_rad) +
        np.cos(set_elevation_rad) * np.cos(elevation_rad) * np.cos(azimuth_rad - set_azimuth_rad)
    )
    
    # Convert the pointing error back to degrees
    pointing_error_deg = np.degrees(pointing_error_rad)
    return pointing_error_deg

def remove_outlier(obj):
    outlier_indices = []
    valid_range_bins = int(2 * 11e3 / (250e3 / obj.noise_obj.fft_size))  # Total valid bins for the maximum doppler shift of Â±11kHz when sampling rate is 250kHz
    for i , row in enumerate(obj.noise_obj.waterfall):
        fft_data =  10*np.log10(np.squeeze(row))  # FFT data (magnitude spectrum)
        center_bin = obj.noise_obj.fft_size // 2  # Center bin of the FFT, integer division to get an actual number

        # Determine valid bin range for Doppler effect
        valid_min = max(0, center_bin - valid_range_bins // 2)
        valid_max = min(obj.noise_obj.fft_size, center_bin + valid_range_bins // 2)

        # Extract FFT values in the valid range
        valid_fft_values = fft_data[valid_min:valid_max]

        # Find the maximum value in the valid range
        peak_value = np.max(valid_fft_values)


        if (peak_value-np.mean(fft_data)>5):
            outlier_indices.append(i) #If the peak is gucci, then we keep it

    return obj[outlier_indices]#Basically deleting all the points where it was not transmitting


def prepmovavgfunc(obj, window_size=199):

    #first outlier removal using threshold
    idx = np.argwhere(10*np.log10(obj.noise_obj.signal_abs) > -138) #Threshold set to -138 dBm on the data without the FSPL correction
    obj=obj[idx] #Basically deleting all the points where the moving average is below the threshold
    #second outlier removal using peak detection
    obj=remove_outlier(obj)
    
    # Calculate the target values
    clean_sig_abs = np.squeeze(obj.clean_sig_abs)
    # window_size = 99   # Window size for moving average
    padded_signal = np.pad(clean_sig_abs, pad_width=window_size // 2, mode='reflect')
    moving_avg = np.convolve(padded_signal, np.ones(window_size) / window_size, mode='valid')
    time_ax_avg = obj.time_ax[:len(moving_avg)]

    # obj_act[i]=obj #delimit the object
    obj.target = moving_avg

    # Calculate pointing error for this object
    pointing_error = calculate_pointing_error(
        set_azimuth=obj.station_obj.set_azimuth,
        set_elevation=obj.station_obj.set_elevation,
        azimuth=obj.station_obj.azimuth,
        elevation=obj.station_obj.elevation
    )
    # Append the pointing error as a new attribute to the object
    obj.pointing_error = pointing_error

    return obj