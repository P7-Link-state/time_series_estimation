import matplotlib.pyplot as plt
import numpy as np
import noisy_moose as nm
import pickle
import os
from multiprocessing import Pool
import datetime

# data_files_active = os.listdir("../output/VZLUSAT/water/")

# vzlusat_active = list(i for i in data_files_active if "VZLUSAT-2" in i)


# obj_act: list[nm.prepping] = [] #: list[nm.prepping] = [] means that the variable is a list of nm.prepping objects, which makes it possible to autocomplete the methods and attributes of the nm.prepping class.C:\Users\StoreElberg\OneDrive - Aalborg Universitet\Git3\output\VZLUSAT

# for i in vzlusat_active[:10]: #choose how many to load
#     with open("../output/VZLUSAT/water/" + i, 'rb') as f:
#         obj_act.append(pickle.loads(f.read()))

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

def remove_outlier(obj,peak_level=5):
    outlier_indices = []
    # valid_range_bins = int(2 * 11e3 / (250e3 / obj.noise_obj.fft_size))  # Total valid bins for the maximum doppler shift of Â±11kHz when sampling rate is 250kHz
    signal_bandwidth =6e3 #evaluated from fft
    # print(f"Filled bins {np.round(signal_bandwidth/(250e3/obj.noise_obj.fft_size))}")
    valid_range_bins=np.round(signal_bandwidth/(250e3/obj.noise_obj.fft_size))
    
    sp=np.zeros(len(obj.noise_obj.waterfall))
    noipow=np.zeros(len(obj.noise_obj.waterfall))
    #initialise speed
    speed=(obj.station_obj.dist[1]-obj.station_obj.dist[0])/(obj.time_ax[1]-obj.time_ax[0])

    for i , row in enumerate(obj.noise_obj.waterfall):
        if i !=0:
            speed = (obj.station_obj.dist[i]-obj.station_obj.dist[i-1])/(obj.time_ax[i]-obj.time_ax[i-1])
            # print(f"startspeed {startspeed}")

        dopplershift=-((speed)/3e8)*obj.station_obj._freq
        # print(f"Doppler shift {dopplershift}")
        center_bin = np.round(obj.noise_obj.fft_size/2) + np.round((dopplershift) / (250e3 / obj.noise_obj.fft_size)) # center freq is the 17th index.
        # print(f"center freq {center_bin}")


        fft_data =  10*np.log10(np.squeeze(row))  # FFT data (magnitude spectrum)
        # center_bin = obj.noise_obj.fft_size // 2  # Center bin of the FFT, integer division to get an actual number

        # Determine valid bin range for Doppler effect
        valid_min = int(center_bin - (valid_range_bins // 2))
        valid_max = int(center_bin + (valid_range_bins // 2)) + 1 # +1 to include the last bin
        # print(f"valid min {valid_min} valid max {valid_max}")

        # Extract FFT values in the valid range
        valid_fft_values = fft_data[valid_min:valid_max]
        invalid_fft_values = np.concatenate((fft_data[:valid_min], fft_data[valid_max:]))
        # Find the maximum value in the valid range
        signal_power = np.mean(valid_fft_values)
        noise_power=np.mean(invalid_fft_values)
        sp[i]=signal_power
        noipow[i]=noise_power

        # plt.figure()
        # plt.scatter(obj.noise_obj.freqs[valid_min:valid_max],valid_fft_values,label="Signal power to noise power")
        # plt.scatter(np.concatenate((obj.noise_obj.freqs[:valid_min], obj.noise_obj.freqs[valid_max:])),invalid_fft_values,label="Signal power to noise power")
        # plt.show()

        if (signal_power-noise_power>peak_level):
            outlier_indices.append(i) #If the peak is gucci, then we keep it
        
    
    # plt.figure()
    # plt.scatter(obj.time_ax,sp-noipow,label="Signal power to noise power",alpha=0.2)
    # plt.show()
    # plt.figure()
    # plt.scatter(obj.time_ax,noipow,label="Noise power")
    # # plt.scatter(obj.time_ax,sp-noipow,label="Signal power to noise power")
    # plt.show()

    # print(f"outlier_indices {outlier_indices}")
    return obj[outlier_indices]#Basically deleting all the points where it was not transmitting



# print(f"Length of list {len(obj_act)}")
def causal_moving_average_fast(data, window_size):
    cumsum = np.cumsum(np.insert(data, 0, 0))  # Cumulative sum
    moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    return np.concatenate((data[:window_size-1], moving_avg))


def prepmovavgfunc(obj,signal_threshold=-138,   peak_level=5, noise_threshold = -170 ,  window_size=99):
    #first outlier removal using threshold
    idx = np.argwhere(10*np.log10(obj.noise_obj.signal_abs) > signal_threshold) #Threshold set to -138 dBm on the data without the FSPL correction
    # print(f"idx signal {idx}")
    obj=obj[idx.flatten()] #Basically deleting all the points where the signal is below the threshold

    #second outlier removal using peak detection and doppler shift
    obj=remove_outlier(obj, peak_level) #can be commented out if not wanted, it does not change that much

    # print(10*np.log10(obj.noise_obj.noise))
    # Third outlier removal using noise power
    # Flatten noise to 1D for threshold comparison
    idx = np.argwhere((10 * np.log10(obj.noise_obj.noise.flatten())) < noise_threshold)

    # print(f"idx noise {idx}")
    obj=obj[idx.flatten()] #Basically deleting all the points where the noise is above the threshold
    # print(f"Length of signal after noise threshold {obj.clean_sig_abs.shape}")

    # Remove outliers from the signal a bit like cheating, i got a better idea in Create X and Y file
    # window_size = 99
    # moving_avg= causal_moving_average_fast(np.squeeze(obj.clean_sig_abs), window_size)

    # squared_errors = (clean_sig_abs - moving_avg) ** 2  # Compute squared errors

    # # removing outliers
    # obj=obj[np.where(squared_errors < 5 )[0]]

    # Calculate the target
    clean_sig_abs = np.squeeze(obj.clean_sig_abs)
    padded_signal = np.pad(clean_sig_abs, pad_width=window_size // 2, mode='reflect')
    moving_avg = np.convolve(padded_signal, np.ones(window_size) / window_size, mode='valid')
    # time_ax_avg = obj.time_ax[:len(moving_avg)]

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