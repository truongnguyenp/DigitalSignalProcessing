import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np


# find dips that match threshold condition
def find_dips(arr):
    # Parameter: an AMDF array
    # return an array of dips that has value <= THRESHOLD (it's mean that the signal is VOICED) ELSE return an array of zero (the signal is UNVOICED)
    return np.array([i for i in range(1, arr.size - 1) if arr[i - 1] > arr[i] and arr[i + 1] > arr[i] and arr[i] <= AMDF_THRESHOLD]) if arr.size > 2 else np.zeros(0)

# median filter to fill the virtual pitch
def median_filter_ver2(data, filter_size = 5):
    # Parameter: a array of pitch, filter size
    # return array of pitch after using median filter to fill out virtual pitch
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data)))
    

    for j in range(len(data)):

        for z in range(filter_size):
            if j + z - indexer < 0 or j + z - indexer > len(data) - 1:
                for c in range(filter_size):
                        temp.append(0)
            else:
                if j + z - indexer < 0 or j + indexer > len(data) - 1:
                    temp.append(0)
                else:
                    for k in range(filter_size):
                        temp.append(data[j + k - indexer])

        temp.sort()
        data_final[j] = temp[len(temp) // 2]
        temp = []

    return data_final

# median filter to fill the virtual pitch
def median_filter(arr, filter_size =5):
    # Parameter: a array of pitch
    # return array of pitch after using median filter to fill out virtual pitch
    arr = np.concatenate(([0], arr, [0]))
    indexes = []

    # each pitch is not shift too much from their neighbour pitch
    for i in range(1, arr.size - 1):
        if abs(arr[i] - arr[i - 1]) <= PITCH_SHIFT and abs(arr[i] - arr[i + 1]) <= PITCH_SHIFT:
            indexes.append(i)

    arr = arr[indexes]

    for i in range(filter_size, arr.size - filter_size):
        if (arr[i] == 0): continue
        else: arr[i] = np.mean(arr[i - filter_size:i + filter_size])

    return arr

def amdf(frame, max_period_in_sample, min_period_in_sample):
    #Parameter: frame:(a fration of input signal), max period, min period
    # return two value (d,dips[...]) with d is the AMDF frame of any signal fragment, dips[...] is the index of lowest local dip
    d = np.zeros(max_period_in_sample + 1)
    # 
    if frame.size >= max_period_in_sample:
        for n in range(min_period_in_sample, max_period_in_sample + 1):
            d[n] = np.sum(abs(frame[:frame.size - n] -
                          frame[n:]) / (frame.size - n))
    else: return d,-1

    # Find position of local dips in frame
    dips = find_dips(d)

    # 
    if dips.size == 0:
        return d, -1
    # d[dips] contain value of local dips
    # use np.argmin to find index of the lowest dip
    minimum_local_dip_pos = np.argmin(d[dips])

    return d, dips[minimum_local_dip_pos]


def compute(signal_name):
    print(f'Computing {signal_name}')

    Fs, signal = wavfile.read(f'{PATH}{signal_name}.wav')
    #normalized signal by it max abs value
    signal = signal / max(np.max(signal), abs(np.min(signal)))

    # read lab result attribute
    with open(f'{PATH}{signal_name}.lab', 'r') as file_lab:
        F0_mean_lab = float(file_lab.readlines()[-2].split()[1])

    with open(f'{PATH}{signal_name}.lab', 'r') as file_lab:
        F0_std_lab = float(file_lab.readlines()[-1].split()[1])

    frame_length_in_sample = int(FRAME_LENGTH_IN_SECOND * Fs)
    max_period_in_sample= Fs // MIN_PITCH_VALUE
    min_period_in_sample = Fs // MAX_PITCH_VALUE

    pitchs = np.zeros(0)
    pitch_to_plot = np.zeros(0)

    amdf_voiced_frame = []
    amdf_voiced_signal = []
    amdf_unvoiced_frame = []
    amdf_unvoiced_signal = []

    for i in range(0, signal.size, frame_length_in_sample):
        AMDF_Frame, T0_in_sample = amdf(
            signal[i:i + frame_length_in_sample], max_period_in_sample, min_period_in_sample)
            
        if T0_in_sample != -1:

            pitchs = np.append(pitchs, [Fs / T0_in_sample])
            pitch_to_plot = np.append(pitch_to_plot, [Fs / T0_in_sample])
            dips = find_dips(AMDF_Frame)


            minimum_local_dip_pos = np.argmin(AMDF_Frame[dips])

            # if a frame has minimum local dip that <= threshold so it is a voiced frame else: unvoiced
            if minimum_local_dip_pos: 
                amdf_voiced_frame= AMDF_Frame
                amdf_voiced_signal = signal[i:i + frame_length_in_sample]
            else:
                 amdf_unvoiced_frame=AMDF_Frame 
                 amdf_unvoiced_signal = signal[i:i + frame_length_in_sample]
                 
            
        else:
            pitch_to_plot = np.append(pitch_to_plot, np.nan)


    pitchs = median_filter(pitchs)
    
    # Use built-in mean and std to cal F0 mean,std and difference from lab result
    F0_mean = np.round(np.mean(pitchs))
    F0_std=np.round(np.std(pitchs))
    diff_mean = np.round(abs(F0_mean-F0_mean_lab))
    diff_std = np.round(abs(F0_std- F0_std_lab))
    data_export.append((signal_name, F0_mean, F0_mean_lab, F0_std, F0_std_lab))
    
    # Initialise the subplot 
    figure, axis = plt.subplots(4)
    plt.figure(figure)
    plt.title(signal_name)

    #F0 contour plot
    axis[0].set_ylim(0,400)
    axis[0].plot(pitch_to_plot,'.')
    axis[0].set_title(
        f"F0 contour")

    # Signal
    axis[1].plot(signal)
    axis[1].set_title(f"{signal_name}.wav , F0_mean: {F0_mean} ({diff_mean}), F0_std: {F0_std} ({diff_std})")

    # AMDF of voiced frame
    axis[2].plot(amdf_voiced_frame)
    axis[2].set_title("AMDF of a voiced frame in audio")

    # AMDF of unvoiced frame
    axis[3].plot(amdf_unvoiced_frame)
    axis[3].set_title("AMDF of a unvoiced frame in audio")

    # Add padding for plots
    plt.tight_layout()

    print(f'Compute for {signal_name} is done !')


if __name__ == '__main__':

    # LIST_OF_SIGNAL=['phone_F2', 'phone_M2','studio_F2', 'studio_M2']  # HuanLuyen
    # PATH='./TinHieuHuanLuyen/'  # HuanLuyen

    LIST_OF_SIGNAL = ['phone_F1', 'phone_M1', 'studio_F1', 'studio_M1'] # Kiem thu
    PATH = './TinHieuKiemThu/' #KiemThu

    TIME_FRAME=0.03
    AMDF_THRESHOLD=0.4325
    FRAME_LENGTH_IN_SECOND=0.025
    MAX_PITCH_VALUE=400
    MIN_PITCH_VALUE=70
    PITCH_SHIFT=10

    # stores the neccessary data of each audio
    data_export=[]

    # loop through each audio file to estimate pitch 
    for signal in LIST_OF_SIGNAL:
        compute(signal)

    plt.show()

    # Print table of data
    print(f'{"File":^10}{"Type":^10}{"AMDF (Hz)":^10}{"Lab (Hz)":^10} {"Relative error (%)":^10}')

    for name, F0_mean, F0_mean_lab, F0_std, F0_std_lab in data_export:
        print('----------------------------------------------------------------')
        print(f'{name:^10} {"F0 mean":10} {F0_mean:^10.1f} {F0_mean_lab:^10.1f} {abs(F0_mean-F0_mean_lab)*100/F0_mean_lab:^10.2f}')
        print(f'{"":10} {"F0 std":10} {F0_std:^10.1f} {F0_std_lab:^10.1f} {abs(F0_std-F0_std_lab)*100/F0_std_lab:^10.2f}')
        print('----------------------------------------------------------------')
