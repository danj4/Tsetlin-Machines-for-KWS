import os
import librosa
import librosa.display
import matplotlib
matplotlib.use('TkAgg')  # Choose an appropriate backend (e.g., 'TkAgg', 'Qt5Agg', 'WXAgg')
import matplotlib.pyplot as plt
import numpy as np
import wave

frame_duration = 0.16 
hop_duration = 0.03   

target_duration = 0.6

def normalize_matrix(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    range_val = max_val - min_val
    matrix = (matrix - min_val) / range_val
    return matrix

def threshold_matrix(matrix, threshold):
    thresholded_matrix = np.where(matrix > threshold, 1, 0)
    return thresholded_matrix

def percentile_binning(matrix, percentile=50):
    threshold_value = np.percentile(matrix, percentile)
    result_matrix = np.zeros_like(matrix)
    result_matrix[matrix >= threshold_value] = 1
    return result_matrix

def plot_waveform(word):
    dir_path = "./data/" + word + "/"
    print(dir_path)
    file_list=os.listdir(dir_path)
    with wave.open(dir_path + file_list[5], 'rb') as wav:
        framerate = wav.getframerate()
        nframes = wav.getnframes()
        duration = nframes / framerate
        audio_data = np.frombuffer(wav.readframes(nframes), dtype=np.int16)

    t = np.linspace(0, duration, num=len(audio_data))

    plt.figure(figsize=(10, 4))
    plt.plot(t, audio_data, color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Waveform')
    plt.grid(True)
    plt.show()

X_train=[]
Y_train=[]
#arr=['on' , 'off', 'stop', 'down', 'yes', 'no', 'left', 'right']
arr=['yes','no','stop','seven']
def smart_trim(y, sr):
    # y, sr = librosa.load(audio_file_path)

    l = 0
    r = len(y) - 1
    des_y_len = int(sr * target_duration) 

    while(r-l >= des_y_len):
        if(abs(y[l]) <= abs(y[r])):
            l+=1
        else:
            r-=1
    # audio_duration = librosa.get_duration(y=y[l:r], sr=sr)
    # print(audio_duration)
    # print(l, r)
    #plot_wave(y, sr)
    #plot_wave(y[l:r], sr)
    return y[l: r], sr

def plot_features(mfcc_features):
    fig, ax = plt.subplots(figsize=(10, 6))

    cax = ax.matshow(mfcc_features, cmap='viridis', origin='lower', aspect='auto')

    plt.colorbar(cax)

    ax.set_xlabel('Time Frames')
    ax.set_ylabel('MFCC Coefficients')

    plt.title('MFCC Features')
    plt.show(block = True)

def get_mfcc():
    for i in range(4):
        directory_path1="./data/"
        directory_path1+=arr[i]

        print(directory_path1)

        file_list=os.listdir(directory_path1)
        print(len(file_list))
        for file in file_list:
            audio_file_path = directory_path1+"/"+file

            y, sr = librosa.load(audio_file_path)

            y, sr = smart_trim(y, sr)

            des_y_len = int(target_duration * sr)
            if(len(y) < des_y_len):
                y = np.pad(y, (0, des_y_len - len(y)), 'constant')

            # padding to make length of input audio equal
            # audio_duration = librosa.get_duration(y=y, sr=sr)
            # if audio_duration > target_duration:
            #     y = y[:int(target_duration * sr)]
            # elif audio_duration < target_duration:
            #     y = np.pad(y, (0, int(target_duration * sr) - len(y)), 'constant')
            # sf.write(audio_file_path, y, sr)

            frame_length = int(sr * frame_duration)
            hop_length = int(sr * hop_duration)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=frame_length, hop_length=hop_length)
            mfcc_features = np.array(mfcc)
            # print(mfcc_features.shape)
        
            # mfcc_features = normalize_matrix(mfcc_features)
            # mfcc_features = threshold_matrix(mfcc_features,0)

            mfcc_features = percentile_binning(mfcc_features, 50)

            # print(mfcc_features.shape, sr)
            # plot_features(mfcc_features)

            # print(len(mfcc_features))
            X_train.append(mfcc_features)
            Y_train.append(i)

        
        # for x in X_train:
            # print(len(x[0]))
        X_trainn=np.array(X_train)
        Y_trainn=np.array(Y_train)
        np.save("X_trainn444.npy", X_trainn)
        np.save("Y_trainn444.npy", Y_trainn)

def analyze():
    X = np.load("X_trainny.npy")
    Y = np.load("Y_trainny.npy")

    X_flat = X.reshape(X.shape[0], -1) 
    print(len(X_flat[203]))

# get_mfcc()
# analyze()

# plot_waveform("seven")
def plot_wave(y, sr):
    # Calculate time values for x-axis
    duration = len(y) / sr
    t = np.linspace(0, duration, num=len(y))

    # Plot the waveform
    plt.figure(figsize=(10, 4))
    plt.plot(t, y, color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Waveform')
    plt.grid(True)
    plt.show()

audio_file_path = "./data/yes/0a7c2a8d_nohash_0.wav"

# smart_trim(audio_file_path)

get_mfcc()