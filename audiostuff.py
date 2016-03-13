from scipy.signal import hann
from scipy.fftpack import fft
from librosa import load
import scipy as sp
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('/usr/local/lib/python2.7/site-packages')


def save_spectrogram(path_to_audio, name):
    clip, sr = load(path_to_audio)
    window_size = 30
    hop_size = 15
    X = stft(clip, window_size, hop_size)
    plt_spectrogram(X, window_size, hop_size, sr, './static/img/spec' + name + '.png')

def plt_spectrogram(X,win_length, hop_size, sample_rate, path_to_image, zoom_x=None, zoom_y=None,tick_labels='time-freq'):
    """
    Plots the log magnitude spectrogram.

    Input Parameters:
    ------------------
    X: 2D complex numpy array containing the stft values. Rows correspond to frequency bins and columns to time frames.
    win_length: the length of the analysis window
    hop_size: the hop size between adjacent windows
    sample_rate: sampling frequency
    tick_labels: the type of x and y tick labels, there are two options:
                 'time-freq': shows times (sec) on the x-axis and frequency (Hz) on the y-axis (default)
                 'bin-frame': shows time frame number on the x-axis and frequency bin number on the y-axis

    zoom_x: 1 by 2 numpy array containing the range of values on the x-axis, e.g. zoom_t=np.array([x_start,x_end])
    zoom_y: 1 by 2 numpy array containing the range of values on the y-axis, e.g. zoom_f=np.array([y_start,y_end])


    Returns:
    ---------
    times: 1D real numpy array containing time instances corresponding to stft frames
    freqs: 1D real numpy array containing frequencies of analyasis up to Nyquist rate
    2D plot of the magnitude spectrogram
    """

    # Find the size of stft
    Nf,Nt=np.shape(X)

    # Compute the log magnitude spectrogram
    X=20*np.log10(np.abs(X))

    # Extract the first half of the spectrum for each time frame
    X=X[0:Nf/2]
    Nf=np.shape(X)[0]

    # Generate time vector for plotting
    times=(hop_size/float(sample_rate))*np.arange(Nt)

    # Generate frequency vector for plotting
    freqs=(float(sample_rate)/win_length)*np.arange(Nf)

    # Generate time and frequency matrices for pcolormesh
    times_matrix,freqs_matrix=np.meshgrid(times,freqs)

    # Plot the log magnitude spectrogram
    plt.title('Log magnitude spectrogram')
    if tick_labels == 'bin-frame':
        plt.pcolormesh(X)
        plt.xlabel('Time-frame Number')
        plt.ylabel('Frequency-bin Number')
    else:
        plt.pcolormesh(times_matrix,freqs_matrix,X)
        plt.xlabel('Time (sec)')
        plt.ylabel('Frequency (Hz)')
        plt.savefig(path_to_image)

    # Zoom in on the plot if specified
    if zoom_x is None and zoom_y is None:
        plt.axis('tight')

    if zoom_x is not None:
        plt.xlim(zoom_x)

    if zoom_y is not None:
        plt.ylim(zoom_y)

    return X

def stft(signal, window_size, hop_size, window_type = 'hann'):
    """
    Computes the short term fourier transform of a 1-D numpy array, where the array
    is windowed into a set of subarrays, each of length window_size. The distance between
    window centers (in samples) is given by hop_size. The type of window applied is
    determined by window_type. This returns a 2-D numpy array where the ith column
    is the FFT of the ith window. Each column contains an array of complex values.

    Input Parameters
    ----------------
    signal: The 1-d (complex or real) numpy array containing the signal
    window_size: an integer scalar specifying the number of samples in a window
    hop_size: an integer specifying the number of samples between the start of adjacent windows
    window_type: a string specifying one of two "hann" or "rectangular"

    Returns
    -------
    a 2D numpy array of complex numbers where the array column is the FFT of the ith window,
    and the jth element in the ith column is the jth frequency of analysis.
    """

    # figure out how many hops
    length_to_cover_with_hops = len(signal) - window_size;
    assert (length_to_cover_with_hops >= 0), "window_size cannot be longer than the signal to be windowed"
    num_hops = 1 + length_to_cover_with_hops/hop_size;

    # make our window function
    if (window_type == 'hann'):
        window = sp.signal.hann(window_size, sym=False)
    else:
        window = np.ones(window_size)

    stft = [0]*num_hops
    # fill the array with values
    for hop in range(num_hops):
        start = hop*hop_size
        end = start + window_size
        unwindowed_sound = signal[start:end]
        windowed_sound =  unwindowed_sound * window
        stft[hop]= fft(windowed_sound, window_size)
    return np.array(stft).T
