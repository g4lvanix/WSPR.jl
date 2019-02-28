module WSPR

using WAV, FFTW, DSP
using Plots

struct Candidate 
    bin::Unsigned
    freq::AbstractFloat
    snr::AbstractFloat
end

"""
    read_wavfile(fname)

    Read a WAV file sampled at 12 ksps and decimate it by 32 using an FFT algorithm
"""
function read_wavfile(fname)
    (y, fs) = wavread(fname)
    # taken from wsprd.c, number of samples to return from this function
    nfft2 = 46080
    # derived variables
    nfft1::Int = 32*nfft2   # size of the first FFT, decimate by a factor of 32
    df = fs/nfft1           # frequency resolution of the first FFT
    i0 = 1500/df            # bin index for f = 1500 Hz 
    npoints::Int = 114*fs   # 114 seconds worth of input samples
    # select npoints number of input samples, zero pad to length nfft1 and perform FFT
    fd = fft(cat(y[1:npoints], zeros(nfft1-npoints), dims=1))
    # calculate the relevant FFT bins in order to truncate the spectrum
    # around 1500 Hz (bandpass filtering in the frequency domain)
    fstart::Int = i0+1 - nfft2/2
    fstop::Int = i0 + nfft2/2
    # truncate the input spectrum to the frequency range of interest --> bandpass filtering
    # and use IFFT to get back to the time domain --> decimation (downsampling)
    ifft(fftshift(fd[fstart:fstop]))/1000
end

window_func(n) = sin.(pi/n*(0:n-1))

"""
    find_candidates(s::Vector; fmax=110)

    Return vector of candidate signals for WSPR demodulation in a bandwidth of +/- fmax Hz
    
    First a periodogram of the signal is calculated using Welch's method in order to 
    estimate the power spectral density (PSD) of the input signal. The periodogram is 
    smoothed using a moving average, the noise level is estimated and a simple peak 
    finding algorithm is used to detect possible candidate signals.  
"""
function find_candidates(s::Vector; fmax=110)
    # create a periodogram using Welch's method to estimate power spectral density 
    p = Periodograms.welch_pgram(s, 512, 384, fs=375, window=window_func)
    # smooth the resulting periodogram using a rectangular window of 7 bins
    x = filt(ones(7), 1, p.power)
    # estimate the noise level as the 123/411 = 30th percentile
    noise_level = sort(x)[123]
    min_snr = 10^-0.8
    snr_scaling_factor = 26.3
    # rescale such that peaks directly represent snr
    x .= x ./ noise_level .- 1
    # set values below minimum SNR to a small value so they don't cause false peak detections
    x[findall(v -> v < min_snr, x)] .= 0.1*min_snr
    # simple peak finding algorithm, store bin index, frequency and SNR
    candidates = []
    for i in 2:length(x)-1
        if x[i] > x[i-1] && x[i] > x[i+1] && length(candidates)<200
            push!(candidates, 
                  Candidate(i, p.freq[i], 10*log10(x[i])-snr_scaling_factor ))
        end
    end
    # discard peaks outside the band of interest
    filter!(v -> abs(v.freq) < fmax, candidates)
    # sort the found peaks by SNR from high to low
    (sort(candidates, by = v -> v.snr, rev = true), x)
end

function sim() 
    pyplot()
    samples = read_wavfile("150426_0918.wav")

    spectrogram = Periodograms.spectrogram(samples, 512, 384, fs=375, window=window_func)

    candidates, periodogram = find_candidates(samples, fmax=150)

    plot(spectrogram.freq, periodogram)
    bins = [c.bin for c in candidates]
    scatter!(spectrogram.freq[bins], periodogram[bins])

    #heatmap(10*log10.(spectrogram.power))
end

end