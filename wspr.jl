module WSPR

using WAV, DSP
using Plots

using FFTW: fftshift

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

# half-sine window function used for periodogram and spectrogram calculation
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
    # first shift the spectrogram such that DC is in the center 
    x = fftshift(p.power)
    f = fftshift(p.freq)
    # smooth the resulting periodogram using a rectangular window of 7 bins
    # and select the central part of the convolution
    x = conv(ones(7), x)[(1:length(x)).+3]
    # estimate the noise level as the 123/411 = 30th percentile
    noise_level = sort(x)[123]
    min_snr = 10^-0.8
    snr_scaling_factor = 26.3
    # rescale such that peaks directly represent snr
    x .= x ./ noise_level .- 1
    # set values below minimum SNR to a small value so they don't cause 
    # false peak detections
    x[findall(v -> v < min_snr, x)] .= 0.1*min_snr
    # simple peak finding algorithm, store bin index, frequency and SNR
    candidates = []
    for i in 2:length(x)-1
        if x[i] > x[i-1] && x[i] > x[i+1] && length(candidates)<200
            push!(candidates, 
                  Candidate(i, f[i], 10*log10(x[i])-snr_scaling_factor ))
        end
    end
    # discard peaks outside the band of interest
    filter!(v -> abs(v.freq) < fmax, candidates)
    # sort the found peaks by SNR from high to low
    sort(candidates, by = v -> v.snr, rev = true)
end

function coarse_sync_metric(s::Matrix, f0, t0, drift)
    # time index i.e. columns in spectrogram matrix
    t = filter(x -> 0 < x <= size(s)[2], t0 .+ (0:2:161))
    # bin index i.e. rows in spectrogram matrix 
    f = Int.(round.(f0 .+ drift.*(t.-81)./81))

    p = zeros(Float64, 4, length(t))
    for (m, x) in enumerate(-3:2:3)
        p[m, :] = [s[r,c] for (r, c) in zip(f .+ x, t)]
    end
    
    p .= sqrt.(p)

    ss = reduce(+, (2*p[4,:].-1).*((p[2,:].+p[4,:]).-(p[1,:].+p[3,:])))
    pow = reduce(+, reduce(+, p))
    ss/pow
end

function coarse_sync(candidates, s)
    dmax = 4
    df = s.freq[2]

    # fftshift to get DC into the middle
    freq = fftshift(s.freq)
    power = fftshift(s.power, 1)
    
    best_parameters = [] 
    for c in candidates
        smax = -1e30
        params = []
        # perform search across
        # 1. starting bin b
        # 2. starting time t0
        # 3. drift d 
        for b = (c.bin-2):(c.bin+2), t0 = -10:21, d = -dmax:dmax 
            m = d/2/df
            sync = coarse_sync_metric(power, b, t0, m)
            if sync > smax 
                push!(params, (timing=128*(t0+1), drift=d, bin=b, sync=sync))
                smax = sync
            end
        end

        sort!(params, by=v->v.sync, rev=true)
        push!(best_parameters, params[1])
    end
    best_parameters
end

function get_stuff() 
    samples = read_wavfile("150426_0918.wav")

    spectrogram = Periodograms.spectrogram(samples, 512, 384, fs=375, window=window_func)

    candidates = find_candidates(samples, fmax=150)

    coarse_parameters = coarse_sync(candidates, spectrogram)

    samples, spectrogram, candidates, periodogram, coarse_parameters
end

function sim() 
    pyplot()
    samples = read_wavfile("150426_0918.wav")

    spectrogram = Periodograms.spectrogram(samples, 512, 384, fs=375, window=window_func)

    candidates = find_candidates(samples, fmax=150)

    coarse_parameters = coarse_sync(candidates, spectrogram)

    plot(fftshift(spectrogram.freq), fftshift(periodogram))
    bins = [c.bin for c in candidates]
    scatter!(spectrogram.freq[bins], periodogram[bins])

    #heatmap(10*log10.(spectrogram.power))
end

end
