module WSPR

using WAV, FFTW, DSP
using Plots

struct Spectrogram
    sample_rate::Unsigned
    frequency_resolution::AbstractFloat
    data
end

struct PowerSpectralDensity 
    sample_rate::Unsigned
    frequency_resolution::AbstractFloat
    data
end

struct Candidate 
    bin::Unsigned
    frequency::AbstractFloat
    snr::AbstractFloat
end

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

function spectrogram(x)
    NFFT = 512      # size of an FFT window for spectrogram calculation
    NSHIFT = NFFT/4 # shift the FFT window by this amount every iteration
    # number of FFT windows that fit into the data
    nwin::Int = (length(x) - NFFT)/NSHIFT
    # sine window used for PSD calculation
    w = sin.(pi/NFFT*(0:NFFT-1))
    # calculate windowed FFTs over 2 symbols and advance at 0.5 symbols
    psd = zeros(Complex{Float64}, NFFT, nwin+1)
    for n in 0:nwin
        m::Int = n*NSHIFT+1
        psd[:, n+1] = fft(x[m:m+NFFT-1] .* w) 
    end
    psd = abs2.(psd)
    # shift the PSD such that frequencies increase linearly with bin number
    Spectrogram(375, 375/NFFT, fftshift(psd, 1))
end

function spectrogram_from_wavfile(fname)
    samples = read_wavfile(fname)
    spectrogram(samples)
end

function psd_from_spectrogram(s::Spectrogram)
    # sum up all calculated power spectra
    x = sum(s.data, dims=2)
    # smooth the power spectrum using a rectangular window of 7 samples 
    # (centered moving average) and truncate output signal to frequency
    # range of interest (1500 Hz +/- 150 Hz)  
    # bins of interest are at 51 to 462, but need to take into account
    # 3 sample shift introduced by the convolution
    PowerSpectralDensity(s.sample_rate, s.frequency_resolution, conv(vec(x), ones(7))[54:464])
end

function find_candidates(s::PowerSpectralDensity)
    # estimate then noise level as the 123/411 = 30th percentile
    noise_level = sort(s.data)[123]
    min_snr = 10^-0.8
    snr_scaling_factor = 26.3
    # rescale such that peaks directly represent SNR
    x = s.data ./ noise_level .- 1.0
    # set values below minimum SNR to a low value so they don't cause false peak detections
    x = map(v -> if v < min_snr 0.1*min_snr else v end, x)
    # simple peak finding algorithm, store bin index, frequency and SNR
    candidates = []
    for i in 2:length(x)-1
        if x[i] > x[i-1] && x[i] > x[i+1] && length(candidates)<200
            push!(candidates, 
                  Candidate(i,(i-205)*s.frequency_resolution,10*log10(x[i])-snr_scaling_factor ))
        end
    end
    # discard peaks outside the band of interest
    filter!(v -> abs(v.frequency) < 110, candidates)
    # sort the found peaks by SNR from high to low
    sort(candidates, by = v -> v.snr, rev = true)
end

function sim() 
    spec = spectrogram_from_wavfile("150426_0918.wav")
    psd = psd_from_spectrogram(spec)
    candidates = find_candidates(psd)
    plot(psd.data)
    bins = [c.bin for c in candidates]
    scatter!(bins, psd.data[bins])
end

end