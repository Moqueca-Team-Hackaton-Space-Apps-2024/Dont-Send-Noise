
# Algorithm Don't Send Noise (DSN) part of Hackathon NASA Space Apps 2024

This repository includes the code created by the Team Moqueca Capixaba developed during the Hackathon.

The Algorithm DSN starts by identifying the dominant frequencies of the signal analyzed, from which it filters the noise of unwanted frequencies by selecting only the regions close to the peaks of highest amplitude (Module of Fourier Transform values). Next it transforms the filtered values back to the original time-velocity space and continues calculating the spectrogram of the results. Finally, it calculates the maximum power spectrogram density (PSD) point to obtain the starting time of the seismic event.
