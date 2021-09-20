import os
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Let's get some data
for directory in os.walk("Audio"):
    # Ignore empties
    if directory[2] == []: continue

    word = ""

    for fileName in directory[2]:
        # Open the wav file
        if fileName.split(".")[1] != "wav": continue
        rate, data = wavfile.read(directory[0] + "\\\\" + fileName)
        plt.specgram(data, Fs=rate)
        plt.axis("off")
        plt.savefig("Spectrograms/" + directory[0].split("\\")[1] + "/" + fileName.replace(".wav", ".png"), bbox_inches="tight", pad_inches = 0)

