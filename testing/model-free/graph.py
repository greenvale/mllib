import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import data
stateValuePd = pd.read_csv(r'stateValue.csv')
stateValue = stateValuePd.to_numpy()

numEpisodes = stateValue.shape[0]
numStates = stateValue.shape[1]

# Plots
xEpisode = np.arange(0, numEpisodes)

plt.plot(xEpisode, stateValue);
plt.show()
