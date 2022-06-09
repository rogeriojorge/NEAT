import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
from mpl_toolkits import mplot3d
df= pd.read_csv('output_2.txt', sep=' ',header=None, skiprows=4)
df.columns = ["t", "S", "theta", "zeta", "vpar", "Pphi/e", "Eperp/Ealfven", "Epar/Ealfven", "R", "phi", "Z"]
fig = plt.subplots()
ax=plt.axes(projection='3d')
ax.plot3D(df['R']*np.cos(df['phi']),df['R']*np.sin(df['phi']),df['Z'], label='Particle 1')
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.xlabel('X')
plt.ylabel('Y')
#plt.zlabel('Z')
plt.title('First particle tracing')
plt.legend()
plt.show()
