#%% Load iin data

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



plt.figure(1)
plt.plot(D_O_T.Dato[0:-1], D_O_T.Antal_døde[0:-1])
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
plt.xticks(rotation=45)
plt.title('Deaths per day')
plt.ylabel('Number of deaths')
plt.show()

