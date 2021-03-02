#%% Load iin data
#%% Load in data

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from Alex_lege_med_data import data_age, data_time_pos

D_O_T = pd.read_csv('Data/Deaths_over_time.csv',sep = ';')
Cases_sex = pd.read_csv('Data/Cases_by_sex.csv', sep = ';')


plt.figure(1)
plt.plot(D_O_T.Dato[0:-1], D_O_T.Antal_døde[0:-1])
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
plt.xticks(rotation=45)
plt.title('Deaths per day')
plt.ylabel('Number of deaths')
plt.show()

