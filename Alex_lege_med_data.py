#%%
# coding=utf-8
# Brug cmd + i for at kører - ellers så virker pandas ikke

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

#Get the current working directory
# dir for MacBook
#dir = '/Users/Alex/Library/Mobile Documents/com~apple~CloudDocs/DTU/4. Semeter/Bachelor_project/Data/Cases_by_age.csv'

# dir for iMac
data_age = pd.read_csv('Data/Cases_by_age.csv',sep = ';')
data_time_pos = pd.read_csv('Data/Test_pos_over_time.csv',sep = ';')

# figure over new positive
plt.figure(1)
plt.subplot(2,1,1)
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
plt.xticks(rotation=45)
plt.subplots_adjust(bottom = 0.22)
plt.plot(data_time_pos.Date[range(len(data_time_pos.Date)-2)],data_time_pos.NewPositive[range(len(data_time_pos.Date)-2)],'r-')
plt.ylabel('# smittede')
plt.title('New positive cases pr. day')



# figure over cases by age
# plt.figure(2)
plt.subplot(2,1,2)
plt.plot(data_age.Aldersgruppe[range(len(data_age.Aldersgruppe)-1)], data_age.Antal_testede[range(len(data_age.Antal_testede)-1)],'bo-')
plt.plot(data_age.Aldersgruppe[range(len(data_age.Aldersgruppe)-1)], data_age.Antal_bekraftede_COVID_19[range(len(data_age.Antal_bekraftede_COVID_19)-1)], 'ro-')
plt.xlabel('Alder')
plt.ylabel('#personer')
plt.legend(['# Tester','# Bekræftede'])
plt.title('Testede i aldersgrupper')
plt.tight_layout()
plt.show()
