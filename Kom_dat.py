#%%

from numpy.lib.function_base import append
import pandas as pd

all_dat = pd.read_csv('Kommune_data/dat.csv', sep=';')

temp = []
temp2 = []
mat_num_people = []
travel_in = []
population = []
ib = all_dat.Bopæl[0]

#Selective cities
num_city = 3
travel_in_2 = []
population_2 = []



k = 0
for i in all_dat.index:
    bo = all_dat.Bopæl[i]

    if bo != ib:
        mat_num_people.append(temp)

        temp = []
        ib = bo
        k += 1
    
    temp.append(all_dat.Mænd[i] + all_dat.Kvinder[i])


mat_num_people.append(temp)





#%% Remove 
mat_num_people.pop(0)
mat_num_people.pop(1-1)
mat_num_people.pop(6-2)
mat_num_people.pop(20-3)
mat_num_people.pop(32-4)
mat_num_people.pop(33-5)
mat_num_people.pop(35-6)
mat_num_people.pop(36-7)
mat_num_people.pop(42-8)
mat_num_people.pop(55-9)
mat_num_people.pop(56-10)
mat_num_people.pop(67-11)
mat_num_people.pop(80-12)
mat_num_people.pop(81-13)
mat_num_people.pop(93-14)
mat_num_people.pop(102-15)
mat_num_people.pop(103-16)

for elem in mat_num_people:
    elem.pop(0)
    elem.pop(1-1)
    elem.pop(6-2)
    elem.pop(20-3)
    elem.pop(32-4)
    elem.pop(33-5)
    elem.pop(35-6)
    elem.pop(36-7)
    elem.pop(42-8)
    elem.pop(55-9)
    elem.pop(56-10)
    elem.pop(67-11)
    elem.pop(80-12)
    elem.pop(81-13)
    elem.pop(93-14)
    elem.pop(102-15)
    elem.pop(103-16)

#%% Percentage conversion

for row in mat_num_people:
    travel_in.append(row.copy())
    population.append(sum(row))

for row in mat_num_people[0:num_city]:
    travel_in_2.append(row[0:num_city].copy())
    population_2.append(sum(row[0:num_city]))

travel_out = population.copy()
travel_out_2 = population_2.copy()

for i in range(len(travel_in)):
    travel_in[i][i] = 0.0
    s = sum(travel_in[i])
    travel_out[i] = s/travel_out[i]
    for j in range(len(travel_in[i])):
        travel_in[i][j] /= s

for i in range(len(travel_in_2)):
    travel_in_2[i][i] = 0.0
    s = sum(travel_in_2[i])
    travel_out_2[i] = s/travel_out_2[i]
    for j in range(len(travel_in_2[i])):
        travel_in_2[i][j] /= s


# %% Delete non-usable variables

del all_dat, bo, elem, i, ib, j, k, mat_num_people, row, s, temp, temp2, num_city
