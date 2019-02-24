import numpy as np
import netCDF4

import matplotlib.pyplot as plt

data_file = 'data/climate_data/rcp26/zosga/zosga_Omon_CanESM2_rcp26_r1i1p1_200601-210012.nc'
# data_file = 'data/climate_data/rcp85/tas/tas_Amon_CanESM2_rcp85_r1i1p1_200601-210012.nc'
# data_file = 'data/climate_data/rcp26/tas/tas_Amon_CanESM2_rcp26_r1i1p1_200601-210012.nc'

# data_file = 'data/climate_data/tas_Amon_MIROC6_1pctCO2_r1i1p1f1_gn_320001-329912.nc'
# data_file = 'data/climate_data/tos_Omon_MIROC6_1pctCO2_r1i1p1f1_gn_320001-329912.nc'

nc = netCDF4.Dataset(data_file)

print (dir(nc))

# print f.history
# time = f.variables['time']
# print time.units
# print time.shape
# print time[:]
print (nc.variables) #['variables'])

zosga = nc['zosga']
plt.plot(zosga)
plt.show()

nc.close()
exit()


tas = nc['tas']
vals = np.empty((tas.shape[0],))
for i in range(tas.shape[0]):
    vals[i] = np.mean(tas[i,:,:])

nc.close()

xvals = np.arange(2006, 2101, 1.0/12.)
from scipy.constants import convert_temperature

yvals = convert_temperature(vals, 'kelvin', 'fahrenheit')
plt.plot(xvals, yvals)
plt.show()
