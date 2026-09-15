import numpy as np
from scipy import interpolate

amp = np.arange(0, 1.01, 0.1)
voltage = np.array([0, 8, 30, 66, 113.8, 170.0, 231.5, 298.75, 363.75, 428.8, 493.8])
voltage = voltage/np.max(voltage)
amplitude_compensator = interpolate.interp1d(voltage, amp, 'cubic')
