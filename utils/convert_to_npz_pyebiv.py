from pyebiv import pyEBIV as EBIV
import numpy as np
import scipy.io

#%% convert .raw or .dat files to .npz or .mat
# make sure the prophesee SDK is installed correctly and added to your python paths

# select output format:
use_npz = True
use_mat = False

# enter path to the .raw or .dat file
event_file_path = "[path to .raw file]"

# enter output filename (add .mat or .npz):
filename = "filename.npz"

# enter maximum duration of the output file in micro seconds:
max_dur = 1000000

#%% run EventsIterator
# you can set max_duration (maximum duration of the converted file) and delta_t (just for the read in, the accumulation time is set in STELLA) 
evData = EBIV()
evData.setDebugLevel(0) 
evData.loadRaw(event_file_path)
    
t = np.array(evData.time())
x = np.array(evData.x())
y = np.array(evData.y())
p = np.array(evData.p())

tmin = min(t)
mask = (t<=tmin+max_dur)
t = t[mask]
x = x[mask]
y = y[mask]
p = p[mask]

#%% save file
# please enter a filename for the converted file
if use_mat:
    scipy.io.savemat(filename, {'X': x, 'Y': y, 'T': t, 'P': p})
if use_npz:
    np.savez(
        filename,
        X=x,
        Y=y,
        T=t,
        P=p
    )
