from metavision_core.event_io import EventsIterator
import numpy as np
import scipy.io

#%% convert .raw or .dat files to .npz or .mat
# make sure the prophesee SDK is installed correctly and added to your python paths

# select output format:
use_npz = True
use_mat = False

# enter path to the .raw or .dat file
event_file_path = "[path to .raw or .dat file]"

# enter output filename (add .mat or .npz):
filename = "filename.npz"

# enter maximum duration of the output file in micro seconds:
max_dur = 1000000

#%% run EventsIterator
# you can set max_duration (maximum duration of the converted file) and delta_t (just for the read in, the accumulation time is set in STELLA) 
mv_iterator = EventsIterator(input_path=event_file_path, delta_t=1000,max_duration=max_dur)
    
mylist = list(mv_iterator)
X=[evs['x'] for evs in mylist]
Y=[evs['y'] for evs in mylist]
T=[evs['t'] for evs in mylist]
P=[evs['p'] for evs in mylist]

x_plot = np.concatenate(X)
y_plot = np.concatenate(Y)
t_plot = np.concatenate(T)
p_plot = np.concatenate(P)

#%% save file
# please enter a filename for the converted file
if use_mat:
    scipy.io.savemat(filename, {'X': x_plot, 'Y': y_plot, 'T': t_plot, 'P': p_plot})
if use_npz:
    np.savez(
        filename,
        X=x_plot,
        Y=y_plot,
        T=t_plot,
        P=p_plot
    )
