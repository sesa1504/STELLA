import numpy as np
from kalman_filter_v2 import Kalman_v2_both_optimized_lightweight, Kalman_v2_both_buffer_optimized_lightweight_afterwards, Kalman_v2_both_optimized, Kalman_v2_both_buffer_optimized_afterwards, dummy_Kalman_lightweight, dummy_Kalman
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from utils_v2 import * 
from utils_spline_v1 import *
from joblib import Parallel, delayed, parallel_backend
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm
import multiprocessing
warnings.filterwarnings("ignore", category=FutureWarning)

# %% init
if lightweight_mode:
    dummy_Kalman_lightweight()
else:
    dummy_Kalman()

P_new = [[] for _ in range(max_tracks)]
P_newz = [[] for _ in range(max_tracks)]
x_plot  = [np.array([], dtype=float) for _ in range(max_tracks)]
x_plota = [np.array([], dtype=float) for _ in range(max_tracks)]
x_plotv = [np.array([], dtype=float) for _ in range(max_tracks)]
y_plot  = [np.array([], dtype=float) for _ in range(max_tracks)]
y_plota = [np.array([], dtype=float) for _ in range(max_tracks)]
y_plotv = [np.array([], dtype=float) for _ in range(max_tracks)]
t_plot  = [np.array([], dtype=float) for _ in range(max_tracks)]
track_to_cluster = np.full(len(x_plot), -1, dtype=int)
x_est = [[] for _ in range(max_tracks)]
y_est = [[] for _ in range(max_tracks)]
x_temp = np.zeros(max_tracks)
y_temp = np.zeros(max_tracks)
#%% main loop
# Kalman filter afterwards
if Kalman_afterwards or use_both or both2:
    if lightweight_mode or pseudo_images:
        partial_func = partial(
            process_index_lightweight_afterwards,
            P=P,
            t_m=t_m,
            x_m=x_m,
            y_m=y_m,
            C=C,
            Q=Q,
            R=R,
            P_new=P_new,
            x_est=x_est,
            P_newz=P_newz,
            y_est=y_est,
            Kalman_v2_both_optimized=Kalman_v2_both_optimized_lightweight,
            Kalman_v2_both_buffer_optimized=Kalman_v2_both_buffer_optimized_lightweight_afterwards,
            dtt=dt,
            factor=resol
        )
    else:   
        partial_func = partial(
            process_index_afterwards,
            P=P,
            t_clust=t_clust,
            x_clust=x_clust,
            y_clust=y_clust,
            C=C,
            Q=Q,
            R=R,
            P_new=P_new,
            x_est=x_est,
            P_newz=P_newz,
            y_est=y_est,
            Kalman_v2_both_optimized=Kalman_v2_both_optimized,
            Kalman_v2_both_buffer_optimized=Kalman_v2_both_buffer_optimized_afterwards,
            dtt = dt,
            factor=resol
        )
    if lightweight_mode==False:
        ex_len = len(x_clust)
    else:
        ex_len = len(t_m)
    track_to_cluster[:ex_len] = np.arange(ex_len, dtype=int)
    with ThreadPoolExecutor(max_workers=num_worker) as executor:
        results_iterator = executor.map(partial_func, range(ex_len))

        results = []
        for r in tqdm(results_iterator, total=ex_len, desc="Kalman"):
            results.append(r)

    for i, result in results:
        if result is None:
            continue
       res = add_results(result, 1, lightweight_mode)

        P_new[i]   = res[0]
        x_est[i]   = res[1]
        P_newz[i]  = res[2]
        y_est[i]   = res[3]
        
        t_plot[i]  = np.asarray(res[4], dtype=float)
        x_plot[i]  = np.asarray(res[5], dtype=float)
        x_plotv[i] = np.asarray(res[6], dtype=float)
        x_plota[i] = np.asarray(res[7], dtype=float)
        
        y_plot[i]  = np.asarray(res[8], dtype=float)
        y_plotv[i] = np.asarray(res[9], dtype=float)
        y_plota[i] = np.asarray(res[10], dtype=float)        
    if not both2 and not use_both:
        if correction:
            if lightweight_mode:
                x_plotv_temp = [arr * 1e-6 for arr in x_plotv]    
                y_plotv_temp = [arr * 1e-6 for arr in y_plotv]
                with parallel_backend('loky', n_jobs=multiprocessing.cpu_count()):
                    with tqdm_joblib(tqdm(total=len(x_plot), desc="correction")):
                        results = Parallel()( 
                            delayed(process_correction)(x, y, t, xv, yv, x_c, y_c, t_c, dt, beta,lightweight_mode)
                            for x,y,t,xv,yv,x_c,y_c,t_c in zip(x_plot, y_plot,t_plot,x_plotv_temp,y_plotv_temp,x_clust,y_clust,t_clust)
                        )
     
                x_plot = [res[0] for res in results]
                y_plot = [res[1] for res in results]
                t_plot = [res[2] for res in results]
            else:
                
                with parallel_backend('loky', n_jobs=multiprocessing.cpu_count()):
                    with tqdm_joblib(tqdm(total=len(x_plot), desc="correction")):
                        results = Parallel()( 
                            delayed(process_correction)(x, y, t, xv, yv, x_c, y_c, t_c, dt, beta,lightweight_mode)
                            for x,y,t,xv,yv,x_c,y_c,t_c in zip(x_plot, y_plot,t_plot,x_plotv,y_plotv,x_clust,y_clust,t_clust)
                        )
     
                x_plot = [res[0] for res in results]
                y_plot = [res[1] for res in results]
                t_plot = [res[2] for res in results]
    
    valid_idx = [i for i, a in enumerate(x_plot) if not np.isnan(a).all()]
    track_to_cluster = track_to_cluster[valid_idx]
    t_plot = [t_plot[i] for i in valid_idx]
    x_plot = [x_plot[i] for i in valid_idx]
    x_plota = [x_plota[i] for i in valid_idx]
    y_plot = [y_plot[i] for i in valid_idx]
    y_plota = [y_plota[i] for i in valid_idx]
    if use_both or both2:
        x_plotv = [x_plotv[i] for i in valid_idx]
        y_plotv = [y_plotv[i] for i in valid_idx]
    else:
        x_plotv = [x_plotv[i]*1e-6 for i in valid_idx]
        y_plotv = [y_plotv[i]*1e-6 for i in valid_idx]
    
    t_clust = [t_clust[i] for i in valid_idx]
    x_clust = [x_clust[i] for i in valid_idx]
    y_clust = [y_clust[i] for i in valid_idx]
    
    t_plot = [arr[~np.isnan(arr)] for arr in t_plot]
    x_plot = [arr[~np.isnan(arr)] for arr in x_plot]
    y_plot = [arr[~np.isnan(arr)] for arr in y_plot]
    x_plotv = [arr[~np.isnan(arr)] for arr in x_plotv]
    x_plota = [arr[~np.isnan(arr)] for arr in x_plota]
    y_plotv = [arr[~np.isnan(arr)] for arr in y_plotv]
    y_plota = [arr[~np.isnan(arr)] for arr in y_plota]
    t_plot = [t if len(x) > 0 else np.array([]) for x, t in zip(x_plot, t_plot)]
    
    for ij in range(2):
        x_plot = [uniform_filter1d(a, size=3, mode='nearest') for a in x_plot]
        x_plotv = [uniform_filter1d(a, size=3, mode='nearest') for a in x_plotv]
        y_plot = [uniform_filter1d(a, size=3,  mode='nearest') for a in y_plot]
        y_plotv = [uniform_filter1d(a, size=3,  mode='nearest') for a in y_plotv]
        
    for j, arr in enumerate(x_plot):
        if len(arr) < len(x_plotv[j]):
            x_plotv[j] = x_plotv[j][:len(arr)]
            x_plota[j] = x_plota[j][:len(arr)]
            y_plotv[j] = y_plotv[j][:len(arr)]
            y_plota[j] = y_plota[j][:len(arr)]
                    
    
if both2:
    valid_idx = [i for i, arr in enumerate(t_plot) if len(arr) > 2]
    track_to_cluster = track_to_cluster[valid_idx]
    t_plot = [t_plot[i] for i in valid_idx]
    x_plot = [x_plot[i] for i in valid_idx]
    y_plot = [y_plot[i] for i in valid_idx]
    t_clust_temp = [t_clust[i] for i in valid_idx]
    x_clust_temp = [x_clust[i] for i in valid_idx]
    y_clust_temp = [y_clust[i] for i in valid_idx]

    # spline fitting
    multiprocessing.set_start_method('spawn', force=True)  
    NUM_CLUSTERS_TO_TEST = len(t_clust)
    s = 1e6

    with parallel_backend('loky', n_jobs=num_worker):
        with tqdm_joblib(tqdm(total=NUM_CLUSTERS_TO_TEST, desc="Spline fitting")):
            results = Parallel()( 
                delayed(fit_model_bspline_fast)(t, x, y, s, dt,target_RMS,tt,resol) 
                for t, x, y,tt in zip(t_plot, x_plot, y_plot,t_plot)
            )

    x_plotv = []
    t_plot = []
    y_plotv = []

    for x_fit, u_fit, t_fit, y_fit, v_fit in results:
        x_plotv.append(u_fit)
        t_plot.append(t_fit)
        y_plotv.append(v_fit)
           
    if correction:
        if lightweight_mode:
            x_plotv_temp = [arr  for arr in x_plotv]    
            y_plotv_temp = [arr  for arr in y_plotv]
            with parallel_backend('loky', n_jobs=multiprocessing.cpu_count()):
                with tqdm_joblib(tqdm(total=len(x_plot), desc="correction")):
                    results = Parallel()( 
                        delayed(process_correction)(x, y, t, xv, yv, x_c, y_c, t_c, dt, beta,lightweight_mode)
                        for x,y,t,xv,yv,x_c,y_c,t_c in zip(x_plot, y_plot,t_plot,x_plotv_temp,y_plotv_temp,x_clust_temp,y_clust_temp,t_clust_temp)
                    )
 
            x_plot = [res[0] for res in results]
            y_plot = [res[1] for res in results]
            t_plot = [res[2] for res in results]
        else:
            
            with parallel_backend('loky', n_jobs=multiprocessing.cpu_count()):
                with tqdm_joblib(tqdm(total=len(x_plot), desc="correction")):
                    results = Parallel()( 
                        delayed(process_correction)(x, y, t, xv, yv, x_c, y_c, t_c, dt, beta,lightweight_mode)
                        for x,y,t,xv,yv,x_c,y_c,t_c in zip(x_plot, y_plot,t_plot,x_plotv,y_plotv,x_clust_temp,y_clust_temp,t_clust_temp)
                    )
 
            x_plot = [res[0] for res in results]
            y_plot = [res[1] for res in results]
            t_plot = [res[2] for res in results]

if spline_fitting or use_both:
    multiprocessing.set_start_method('spawn', force=True)  
    NUM_CLUSTERS_TO_TEST = len(t_clust)
    s = 1e6
    
    if pseudo_images:
        if not use_both:
            with parallel_backend('loky', n_jobs=num_worker):
                with tqdm_joblib(tqdm(total=NUM_CLUSTERS_TO_TEST, desc="Spline fitting")):
                    results = Parallel()( 
                        delayed(fit_model_bspline_fast)(t, x, y, s, dt,target_RMS,[],resol) 
                        for t, x, y in zip(t_m, x_m, y_m)
                    )
        else:
            with parallel_backend('loky', n_jobs=num_worker):
                with tqdm_joblib(tqdm(total=NUM_CLUSTERS_TO_TEST, desc="Spline fitting")):
                    results = Parallel()( 
                        delayed(fit_model_bspline_fast)(t, x, y, s, dt,target_RMS,tt,resol) 
                        for t, x, y,tt in zip(t_m, x_m, y_m,t_plot)
                    )
    else:
        if not use_both:
            with parallel_backend('loky', n_jobs=num_worker):
                with tqdm_joblib(tqdm(total=NUM_CLUSTERS_TO_TEST, desc="Spline fitting")):
                    results = Parallel()( 
                        delayed(fit_model_bspline_fast)(t, x, y, s, dt,target_RMS,[],resol) 
                        for i, (t, x, y) in enumerate(zip(t_clust, x_clust, y_clust))
                    )
        else:
            with parallel_backend('loky', n_jobs=num_worker):
                with tqdm_joblib(tqdm(total=NUM_CLUSTERS_TO_TEST, desc="Spline fitting")):
                    results = Parallel()( 
                        delayed(fit_model_bspline_fast)(t, x, y, s, dt,target_RMS,tt,resol)
                        for t, x, y,tt in zip(t_clust, x_clust, y_clust,t_plot)
                    )

    if use_both:
        x_plotv = []
        t_plot = []
        y_plotv = []
        x_plot_s = []
        y_plot_s = []
    else:
        x_plot = []
        x_plotv = []
        t_plot = []
        y_plot = []
        y_plotv = []
        track_to_cluster = []
    
    # for x_fit, u_fit, t_fit, y_fit, v_fit in results:
    for cluster_idx, (x_fit, u_fit, t_fit, y_fit, v_fit) in enumerate(results):
        if use_both:
            x_plotv.append(u_fit)
            t_plot.append(t_fit)
            y_plotv.append(v_fit)
            x_plot_s.append(x_fit)
            y_plot_s.append(y_fit)
        else:
            x_plot.append(x_fit)
            x_plotv.append(u_fit)
            t_plot.append(t_fit)
            y_plot.append(y_fit)
            y_plotv.append(v_fit)
            track_to_cluster.append(cluster_idx)
            
    if pseudo_images and use_both:
        invalid_idx = [i for i, a in enumerate(x_plotv) if len(a)==0]
        x_plot = [np.array([]) if i in invalid_idx else a for i, a in enumerate(x_plot)]
        y_plot = [np.array([]) if i in invalid_idx else a for i, a in enumerate(y_plot)]
        
   
    if correction:
        if lightweight_mode:
            x_plotv_temp = [arr  for arr in x_plotv]    
            y_plotv_temp = [arr  for arr in y_plotv]
            with parallel_backend('loky', n_jobs=multiprocessing.cpu_count()):
                with tqdm_joblib(tqdm(total=len(x_plot), desc="correction")):
                    results = Parallel()( 
                        delayed(process_correction)(x, y, t, xv, yv, x_c, y_c, t_c, dt, beta,lightweight_mode)
                        for x,y,t,xv,yv,x_c,y_c,t_c in zip(x_plot, y_plot,t_plot,x_plotv_temp,y_plotv_temp,x_clust,y_clust,t_clust)
                    )
 
            x_plot = [res[0] for res in results]
            y_plot = [res[1] for res in results]
            t_plot = [res[2] for res in results]
        else:
            
            with parallel_backend('loky', n_jobs=multiprocessing.cpu_count()):
                with tqdm_joblib(tqdm(total=len(x_plot), desc="correction")):
                    results = Parallel()( 
                        delayed(process_correction)(x, y, t, xv, yv, x_c, y_c, t_c, dt, beta,lightweight_mode)
                        for x,y,t,xv,yv,x_c,y_c,t_c in zip(x_plot, y_plot,t_plot,x_plotv,y_plotv,x_clust,y_clust,t_clust)
                    )
 
            x_plot = [res[0] for res in results]
            y_plot = [res[1] for res in results]
            t_plot = [res[2] for res in results]

if use_hybrid:
    # Kalman filter
    if lightweight_mode or pseudo_images:
        partial_func = partial(
            process_index_lightweight_afterwards,
            P=P,
            t_m=t_m,
            x_m=x_m,
            y_m=y_m,
            C=C,
            Q=Q,
            R=R,
            P_new=P_new,
            x_est=x_est,
            P_newz=P_newz,
            y_est=y_est,
            Kalman_v2_both_optimized=Kalman_v2_both_optimized_lightweight,
            Kalman_v2_both_buffer_optimized=Kalman_v2_both_buffer_optimized_lightweight_afterwards,
            dtt=dt,
            factor=resol
        )
    else:   
        partial_func = partial(
            process_index_afterwards,
            P=P,
            t_clust=t_clust,
            x_clust=x_clust,
            y_clust=y_clust,
            C=C,
            Q=Q,
            R=R,
            P_new=P_new,
            x_est=x_est,
            P_newz=P_newz,
            y_est=y_est,
            Kalman_v2_both_optimized=Kalman_v2_both_optimized,
            Kalman_v2_both_buffer_optimized=Kalman_v2_both_buffer_optimized_afterwards,
            dtt = dt,
            factor = resol
        )
    if lightweight_mode==False:
        ex_len = len(x_clust)
    else:
        ex_len = len(t_m)
    track_to_cluster[:ex_len] = np.arange(ex_len, dtype=int)
    with ThreadPoolExecutor(max_workers=num_worker) as executor:
        results_iterator = executor.map(partial_func, range(ex_len))

        results = []
        for r in tqdm(results_iterator, total=ex_len, desc="Kalman"):
            results.append(r)

    for i, result in results:
        if result is None:
            continue
        res = add_results(result, 1, lightweight_mode)

        P_new[i]   = res[0]
        x_est[i]   = res[1]
        P_newz[i]  = res[2]
        y_est[i]   = res[3]
        
        t_plot[i]  = np.asarray(res[4], dtype=float)
        x_plot[i]  = np.asarray(res[5], dtype=float)
        x_plotv[i] = np.asarray(res[6], dtype=float)
        x_plota[i] = np.asarray(res[7], dtype=float)
        
        y_plot[i]  = np.asarray(res[8], dtype=float)
        y_plotv[i] = np.asarray(res[9], dtype=float)
        y_plota[i] = np.asarray(res[10], dtype=float)
                    
    valid_idx = [i for i, a in enumerate(x_plot) if not np.isnan(a).all()]
    track_to_cluster = track_to_cluster[valid_idx]
    t_plot = [t_plot[i] for i in valid_idx]
    x_plot = [x_plot[i] for i in valid_idx]
    x_plotv = [x_plotv[i] for i in valid_idx]
    x_plota = [x_plota[i] for i in valid_idx]
    y_plot = [y_plot[i] for i in valid_idx]
    y_plotv = [y_plotv[i] for i in valid_idx]
    y_plota = [y_plota[i] for i in valid_idx]
    
    t_clust = [t_clust[i] for i in valid_idx]
    x_clust = [x_clust[i] for i in valid_idx]
    y_clust = [y_clust[i] for i in valid_idx]
    
    t_plot = [arr[~np.isnan(arr)] for arr in t_plot]
    x_plot = [arr[~np.isnan(arr)] for arr in x_plot]
    y_plot = [arr[~np.isnan(arr)] for arr in y_plot]
    
    for ij in range(2):
        x_plot = [uniform_filter1d(a, size=3, mode='nearest') for a in x_plot]
        y_plot = [uniform_filter1d(a, size=3,  mode='nearest') for a in y_plot]
    
    valid_idx = [i for i, arr in enumerate(t_plot) if len(arr) > 2]
    track_to_cluster = track_to_cluster[valid_idx]
    t_plot = [t_plot[i] for i in valid_idx]
    x_plot = [x_plot[i] for i in valid_idx]
    y_plot = [y_plot[i] for i in valid_idx]
    t_clust_temp = [t_clust[i] for i in valid_idx]
    x_clust_temp = [x_clust[i] for i in valid_idx]
    y_clust_temp = [y_clust[i] for i in valid_idx]
    
    # spline fitting
    multiprocessing.set_start_method('spawn', force=True)  
    NUM_CLUSTERS_TO_TEST = len(t_clust)
    s = 1e6
 
    with parallel_backend('loky', n_jobs=multiprocessing.cpu_count()):
        with tqdm_joblib(tqdm(total=NUM_CLUSTERS_TO_TEST, desc="Spline fitting")):
            results = Parallel()( 
                delayed(fit_model_bspline_fast)(t, x, y, s, dt,target_RMS,tt,resol) 
                for t, x, y,tt in zip(t_plot, x_plot, y_plot,t_plot)
            )

    x_plotv = []
    t_plot = []
    y_plotv = []
    x_plot = []
    y_plot = []

    for x_fit, u_fit, t_fit, y_fit, v_fit in results:
        x_plot.append(x_fit)
        x_plotv.append(u_fit)
        t_plot.append(t_fit)
        y_plot.append(y_fit)
        y_plotv.append(v_fit)
          
    if correction:
        if lightweight_mode:
            with parallel_backend('loky', n_jobs=num_worker):
                with tqdm_joblib(tqdm(total=len(x_plot), desc="correction")):
                    results = Parallel()(
                        delayed(process_correction)(x, y, t, xv, yv, x_c, y_c, t_c, dt, beta,lightweight_mode)
                        for x,y,t,xv,yv,x_c,y_c,t_c in zip(x_plot, y_plot,t_plot,x_plotv,y_plotv,x_clust_temp,y_clust_temp,t_clust_temp)
                    )
            x_plot = [res[0] for res in results]
            y_plot = [res[1] for res in results]
            t_plot = [res[2] for res in results]
        else:
            with parallel_backend('loky', n_jobs=num_worker):
                with tqdm_joblib(tqdm(total=len(x_plot), desc="correction")):
                    results = Parallel()( 
                        delayed(process_correction)(x, y, t, xv, yv, x_c, y_c, t_c, dt, beta,lightweight_mode)
                        for x,y,t,xv,yv,x_c,y_c,t_c in zip(x_plot, y_plot,t_plot,x_plotv,y_plotv,x_clust_temp,y_clust_temp,t_clust_temp)
                    )
 
            x_plot = [res[0] for res in results]
            y_plot = [res[1] for res in results]
            t_plot = [res[2] for res in results]
track_to_cluster = np.asarray(track_to_cluster, dtype=int)
