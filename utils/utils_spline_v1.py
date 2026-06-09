import concurrent.futures
import numpy as np
import warnings
from scipy.interpolate import UnivariateSpline
from contextlib import contextmanager
from joblib.parallel import BatchCompletionCallBack
from joblib import Parallel
warnings.filterwarnings("ignore", category=FutureWarning)
np.seterr(invalid='ignore')  
np.seterr(all='ignore')      

@contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = Parallel.__init__.__globals__['BatchCompletionCallBack']
    Parallel.__init__.__globals__['BatchCompletionCallBack'] = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        Parallel.__init__.__globals__['BatchCompletionCallBack'] = old_callback
        tqdm_object.close()
    
def _spline_fit_worker(t_data, x_data, current_s, queue):
    try:
        spline_x_temp = UnivariateSpline(t_data, x_data, s=current_s)
        queue.put({'status': 'success', 'spline': spline_x_temp})
    except Exception as e:
        queue.put({'status': 'error', 'message': str(e)})
        
def _spline_fit_in_thread(t_data_thread, x_data_thread, current_s_thread):
    return UnivariateSpline(t_data_thread, x_data_thread, s=current_s_thread)
    
def fit_model_bspline_fast(t, x, y, s, dt,RMS_threshold,both,factor):
    t = t[~np.isnan(t)]
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if np.sum(~np.isnan(t)) < 3 or np.sum(~np.isnan(x)) < 3:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    sort_idx = np.argsort(t, axis=None)
    t_sorted = np.asarray(t).flatten()[sort_idx]
    x_sorted = np.asarray(x).flatten()[sort_idx]
    y_sorted = np.asarray(y).flatten()[sort_idx]
    
    unique_t, inverse_idx = np.unique(t_sorted, return_inverse=True)

    x_median = np.zeros_like(unique_t, dtype=float)
    y_median = np.zeros_like(unique_t, dtype=float)
    
    for i in range(len(unique_t)):
        mask = inverse_idx == i
        x_median[i] = np.median(x_sorted[mask])
        y_median[i] = np.median(y_sorted[mask])

    t = unique_t
    x = x_median
    y = y_median

    if len(both)==0:
        step = dt / factor
        t_min = np.ceil(t[0] / step) * step
        t_max = t[-1]
        t_fit = np.arange(t_min, t_max, step, dtype=np.float64)
    else:
        t_fit = both[~np.isnan(both)]
    if len(t_fit) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    s = np.concatenate([
        np.arange(9e7, 0.9e7 - 1, -1e7),           
        np.arange(9e6, 1.5e6, -1e6),               
        [1.5e6, 1e6, 9.5e5],                       
        np.arange(9e5, 1e4, -1e5),                 
        np.arange(1e4, 0, -1e3),                    
        np.arange(1e3, 0, -1e2),
        np.arange(1e2, 0, -1e1),
        np.arange(1e1, 0, -1e0),
        np.arange(1e0, 0, -1e-1),
        np.arange(1e-1, 0, -1e-2)
    ])
    steps = len(t_fit)
    if steps == 0: 
        final_s_value_for_fit = s[0] 
        spline_x = UnivariateSpline(t, x, s=final_s_value_for_fit)
        spline_y = UnivariateSpline(t, y, s=final_s_value_for_fit)
        x_fit = spline_x(t_fit)
        u_fit = spline_x.derivative(1)(t_fit)
        y_fit = spline_y(t_fit)
        v_fit = spline_y.derivative(1)(t_fit)
        return x_fit, u_fit, t_fit, y_fit, v_fit
    
    x_mean = []
    y_mean = []
    for j in range(steps):
        mask = np.where((t > t_fit[j] - dt/2) & (t < t_fit[j] + dt/2))
        if np.any(mask):  
            x_mean.append(np.mean(x[mask]))
            y_mean.append(np.mean(y[mask]))
        else:
            x_mean.append(np.nan)  
            y_mean.append(np.nan)  
    x_mean = np.array(x_mean)
    y_mean = np.array(y_mean)

    current_RMS = 100
    old_RMS = current_RMS
    TIMEOUT_PER_SPLINE_FIT = 5
    last_successful_m_index = 0
    m = 0
    mask = np.where(t<t_fit[steps-1])
    t_test = t[mask]
    x_test = x[mask]
    y_test = y[mask]

    if len(t_test) < 3 or len(t) < 3: 
        x_fit = np.array([], dtype=np.float64)
        u_fit = np.array([], dtype=np.float64)
        t_fit = np.array([], dtype=np.float64)
        y_fit = np.array([], dtype=np.float64)
        v_fit = np.array([], dtype=np.float64)  
        return x_fit, u_fit, t_fit, y_fit, v_fit

    x_max = int(np.nanmax(x_test))+100
    x_min = int(np.nanmin(x_test))-100
    y_max = int(np.nanmax(y_test))+100
    y_min = int(np.nanmin(y_test))-100
    while current_RMS > RMS_threshold and m < len(s):
        current_s_value = s[m]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_spline_fit_in_thread, t_test, x_test, current_s_value)
            future2 = executor.submit(_spline_fit_in_thread, t_test, y_test, current_s_value)
            try:
                spline_x_temp = future.result(timeout=TIMEOUT_PER_SPLINE_FIT)
                x_fit_temp = np.ravel(spline_x_temp(t_fit[:steps]))

                spline_y_temp = future2.result(timeout=TIMEOUT_PER_SPLINE_FIT)
                y_fit_temp = np.ravel(spline_y_temp(t_fit[:steps]))

                if not np.all(np.isfinite(x_fit_temp)) or not np.all(np.isfinite(y_fit_temp)):
                    raise ValueError("NaN/Inf in Fit-Ergebnis")
    
                current_RMS = np.sqrt(np.nanmean((x_mean - x_fit_temp)**2+(y_mean-y_fit_temp)**2))
                if np.nanmax(x_fit_temp)>x_max or np.nanmin(x_fit_temp)<x_min or np.nanmax(y_fit_temp)>y_max or np.nanmin(y_fit_temp)<y_min:
                    if current_s_value<2e2:
                        break
                    else:
                        pass
                elif current_RMS<1.02*old_RMS:
                    last_successful_m_index = m
                    old_RMS=current_RMS
                elif current_RMS<1.15*old_RMS:
                    pass
                elif current_s_value<2e2:
                    break
            except concurrent.futures.TimeoutError:
                future.cancel()
                future2.cancel()
                break 
            except Exception as e:
                future.cancel()
                future2.cancel()
                pass 
        m += 1 
    s_take = s[last_successful_m_index]
    spline_x = UnivariateSpline(t, x, s=s_take)
    spline_y = UnivariateSpline(t, y, s=s_take)

    x_fit = spline_x(t_fit)
    u_fit = spline_x.derivative(1)(t_fit)

    y_fit = spline_y(t_fit)
    v_fit = spline_y.derivative(1)(t_fit)
    return x_fit, u_fit, t_fit, y_fit, v_fit