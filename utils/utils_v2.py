import numpy as np
import hdbscan
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree, cKDTree
from scipy.ndimage import convolve
from skimage.measure import label, regionprops
from scipy.optimize import linear_sum_assignment
from skimage.filters import threshold_otsu
from contextlib import contextmanager
from joblib.parallel import BatchCompletionCallBack
from joblib import Parallel
    
def compute_perpendicular_spread(events, velocity):
    if np.linalg.norm(velocity) == 0 or len(events) == 0:
        return 0.0

    v = velocity / np.linalg.norm(velocity)
    n = np.array([-v[1], v[0]])  
    events = events.T
    centered = events - np.mean(events, axis=0)
    projections = np.dot(centered, n)
    return np.std(projections)

def correct_measurement(z, velocity, events, beta=0.8):
    sigma_perp = compute_perpendicular_spread(events, velocity)
    if np.linalg.norm(velocity) == 0:
        return z  

    direction = velocity / np.linalg.norm(velocity)
    offset = beta * sigma_perp * direction
    return z - offset

def extend_1d(arr, extend_size, val=np.nan):
    return np.concatenate((arr, np.full(extend_size, val)))

def extend_list_of_arrays(lst, pad_length, array_shape):
    for _ in range(pad_length):
        lst.append(np.full(array_shape, np.nan))
    return lst
    
def process_index_afterwards(i, P, t_clust, x_clust, y_clust, 
                  C, Q, R, P_new, x_est, P_newz, y_est,
                  Kalman_v2_both_optimized, Kalman_v2_both_buffer_optimized,dtt):
    if len(t_clust[i])==0:
        return i, None  
    sort_idx = np.argsort(t_clust[i])
    t_clust[i] = t_clust[i][sort_idx]
    x_clust[i] = x_clust[i][sort_idx]
    y_clust[i] = y_clust[i][sort_idx]
    
    st = np.floor(min(10,(np.max(t_clust[i])-np.min(t_clust[i]))/dtt))
    mask = t_clust[i]<np.min(t_clust[i])+st*dtt

    A = np.vstack([t_clust[i][mask]* 1e-6, np.ones_like(t_clust[i][mask])]).T
    a, b = np.linalg.lstsq(A, x_clust[i][mask], rcond=None)[0]
    px = a
    a, b = np.linalg.lstsq(A, y_clust[i][mask], rcond=None)[0]
    py = a
    xe = np.array([np.mean(x_clust[i][mask]), px, 0.0], dtype=np.float64)
    ye = np.array([np.mean(y_clust[i][mask]), py, 0.0], dtype=np.float64)
    P, xe, Pz, ye, t, x_p, y_p, xv_p, yv_p, xa_p, ya_p = Kalman_v2_both_buffer_optimized(
        t_clust[i], x_clust[i], y_clust[i], 1, C, Q, R, P*10, xe, P*10, ye, 0, dtt
    )
    return i, {
        "P": P, "xe": xe, "Pz": Pz, "ye": ye,
        "t": t, "x": x_p, "y": y_p,
        "u": xv_p, "v": yv_p, "ax": xa_p, "ay": ya_p
    }

def process_index_lightweight_afterwards(i, P, t_m, x_m, y_m, 
                  C, Q, R, P_new, x_est, P_newz, y_est,
                  Kalman_v2_both_optimized, Kalman_v2_both_buffer_optimized,dtt,factor):
    if len(t_m[i])==0:
        return i, None  
    
    sort_idx = np.argsort(t_m[i])
    t_m[i] = t_m[i][sort_idx]
    x_m[i] = x_m[i][sort_idx]
    y_m[i] = y_m[i][sort_idx]
    
    st = np.floor(min(4,(np.max(t_m[i])-np.min(t_m[i]))/(dtt)))
    mask = t_m[i]<=np.min(t_m[i])+st*dtt
    
    A = np.vstack([t_m[i][mask]*1e-6, np.ones_like(t_m[i][mask])]).T
    a, b = np.linalg.lstsq(A, x_m[i][mask], rcond=None)[0]
    px = a
    a, b = np.linalg.lstsq(A, y_m[i][mask], rcond=None)[0]
    py = a
    xe = np.array([np.mean(x_m[i][mask]), px, 0.0], dtype=np.float64)
    ye = np.array([np.mean(y_m[i][mask]), py, 0.0], dtype=np.float64)
    P, xe, Pz, ye, t, x_p, y_p, xv_p, yv_p, xa_p, ya_p = Kalman_v2_both_buffer_optimized(
        t_m[i], x_m[i], y_m[i], 0, C, Q, R, P*50, xe, P*50, ye, 0, dtt,factor
    )
    return i, {
        "P": P, "xe": xe, "Pz": Pz, "ye": ye,
        "t": t, "x": x_p, "y": y_p,
        "u": xv_p, "v": yv_p, "ax": xa_p, "ay": ya_p
    }
    
def clustering(apply_hdbscan,epsilon,minPts,X,Y):
    if apply_hdbscan:
        db = hdbscan.HDBSCAN(min_cluster_size=minPts, min_samples=minPts)
        idx = db.fit_predict(np.stack((X, Y), axis=-1))
    else:
        db = DBSCAN(eps=epsilon, min_samples=minPts)
        idx = db.fit_predict(np.stack((X, Y), axis=-1))
    return idx

def update_cluster(x_m,y_m,t_m,i,X,Y,T,x_clust,y_clust,t_clust,idx,del_,steps,buffer,steps_rev,inactive_k,k,lightweight_mode,num_points,ind_clust,ind_global):
        if lightweight_mode:
            t_min, t_max = T[idx].min(), T[idx].max()
            bins = np.linspace(t_min, t_max, num=num_points+1)
            for j in range(num_points):
                mask = np.where((T[idx]>=bins[j]) & (T[idx]<=bins[j+1]))
                if mask[0].size > 0:
                    x_m[i] = np.append(x_m[i], np.median(X[idx][mask]))
                    y_m[i] = np.append(y_m[i], np.median(Y[idx][mask]))
                    t_m[i] = np.append(t_m[i], np.median(T[idx][mask])* 1e-6)
        else:
            x_m[i] = np.median(X[idx])
            y_m[i] = np.median(Y[idx])
            t_m[i] = np.median(T[idx]) * 1e-6

        if del_[i] == 1 and steps[i] >= buffer and steps[i]>steps_rev:
            x_clust[i] = X[idx]
            y_clust[i] = Y[idx]
            t_clust[i] = T[idx]
            ind_clust[i] = ind_global[idx]
        else:
            x_clust[i] = np.concatenate([x_clust[i], X[idx]])
            y_clust[i] = np.concatenate([y_clust[i], Y[idx]])
            t_clust[i] = np.concatenate([t_clust[i], T[idx]])
            ind_clust[i] = np.concatenate([ind_clust[i], ind_global[idx]])
        steps[i] += 1
        inactive_k[i] = k
        return x_m, y_m, t_m, x_clust, y_clust, t_clust, steps, inactive_k, ind_clust

def db_matching(valid,x_m,y_m,t_m,x_temp,y_temp,max_tracks,threshold,idx,del_,steps,buffer,steps_rev,X,Y,T,x_clust,y_clust,t_clust,k,inactive_k,lightweight_mode,num_points,num_involved,ind_clust, ind_global):
        valid_indices = np.where(valid == 1)[0]
        pp_temp = np.full(len(valid_indices), 0, dtype=np.float64)
        pp_temp2 = np.full(len(valid_indices), 0, dtype=np.float64)

        if lightweight_mode:
            for j in range(len(valid_indices)):
                if len(x_m[valid_indices[j]])>0:
                    pp_temp[j] = np.mean(x_m[valid_indices[j]][-num_involved[valid_indices[j]]::])
                    pp_temp2[j] = np.mean(y_m[valid_indices[j]][-num_involved[valid_indices[j]]::])
            raw_points_m = np.column_stack((pp_temp, pp_temp2))
        else:   
            raw_points_m = np.column_stack((x_m[valid_indices], y_m[valid_indices]))
        nonzero_mask = ~np.all(raw_points_m == 0, axis=1)
        points_m = raw_points_m[nonzero_mask]
        original_m_indices = valid_indices[nonzero_mask]
        points_temp = np.column_stack((x_temp, y_temp))[~np.all(np.column_stack((x_temp, y_temp)) == 0, axis=1)]

        dist_matrix = cdist(points_temp, points_m)

        nearest_indices = np.full(len(points_temp), -1)
        nearest_distances = np.full(len(points_temp), np.inf)
        
        for i in range(dist_matrix.shape[0]):
            valid_indices = np.where(dist_matrix[i] < threshold)[0]
            if valid_indices.size > 0:
                min_idx = valid_indices[np.argmin(dist_matrix[i][valid_indices])]
                nearest_indices[i] = min_idx
                nearest_distances[i] = dist_matrix[i][min_idx]

        for m_index in np.unique(nearest_indices[nearest_indices != -1]):
            temp_matches = np.where(nearest_indices == m_index)[0]
            if len(temp_matches) > 1:
                best = temp_matches[np.argmin(nearest_distances[temp_matches])]
                for i in temp_matches:
                    if i != best:
                        nearest_indices[i] = -1 
                        nearest_distances[i] = np.inf

        for i in range(len(nearest_indices)):
            if nearest_indices[i]>-1:
                mask = idx == i
                if lightweight_mode:
                    l_temp = len(x_m[original_m_indices[nearest_indices[i]]])
                x_m, y_m, t_m, x_clust, y_clust, t_clust, steps, inactive_k, ind_clust = update_cluster(
                    x_m,y_m,t_m,original_m_indices[nearest_indices[i]],X,Y,T,x_clust,y_clust,t_clust,mask,del_,steps,buffer,steps_rev,inactive_k,k,lightweight_mode,num_points,ind_clust, ind_global)
                if lightweight_mode:
                    num_involved[original_m_indices[nearest_indices[i]]] = len(x_m[original_m_indices[nearest_indices[i]]]) - l_temp  

        missing_m_indices = np.setdiff1d(np.arange(len(points_m)), nearest_indices[nearest_indices != -1])
        for i in range(len(missing_m_indices)):
            idx_real = original_m_indices[missing_m_indices[i]]
            valid[idx_real] = 0
            inactive_k[idx_real] = k
        return nearest_indices, x_clust, y_clust, x_m, y_m, t_m, steps, inactive_k, valid, t_clust, num_involved,ind_clust
    
    
def kdtree_clustering(X,Y,T,valid,x_m,y_m,t_m,Range,Lmin,steps,del_,buffer,steps_rev,inactive_k,k,x_clust,y_clust,t_clust,lightweight_mode,num_points,num_involved,ind_clust,ind_global):
        points = np.column_stack((X, Y))
        valid_indices = np.where(valid == 1)[0]
        
        tree = KDTree(points)
        all_points_to_delete = []
        for i in valid_indices:
            if lightweight_mode:
                if len(x_m[i])==0:
                    continue
                
                center = [np.mean(x_m[i][-num_involved[i]::]), np.mean(y_m[i][-num_involved[i]::])]
            else:   
                if len(x_clust[i])==0:
                    continue
                center = [x_m[i], y_m[i]]
            idx = tree.query_ball_point(center, r=Range+4)

            if len(idx) > Lmin:
                if lightweight_mode:
                    l_temp = len(x_m[i])
                x_m, y_m, t_m, x_clust, y_clust, t_clust, steps, inactive_k, ind_clust = update_cluster(
                    x_m,y_m,t_m,i,X,Y,T,x_clust,y_clust,t_clust,idx,del_,steps,buffer,steps_rev,inactive_k,k,lightweight_mode,num_points, ind_clust, ind_global)
                if lightweight_mode:
                    num_involved[i] = len(x_m[i]) - l_temp  
                all_points_to_delete.extend(idx)
            else:
                valid[i] = 0
                inactive_k[i] = k

        if all_points_to_delete:
            unique_indices_to_delete = np.unique(all_points_to_delete)
            mask = np.ones(len(X), dtype=bool)
            mask[unique_indices_to_delete] = False
    
            X = X[mask]
            Y = Y[mask]
            T = T[mask]
            ind_global = ind_global[mask]
        return x_m, y_m, t_m, x_clust, y_clust, t_clust, steps, inactive_k, valid, X, Y, T, num_involved, ind_clust, ind_global
    
    
def subgrid_search(height,width,search_factor,minPts,inactive,X,Y):
    g = 0
    startx = height / 2 + (np.random.rand() - 0.5) * height / 1
    starty = width / 2 + (np.random.rand() - 0.5) * width / 1
    factor_array = 4 * (search_factor - np.arange(1, search_factor+1))+1
    threshold_sub = np.sum(inactive) * minPts * 5 

    while g < search_factor:
        range_x = height / factor_array[g]
        range_y = width / factor_array[g]

        x_min, x_max = startx - range_x, startx + range_x
        y_min, y_max = starty - range_y, starty + range_y

        idx = (X > x_min) & (X < x_max) & (Y > y_min) & (Y < y_max)

        if np.sum(idx) >= threshold_sub:
            break

        g += 1
    return idx

def add_results(result,buffer,lightweight_mode):
    if buffer:
        return result["P"], result["xe"], result["Pz"], result["ye"], result["t"].ravel(), result["x"], result["u"], result["ax"], result["y"], result["v"], result["ay"]
    else:
        if lightweight_mode:
            return result["P"], result["xe"], result["Pz"], result["ye"], result["t"], result["x"], result["u"], result["ax"], result["y"], result["v"], result["ay"]
        else:
            return result["P"], result["xe"], result["Pz"], result["ye"], result["t"][-1], result["x"][-1], result["u"][-1], result["ax"][-1], result["y"][-1], result["v"][-1], result["ay"][-1]
                          
def circular_average_filter(image, radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel = mask.astype(float)
    kernel /= kernel.sum()  
    return convolve(image, kernel, mode='reflect')               
                
def analyze_pseudo_image(X,Y,height,width,area,filtersize):
    I = np.zeros((height, width))
    I[X-1,Y-1] = 1
    
    I = circular_average_filter(I, filtersize)
    thresh = threshold_otsu(I)
    I = I>thresh
    label_image = label(I)
    props = regionprops(label_image)
    centroids = np.array([prop.centroid for prop in props if prop.area>area])
    coords = [prop.coords for prop in props if prop.area>area]
    return centroids, coords
                
def match_nearest_neighbors_partial_threshold(points_a, points_b, threshold=10.0):
    if len(points_a)<1 or len(points_b)<1:
        return [], [], []
    cost_matrix = cdist(points_a, points_b)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_distances = cost_matrix[row_ind, col_ind]

    valid_mask = matched_distances < threshold
    row_ind_valid = row_ind[valid_mask]
    col_ind_valid = col_ind[valid_mask]
    distances_valid = matched_distances[valid_mask]
    return row_ind_valid, col_ind_valid, distances_valid                
                
def extend_length(array_list,L):
    array_list = [np.pad(arr, (0, L), mode='constant', constant_values=np.nan) for arr in array_list]
    return array_list                
                
def deriv(t_val,coefs):
    return sum(i * coefs[i] * t_val**(i - 1) for i in range(1, len(coefs)))

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

def repair_tracks(x_clust, y_clust, t_clust, ind_clust, steps, dt):
    n = len(x_clust)

    x_s = np.full(n, np.nan)
    y_s = np.full(n, np.nan)
    x_e = np.full(n, np.nan)
    y_e = np.full(n, np.nan)
    t_s = np.zeros(n)
    t_e = np.zeros(n)

    for i in range(n):
        if len(x_clust[i]) == 0 or np.isnan(x_clust[i]).all():
            continue

        t_s[i] = np.min(t_clust[i]) + dt
        t_e[i] = np.max(t_clust[i]) - dt

        mask_start = t_clust[i] < t_s[i]
        mask_end   = t_clust[i] > t_e[i]

        if np.any(mask_start):
            x_s[i] = np.nanmedian(x_clust[i][mask_start])
            y_s[i] = np.nanmedian(y_clust[i][mask_start])

        if np.any(mask_end):
            x_e[i] = np.nanmedian(x_clust[i][mask_end])
            y_e[i] = np.nanmedian(y_clust[i][mask_end])

    max_dtt  = dt
    max_dist = 5

    valid_end = ~np.isnan(x_e) & ~np.isnan(y_e)
    end_points = np.column_stack((x_e[valid_end], y_e[valid_end]))
    end_indices = np.where(valid_end)[0]

    tree = cKDTree(end_points)

    used_sources = set()
    used_targets = set()
    links = []

    for tgt in range(n):
        if np.isnan(x_s[tgt]) or np.isnan(y_s[tgt]):
            continue
        if tgt in used_targets:
            continue

        neigh = tree.query_ball_point([x_s[tgt], y_s[tgt]], r=max_dist)

        for k in neigh:
            src = end_indices[k]

            if src == tgt:
                continue
            if src in used_sources:
                continue
            time_diff = abs((t_s[tgt] - dt) - (t_e[src] + dt))
            if not (0 < time_diff <= max_dtt):
                continue

            links.append((tgt, src))
            used_targets.add(tgt)
            used_sources.add(src)
            break  
    for idx_target, idx_source in links:
        if len(x_clust[idx_source]) == 0:
            continue

        x_clust[idx_target] = np.concatenate((x_clust[idx_target], x_clust[idx_source]))
        y_clust[idx_target] = np.concatenate((y_clust[idx_target], y_clust[idx_source]))
        t_clust[idx_target] = np.concatenate((t_clust[idx_target], t_clust[idx_source]))
        ind_clust[idx_target] = np.concatenate((ind_clust[idx_target], ind_clust[idx_source]))

        x_clust[idx_source] = np.array([])
        y_clust[idx_source] = np.array([])
        t_clust[idx_source] = np.array([])
        ind_clust[idx_source] = np.array([])
    return x_clust, y_clust, t_clust, steps, ind_clust

def repair_tracks_pseudoframes(
    x_clust, y_clust, t_clust, ind_clust,
    steps, dt,
    x_m, y_m, t_m
):
    t_start = np.array([arr[0] * 1e6 for arr in t_m if len(arr) > 0])
    t_end   = np.array([arr[-1] * 1e6 for arr in t_m if len(arr) > 0])

    x_start = np.array([arr[0] for arr in x_m if len(arr) > 0])
    x_end   = np.array([arr[-1] for arr in x_m if len(arr) > 0])

    y_start = np.array([arr[0] for arr in y_m if len(arr) > 0])
    y_end   = np.array([arr[-1] for arr in y_m if len(arr) > 0])

    max_dtt  = 1.5 * dt
    max_dist = 4.0
    max_dist2 = max_dist * max_dist

    order = np.argsort(t_end)
    t_end_s = t_end[order]

    links = []
    used_sources = set()
    used_targets = set()

    for tgt in range(len(t_start)):
        if x_start[tgt] == 0:
            continue

        ts = t_start[tgt]

        left  = np.searchsorted(t_end_s, ts - max_dtt, side="left")
        right = np.searchsorted(t_end_s, ts + max_dtt, side="right")

        for k in range(left, right):
            src = order[k]

            if src == tgt:
                continue
            if src in used_sources or tgt in used_targets:
                continue
            if x_end[src] == 0:
                continue
            if t_end[src] == ts:
                continue

            dx = x_start[tgt] - x_end[src]
            dy = y_start[tgt] - y_end[src]
            if dx*dx + dy*dy > max_dist2:
                continue

            links.append((tgt, src))
            used_sources.add(src)
            used_targets.add(tgt)
            break  
    for idx_target, idx_source in links:
        if len(x_clust[idx_source]) == 0:
            continue

        x_m[idx_target] = np.concatenate((x_m[idx_target], x_m[idx_source]))
        y_m[idx_target] = np.concatenate((y_m[idx_target], y_m[idx_source]))
        t_m[idx_target] = np.concatenate((t_m[idx_target], t_m[idx_source]))

        x_clust[idx_target] = np.concatenate((x_clust[idx_target], x_clust[idx_source]))
        y_clust[idx_target] = np.concatenate((y_clust[idx_target], y_clust[idx_source]))
        t_clust[idx_target] = np.concatenate((t_clust[idx_target], t_clust[idx_source]))
        ind_clust[idx_target] = np.concatenate((ind_clust[idx_target], ind_clust[idx_source]))

        x_m[idx_source] = np.array([])
        y_m[idx_source] = np.array([])
        t_m[idx_source] = np.array([])

        x_clust[idx_source] = np.array([])
        y_clust[idx_source] = np.array([])
        t_clust[idx_source] = np.array([])
        ind_clust[idx_source] = np.array([])

        steps[idx_target] += steps[idx_source]
        steps[idx_source] = 0

    return x_clust, y_clust, t_clust, steps, ind_clust, x_m, y_m, t_m

def process_correction(x_i,y_i,t_i,xv_i,yv_i,x_clust_i,y_clust_i,t_clust_i,dt,beta,lightweight_mode):
    if lightweight_mode:
        delta_t = np.diff(t_i[~np.isnan(t_i)])
        dt_nonzero = delta_t[delta_t != 0]
        if len(dt_nonzero) > 0:
            dt = np.unique(dt_nonzero)[0]*4
    for j in range(len(x_i)):
        velocity = np.array([xv_i[j], yv_i[j]]) * 1e6
        mask = np.logical_and(
            t_clust_i > (t_i[j] - dt/2),
            t_clust_i < (t_i[j] + dt/2)
        )
        events = np.array([x_clust_i[mask], y_clust_i[mask]])
        x_i[j], y_i[j] = correct_measurement([x_i[j], y_i[j]], velocity, events, beta)
        if np.isnan(x_i[j]):
            t_i[j]=np.nan

    return x_i, y_i, t_i