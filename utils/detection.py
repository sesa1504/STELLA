import numpy as np
import time
import warnings
from utils_v2 import * 
from scipy.ndimage import label
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
import collections
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    test_mode
except NameError:
    test_mode = False

# %% init
stop_event = globals().get("stop_event", None)
x_clust = [[] for _ in range(max_tracks)]
y_clust = [[] for _ in range(max_tracks)]
t_clust = [[] for _ in range(max_tracks)]
ind_clust = [[] for _ in range(max_tracks)]
active_pixels = [[] for _ in range(max_tracks)]
x_est = [[] for _ in range(max_tracks)]
y_est = [[] for _ in range(max_tracks)]
x_temp = np.zeros(max_tracks)
y_temp = np.zeros(max_tracks)
inactive_k = np.ones(max_tracks, dtype=np.int32)
del_ = np.zeros(max_tracks, dtype=np.int32)
steps = np.zeros(max_tracks, dtype=np.int32)
valid = np.ones(max_tracks, dtype=np.int32)
inactive = np.ones(max_tracks, dtype=np.int32)
L_before = np.zeros(max_tracks, dtype=np.int32)
num_involved = np.zeros(max_tracks, dtype=np.int32)
if lightweight_mode:
    x_m = [np.array([]) for _ in range(max_tracks)]
    y_m = [np.array([]) for _ in range(max_tracks)]
    t_m = [np.array([]) for _ in range(max_tracks)]
else:
    x_m = np.zeros(max_tracks)
    y_m = np.zeros(max_tracks)
    t_m = np.zeros(max_tracks)
 
T_min = np.min(T_global)
T_max = np.max(T_global)

tic_start_ges = time.time()
#%% main loop
for multi in range(multiN):
    for k in range(N):
        tic_start = time.time()     
        if pixelwise_extension:
            if k>0:
                mask = np.logical_and(T_global >= T_min + (k-overlap) * dt, T_global <= T_min + (k+1) * dt)
            else:
                mask = np.logical_and(T_global >= T_min + k * dt, T_global <= T_min + (k+1) * dt)
        else:
            mask = np.logical_and(T_global >= T_min + k * dt, T_global <= T_min + (k+1) * dt)
        X = X_global[mask] + 1
        Y = Y_global[mask] + 1
        T = T_global[mask]
        ind_global = np.where(mask)[0]
        if pixelwise_extension:
            T[T<T_min+k*dt] = -1
        if pseudo_images:
            X_flat = X.ravel()
            Y_flat = Y.ravel()
            T_flat = T.ravel()
        
        if k == 0:
            if pseudo_images:
                centroids, coords = analyze_pseudo_image(X,Y,height,width,area,filtersize)            
                lookup = defaultdict(list)
                for i, (x, y) in enumerate(zip(X.ravel(), Y.ravel())):
                    lookup[(x, y)].append(ind_global[i])
                for i, arr in enumerate(coords):
                    for x, y in arr:
                        if (x, y) in lookup:
                            for idx in lookup[(x+1, y+1)]:
                                ind_clust[i].append(idx)
                                x_clust[i].append(X.ravel()[ind_global==idx])
                                y_clust[i].append(Y.ravel()[ind_global==idx])
                                t_clust[i].append(T.ravel()[ind_global==idx])
                ind_clust = [np.array(a, dtype=int) if len(a) > 0 else np.array([]) for a in ind_clust]
                x_clust   = [np.array(a) if len(a) > 0 else np.array([]) for a in x_clust]
                y_clust   = [np.array(a) if len(a) > 0 else np.array([]) for a in y_clust]
                
                for i in range(len(centroids)):
                    x_m[i] = np.append(x_m[i],centroids[i,0])
                    y_m[i] = np.append(y_m[i],centroids[i,1])
                    t_m[i] = np.append(t_m[i],(T_min+dt/2+(k)*dt)*1e-6)
                    del_[i] = 0
                    inactive[i] = 0
                    steps[i] = 1
                    num_involved[i] = num_points
                    
                valid = np.ones(len(x_clust))
                
            elif pixelwise_extension:
                structure = np.ones((1, 1), dtype=bool)        
  
                XY = np.column_stack((X-1,Y-1))
                new_mask = np.zeros((height, width), dtype=bool)
                new_mask[X-1,Y-1] = True
                labeled, num_regions = label(new_mask)
                
                region_sizes = np.bincount(labeled.ravel())
                keep = region_sizes >= N_pixelwise
                labeled[~keep[labeled]] = 0
                new_labels = np.unique(labeled)
                new_labels = new_labels[new_labels != 0]
                
                region_coords = defaultdict(list)
                region_indices = defaultdict(list) 
                
                xs, ys = np.nonzero(labeled > 0)
                for i, (x, y) in enumerate(zip(xs, ys)):
                    label_0 = labeled[x, y]
                    if label_0 in new_labels:
                        region_coords[label_0].append((x, y))
                pixel_to_indices = defaultdict(list)
                for i, (x_, y_) in enumerate(XY):
                    pixel_to_indices[(x_, y_)].append(i)
                for label_0 in region_coords:
                    coords = region_coords[label_0]
                    indices = []
                    for x, y in coords:
                        indices.extend(pixel_to_indices[(x, y)])  
                    region_coords[label_0] = np.array(coords, dtype=np.uint16)
                    region_indices[label_0] = np.array(indices, dtype=np.uint32)
                    
                idx = np.zeros(len(X), dtype=np.int16)  
    
                for label_0, indices in region_indices.items():
                    idx[indices] = label_0
                      
                idx[idx==0]=-1    
                    
                if max_tracks > 0 and save_mode==False:
                    idx[idx > max_tracks] = -1   
                    
                valid_clusters = idx != -1
                X = X[valid_clusters]
                Y = Y[valid_clusters]
                T = T[valid_clusters]
                ind_global = ind_global[valid_clusters]
                idx = idx[valid_clusters]
        
                max_cluster = idx.max() if len(idx) > 0 else -1
                
                for i in range(max_cluster):
                    mask = idx == i
                    if not mask.any():
                        continue
                    x_clust[i] = X[mask]
                    y_clust[i] = Y[mask]
                    t_clust[i] = T[mask] 
                    ind_clust[i] = ind_global[mask]
                    active_pixels[i] = np.unique(np.column_stack((x_clust[i]-1,y_clust[i]-1)),axis=0)
                    if lightweight_mode:
                        t_min, t_max = t_clust[i].min(), t_clust[i].max()
                        bins = np.linspace(t_min, t_max, num=num_points+1)
                        for j in range(num_points):
                            mask = np.where((t_clust[i]>=bins[j]) & (t_clust[i]<=bins[j+1]))
                            if mask[0].size > 0:
                                x_m[i] = np.append(x_m[i], np.median(x_clust[i][mask]))
                                y_m[i] = np.append(y_m[i], np.median(y_clust[i][mask]))
                                t_m[i] = np.append(t_m[i], np.median(t_clust[i][mask])* 1e-6)
                                num_involved[i] +=1
                    else:
                        x_m[i] = np.median(x_clust[i])
                        y_m[i] = np.median(y_clust[i])
                        t_m[i] = np.median(t_clust[i])* 1e-6
        
                    del_[i] = 0
                    inactive[i] = 0
                    steps[i] = 1
                    
                valid = np.ones(len(x_clust))
            else:
                idx = clustering(apply_hdbscan=apply_hdbscan,epsilon=epsilon,minPts=minPts,X=X,Y=Y)
                if max_tracks > 0 and save_mode==False:
                    idx[idx > max_tracks] = -1  
        
                valid_clusters = idx != -1
                X = X[valid_clusters]
                Y = Y[valid_clusters]
                T = T[valid_clusters]
                ind_global = ind_global[valid_clusters]
                idx = idx[valid_clusters]
        
                max_cluster = idx.max() if len(idx) > 0 else -1
                
                for i in range(max_cluster):
                    mask = idx == i
                    x_clust[i] = X[mask]
                    y_clust[i] = Y[mask]
                    t_clust[i] = T[mask] 
                    ind_clust[i] = ind_global[mask]
                    if lightweight_mode:
                        t_min, t_max = t_clust[i].min(), t_clust[i].max()
                        bins = np.linspace(t_min, t_max, num=num_points+1)
                        for j in range(num_points):
                            mask = np.where((t_clust[i]>=bins[j]) & (t_clust[i]<=bins[j+1]))
                            if mask[0].size > 0:
                                x_m[i] = np.append(x_m[i], np.median(x_clust[i][mask]))
                                y_m[i] = np.append(y_m[i], np.median(y_clust[i][mask]))
                                t_m[i] = np.append(t_m[i], np.median(t_clust[i][mask])* 1e-6)
                                num_involved[i] +=1
                    else:
                        x_m[i] = np.median(x_clust[i])
                        y_m[i] = np.median(y_clust[i])
                        t_m[i] = np.median(t_clust[i])* 1e-6
        
                    del_[i] = 0
                    inactive[i] = 0
                    steps[i] = 1
                    
                valid = np.ones(len(x_clust))
        else:
            if pseudo_images:
                centroids, coords = analyze_pseudo_image(X,Y,height,width,area,filtersize)
                idx = []
                points_a = []
                for i in range(len(x_m)):
                    if len(x_m[i])<1 or valid[i]==0:
                        continue
                    if t_m[i][-1]==(T_min+dt/2+(k-1)*dt)*1e-6:
                        idx = np.append(idx,i)
                        points_a.append([x_m[i][-1], y_m[i][-1]])
                points_a = np.array(points_a)
                row_ind, col_ind, distances = match_nearest_neighbors_partial_threshold(points_a, centroids,threshold)
                matched_b = set(col_ind)
                
                valid_indices = [i for i in range(len(x_m)) if len(x_m[i]) > 0]
                valid_set = set(valid_indices)- set(idx[row_ind])
                for i in valid_set:
                    valid[i] = 0
                    
                lookup = defaultdict(list)
                for i, (x, y) in enumerate(zip(X_flat, Y_flat)):
                    lookup[(x, y)].append(i)
            
                for i in range(len(col_ind)):
                    track_id = int(idx[row_ind[i]])
                    x_m[track_id] = np.append(x_m[track_id], centroids[col_ind[i]][0])
                    y_m[track_id] = np.append(y_m[track_id], centroids[col_ind[i]][1])
                    t_m[track_id] = np.append(t_m[track_id], (T_min+dt/2+(k)*dt)*1e-6)
                    steps[track_id] +=1
                    inactive_k[track_id] =k
                    num_involved[track_id] = num_points
                    
                    tmp_ind = []
                    tmp_x   = []
                    tmp_y   = []
                    tmp_t   = []
                    for x, y in coords[col_ind[i]]:
                        for idxx in lookup.get((x+1, y+1), []):
                            tmp_ind.append(ind_global[idxx])
                            tmp_x.append(X_flat[idxx])
                            tmp_y.append(Y_flat[idxx])
                            tmp_t.append(T_flat[idxx])
                    ind_clust[track_id] = np.append(ind_clust[track_id],np.array(tmp_ind))
                    x_clust[track_id]   = np.append(x_clust[track_id], np.array(tmp_x))
                    y_clust[track_id]   = np.append(y_clust[track_id], np.array(tmp_y))
                    t_clust[track_id]   = np.append(t_clust[track_id], np.array(tmp_t))
                
            elif pixelwise_extension:
                XY = np.column_stack((X-1,Y-1))
                new_mask = np.zeros((height, width), dtype=bool)
                new_mask[X-1,Y-1] = True
                labeled, num_regions = label(new_mask)
                
                region_sizes = np.bincount(labeled.ravel())
                keep = region_sizes >= N_pixelwise
                labeled[~keep[labeled]] = 0
                new_labels = np.unique(labeled)
                new_labels = new_labels[new_labels != 0]

                region_coords = defaultdict(list)
                region_indices = defaultdict(list) 
                
                xs, ys = np.nonzero(labeled > 0)
                for i, (x, y) in enumerate(zip(xs, ys)):
                    label_0 = labeled[x, y]
                    if label_0 in new_labels:
                        region_coords[label_0].append((x, y))
                pixel_to_indices = defaultdict(list)
                for i, (x_, y_) in enumerate(XY):
                    pixel_to_indices[(x_, y_)].append(i)
                for label_0 in region_coords:
                    coords = region_coords[label_0]
                    indices = []
                    for x, y in coords:
                        indices.extend(pixel_to_indices[(x, y)])  
                    region_coords[label_0] = np.array(coords, dtype=np.uint16)
                    region_indices[label_0] = np.array(indices, dtype=np.uint32)
                
                active_labelmap = np.zeros_like(new_mask, dtype=np.int32)  
                active_labelmap[:, :] = -1  
                
                for i, region in enumerate(active_pixels):
                    if len(region) == 0 or valid[i] == 0:
                        continue
                    active_labelmap[region[:, 0], region[:, 1]] = i
                    
                active_temp =  [[] for _ in range(len(active_pixels))]
                ind_temp = []
                    
                for label_0 in new_labels:
                    coords = region_coords[label_0]
                    if coords.shape[0] == 0:
                        continue    
                    mean_coords = np.mean(coords, axis=0)
                    mean_x, mean_y = mean_coords[0], mean_coords[1]

                    if multi==1:
                        max_x = 1280
                        max_y = 720
                        
                        coords_i = coords.astype(np.int32, copy=False)  
                        
                        min_x = max(int(coords_i[:, 0].min()) - 1, 0)
                        max_x_roi = min(int(coords_i[:, 0].max()) + 2, max_x)
                        min_y = max(int(coords_i[:, 1].min()) - 1, 0)
                        max_y_roi = min(int(coords_i[:, 1].max()) + 2, max_y)

                        if max_x_roi <= min_x or max_y_roi <= min_y:
                            continue
                        
                        mask = np.zeros((max_x_roi - min_x, max_y_roi - min_y), dtype=bool)
                        
                        coords_roi = coords_i.copy()
                        coords_roi[:, 0] -= min_x
                        coords_roi[:, 1] -= min_y
                        
                        mask[coords_roi[:, 0], coords_roi[:, 1]] = True
                        mask_dilated = mask
                        
                        coords_exp = np.argwhere(mask_dilated).astype(np.int32)
                        coords_exp[:, 0] += min_x
                        coords_exp[:, 1] += min_y
                        
                        i_labels = active_labelmap[coords_exp[:, 0], coords_exp[:, 1]]
                    else:
                        i_labels = active_labelmap[coords[:, 0], coords[:, 1]]
                    i_labels = i_labels[i_labels >= 0] 
                    
                    if len(i_labels) == 0:
                        continue
                    
                    unique_clusters, counts = np.unique(i_labels, return_counts=True)
                    
                    if len(unique_clusters) == 1:
                        i_best = unique_clusters[0]
                        xx = region_coords[label_0]
                        idx = np.array(region_indices[label_0])
                    
                        x_m, y_m, t_m, x_clust, y_clust, t_clust, steps, inactive_k, ind_clust = update_cluster(
                            x_m, y_m, t_m, i_best, X, Y, T, x_clust, y_clust, t_clust, idx,
                            del_, steps, buffer, steps_rev, inactive_k, k, lightweight_mode, num_points, ind_clust, ind_global
                        )
                    
                        mask = t_clust[i_best] >= 0
                        x_clust[i_best] = x_clust[i_best][mask]
                        y_clust[i_best] = y_clust[i_best][mask]
                        t_clust[i_best] = t_clust[i_best][mask]
                        ind_clust[i_best] = ind_clust[i_best][mask]
                        
                        active_labelmap[active_labelmap==i_best]=-1
                    
                        active_temp[i_best].append(xx)
                        ind_temp.append(i_best)
                        labeled[xx[:, 0], xx[:, 1]] = 0
                    else:
                        cluster_centers = {
                                i: np.mean(active_pixels[i], axis=0)
                                for i in unique_clusters
                            }
                        
                        keys = list(cluster_centers.keys())
                        centers = np.array(list(cluster_centers.values()))
                        dist_matrix = squareform(pdist(centers))
                        for i in range(len(keys)):
                            for j in range(len(keys)):
                                if i != j and dist_matrix[i, j] < 4:
                                    centers[j] = np.array([-1000, -1000])
                        cluster_centers = dict(zip(keys, centers))
                        
                        assigned_labels = []
                        for x_var, y_var in coords:
                            dists = {
                                i: (x_var - cx)**2 + (y_var - cy)**2
                                for i, (cx, cy) in cluster_centers.items()
                            }
                            assigned_label = unique_clusters[np.argmin(list(dists.values()))]
                            assigned_labels.append((x_var,y_var, assigned_label))
                            
                        label_counts = collections.Counter([label for (_, _, label) in assigned_labels])
                        assigned_labels = [
                            (x, y, label if label_counts[label] >= 4 else 0)
                            for (x, y, label) in assigned_labels
                        ]
                        unique_labels = list(set(label for (_, _, label) in assigned_labels))
                            
                        if len(unique_labels)>1:
                            k
                        for assignit in range(len(unique_labels)):
                            if unique_labels[assignit]==0:
                                continue
                            xx = region_coords[label_0]
                            idx = np.array(region_indices[label_0])
                            indices = [i for i, (_, _, label) in enumerate(assigned_labels) if label == unique_labels[assignit]]
                            
                            xx = xx[indices]
                            idx = idx[indices]
                            i_best = unique_labels[assignit]
                            x_m, y_m, t_m, x_clust, y_clust, t_clust, steps, inactive_k, ind_clust = update_cluster(
                                x_m, y_m, t_m, i_best, X, Y, T, x_clust, y_clust, t_clust, idx,
                                del_, steps, buffer, steps_rev, inactive_k, k, lightweight_mode, num_points, ind_clust, ind_global
                            )
                            
                            mask = t_clust[i_best] >= 0
                            x_clust[i_best] = x_clust[i_best][mask]
                            y_clust[i_best] = y_clust[i_best][mask]
                            t_clust[i_best] = t_clust[i_best][mask]
                            ind_clust[i_best] = ind_clust[i_best][mask]
                            
                            active_labelmap[active_labelmap==i_best]=-1
                            active_temp[i_best].append(xx)
                            ind_temp.append(i_best)
                            labeled[xx[:, 0], xx[:, 1]] = 0
                        
                ind_temp = np.unique(ind_temp)
                for ii in range(len(active_pixels)):
                    if valid[ii]==0:
                        continue
                    if ii in ind_temp:
                        if len(active_temp[ii])>1:
                            active_pixels[ii] = np.concatenate(active_temp[ii], axis=0)
                        else:
                            active_pixels[ii] = active_temp[ii][0]
                    else:
                        valid[ii] = 0
                        inactive_k[ii] = k
                    
            elif db_clustering:
                if len(X)<1:
                    idx = []
                else:
                    idx = clustering(apply_hdbscan=apply_hdbscan,epsilon=epsilon,minPts=minPts,X=X,Y=Y)
                
                    valid_clusters = idx != -1
                    X = X[valid_clusters]
                    Y = Y[valid_clusters]
                    T = T[valid_clusters]
                    ind_global = ind_global[valid_clusters]
                    idx = idx[valid_clusters]
    
                max_cluster = idx.max() if len(idx) > 0 else -1
                for i in range(max_cluster):
                    mask = idx == i
                    x_temp[i] = np.median(X[mask])
                    y_temp[i] = np.median(Y[mask])
                    
                nearest_indices, x_clust, y_clust, x_m, y_m, t_m, steps, inactive_k, valid, t_clust, num_involved, ind_clust = db_matching(
                    valid,x_m,y_m,t_m,x_temp,y_temp,max_tracks,threshold,idx,del_,steps,buffer,steps_rev,X,Y,T,x_clust,y_clust,t_clust,k,inactive_k,lightweight_mode,num_points,num_involved,ind_clust, ind_global)
                
                x_temp = np.zeros(max_tracks)
                y_temp = np.zeros(max_tracks)
    
            else:  
                
                x_m, y_m, t_m, x_clust, y_clust, t_clust, steps, inactive_k, valid, X, Y, T, num_involved,ind_clust,ind_global = kdtree_clustering(
                    X,Y,T,valid,x_m,y_m,t_m,Range,Lmin,steps,del_,buffer,steps_rev,inactive_k,k,x_clust,y_clust,t_clust,lightweight_mode,num_points,num_involved,ind_clust,ind_global)
            
                        
            if max_tracks > 0 and save_mode==False:
                if X.size > 0 and np.sum(inactive) > 0:
                    idx = subgrid_search(height,width,search_factor,minPts,inactive,X,Y)
    
                    if np.any(idx):
                        X = X[idx]
                        Y = Y[idx]
                        T = T[idx]
                        idx = clustering(apply_hdbscan=apply_hdbscan,epsilon=epsilon,minPts=minPts,X=X,Y=Y)
                        idx[idx > np.sum(inactive)] = -1
                    else:
                        idx = np.array([])
                else:
                    idx = np.array([])
            else:
                if pseudo_images:
                    unmatched_b = set(range(len(centroids))) - matched_b
                    idx = []
                    for idx_unmatched in unmatched_b:
                        idx.append(idx_unmatched)
                    idx = np.array(idx)
                elif pixelwise_extension==True:
                    if np.max(labeled) > 0:
                        new_labels = np.unique(labeled)
                        new_labels = new_labels[new_labels != 0]
                        idx = np.full(len(X), -1)

                        m=0
                        for region_id in new_labels:
                            all_indices = []
                            for coord in region_coords[region_id]:
                                all_indices.extend(pixel_to_indices.get(tuple(coord), []))
                            if np.count_nonzero(T[all_indices] > 0) > 5:
                                idx[all_indices] = region_id
                                m+=1
                        if m==0:
                            idx = np.array([])
                    else:
                        idx = np.array([])
    
                elif X.size > 0 and db_clustering==False:
                    idx = clustering(apply_hdbscan=apply_hdbscan,epsilon=epsilon,minPts=minPts,X=X,Y=Y)
                elif X.size > 0 and db_clustering==True:
                    unmatched_indices = np.where(nearest_indices == -1)[0]
                    idx[~np.isin(idx, unmatched_indices)] = -1
                else:
                    idx = np.array([])
    
            if idx.size > 0 and pseudo_images==False:
                valid_idx = idx != -1
                X = X[valid_idx]
                Y = Y[valid_idx]
                T = T[valid_idx]
                ind_global = ind_global[valid_idx]
                idx = idx[valid_idx]
                
    
            if idx.size > 0:
                refil = np.where(inactive == 1)[0]
                if pseudo_images:
                    needed = len(idx)
                else:
                    needed = max(idx)
                missing = needed - len(refil)
                
                if missing > 0:
                    extend_size = missing
                    x_clust = extend_list_of_arrays(x_clust, extend_size, (max_length,))
                    y_clust = extend_list_of_arrays(y_clust, extend_size, (max_length,))
                    t_clust = extend_list_of_arrays(t_clust, extend_size, (max_length,))
                    ind_clust = extend_list_of_arrays(ind_clust, extend_size, (max_length,))
                    active_pixels = extend_list_of_arrays(active_pixels, extend_size, (max_length,))
                    P_new = extend_list_of_arrays(P_new, extend_size, (3,3))
                    P_newz = extend_list_of_arrays(P_newz, extend_size, (3,3))
                    x_est = extend_list_of_arrays(x_est, extend_size, (3,))
                    y_est = extend_list_of_arrays(y_est, extend_size, (3,))
                
                    if lightweight_mode or pseudo_images:
                        x_m = extend_list_of_arrays(x_m, extend_size, (max_length,))
                        y_m = extend_list_of_arrays(y_m, extend_size, (max_length,))
                        t_m = extend_list_of_arrays(t_m, extend_size, (max_length,))
                    else:
                        x_m = extend_1d(x_m, extend_size)
                        y_m = extend_1d(y_m, extend_size)
                        t_m = extend_1d(t_m, extend_size)
                    valid = extend_1d(valid, extend_size, 0)
                    steps = extend_1d(steps, extend_size, 0)
                    num_involved = extend_1d(num_involved, extend_size, 0)
                    L_before = extend_1d(L_before, extend_size, 0)
                    del_ = extend_1d(del_, extend_size, 0)
                    inactive = extend_1d(inactive, extend_size, 1)
                    inactive_k = extend_1d(inactive_k, extend_size, -1)
                    max_tracks = max_tracks+int(extend_size)

                    refil = np.where(inactive == 1)[0]
                
                m=0
                if pseudo_images:
                    lookup = defaultdict(list)
                    for j, (x, y) in enumerate(zip(X.ravel(), Y.ravel())):
                        lookup[(x, y)].append(j)
                for i in range(1, max(idx) + 1):
                    if i not in idx:
                        continue
                    if pseudo_images:
                        clust_idx = i
                    else:
                        clust_idx = idx == i
                    r = refil[m]
                    if pseudo_images:
                        tmp_ind = []
                        tmp_x   = []
                        tmp_y   = []
                        tmp_t   = []
                        for x, y in coords[clust_idx]:
                            for idxx in lookup.get((x+1, y+1), []):
                                tmp_ind.append(ind_global[idxx])
                                tmp_x.append(X_flat[idxx])
                                tmp_y.append(Y_flat[idxx])
                                tmp_t.append(T_flat[idxx])
                        ind_clust[r] = np.array(tmp_ind)
                        x_clust[r]   = np.array(tmp_x)
                        y_clust[r]   = np.array(tmp_y)
                        t_clust[r]   = np.array(tmp_t)
                    else:
                        x_clust[r] = X[clust_idx]
                        y_clust[r] = Y[clust_idx]
                        t_clust[r] = T[clust_idx]
                        ind_clust[r] = ind_global[clust_idx]
                    if pseudo_images:
                        x_m[r]=np.array([centroids[clust_idx][0]])
                        y_m[r]=np.array([centroids[clust_idx][1]])
                        t_m[r]=np.array([(T_min+dt/2+(k)*dt)*1e-6])
                        num_involved[r] = len(x_m[r])
                    elif lightweight_mode:
                        t_min, t_max = t_clust[r].min(), t_clust[r].max()
                        bins = np.linspace(t_min, t_max, num=num_points+1)
                        x_m[r]=[]
                        y_m[r]=[]
                        t_m[r]=[]
                        for j in range(num_points):
                            mask = np.where((t_clust[r]>=bins[j]) & (t_clust[r]<=bins[j+1]))
                            if mask[0].size > 0:
                                x_m[r] = np.append(x_m[r], np.median(x_clust[r][mask]))
                                y_m[r] = np.append(y_m[r], np.median(y_clust[r][mask]))
                                t_m[r] = np.append(t_m[r], np.median(t_clust[r][mask])* 1e-6)
                        num_involved[r] = len(x_m[r])
                    elif pixelwise_extension:
                        mask = t_clust[r] >= 0
                        x_clust[r] = x_clust[r][mask]
                        y_clust[r] = y_clust[r][mask]
                        t_clust[r] = t_clust[r][mask]
                        ind_clust[r] = ind_clust[r][mask]
                        active_pixels[r] = region_coords[np.int32(i)]
                        x_m[r] = np.median(x_clust[r])
                        y_m[r] = np.median(y_clust[r])
                        t_m[r] = np.median(t_clust[r])* 1e-6
                    else:
                        x_m[r] = np.median(x_clust[r])
                        y_m[r] = np.median(y_clust[r])
                        t_m[r] = np.median(t_clust[r])* 1e-6
                    valid[r] = 1
                    steps[r] = 1
                    del_[r] = 0
                    inactive[r] = 0
                    inactive_k[r] = k
                    m+=1        
    
            for i in range(max_tracks):
                if pseudo_images == False:
                    if len(t_clust[i][L_before[i]:])>1:
                        L_before[i] = len(x_clust[i])

                if steps[i]<buffer and (inactive_k[i] + 2) < k:
                    inactive[i] = 1

        del_[:] = 1
        
        tic_end = time.time()
        if k % 10 == 0:
            print(f"Loop {k} completed in {tic_end - tic_start} seconds")
        if stop_event is not None and stop_event.is_set():
            break

    if lightweight_mode==False:
        # for i in range(3):
            x_clust, y_clust, t_clust, steps, ind_clust = repair_tracks(x_clust, y_clust, t_clust, ind_clust, steps,dt)
    else:
        # for i in range(3):
            x_clust, y_clust, t_clust, steps, ind_clust, x_m, y_m, t_m = repair_tracks_pseudoframes(x_clust, y_clust, t_clust, ind_clust, steps, dt,x_m,y_m,t_m)


    if multi==1:
        n_events = [len(a) for a in ind_clust]
        if lightweight_mode or pseudo_images:
            for i in range(len(ind_clust)):
                if n_events[i]<20:
                    x_clust[i] = np.array([], dtype=int)
                    y_clust[i] = np.array([], dtype=int)
                    t_clust[i] = np.array([], dtype=int)
                    ind_clust[i] = np.array([], dtype=int)
                    x_m[i] = np.array([])
                    y_m[i] = np.array([])
                    t_m[i] = np.array([])
        else:
            for i in range(len(ind_clust)):
                if n_events[i]<20:
                    x_clust[i] = np.array([], dtype=int)
                    y_clust[i] = np.array([], dtype=int)
                    t_clust[i] = np.array([], dtype=int)
                    ind_clust[i] = np.array([], dtype=int)
    
    if multi==0:
        if lightweight_mode == False:
            n_events = [len(a) for a in x_clust]
            if not test_mode:
                for i in range(len(ind_clust)):
                    if steps[i] < 3 or n_events[i]<20:
                        x_clust[i] = np.array([], dtype=int)
                        y_clust[i] = np.array([], dtype=int)
                        t_clust[i] = np.array([], dtype=int)
                        ind_clust[i] = np.array([], dtype=int)
            ind_all = np.concatenate(ind_clust)
            ind_all = ind_all.astype(int)
            mask = np.ones(len(X_global), dtype=bool)
            mask[ind_all] = False
            X_global = X_global[mask]
            Y_global = Y_global[mask]
            T_global = T_global[mask]
            
            
            x_clust_fast = [arr.copy() for arr in x_clust]
            y_clust_fast = [arr.copy() for arr in y_clust]
            t_clust_fast = [arr.copy() for arr in t_clust]
            ind_clust_fast = [arr.copy() for arr in ind_clust]
            
            if multirun:
                dt = multitimefactor*dt
                N = int(N//multitimefactor)
                minPts = minPts*multitimefactor
            
            x_clust = [np.array([]) for _ in range(max_tracks)]
            y_clust = [np.array([]) for _ in range(max_tracks)]
            t_clust = [np.array([]) for _ in range(max_tracks)]
            ind_clust = [np.array([]) for _ in range(max_tracks)]
            active_pixels = [[] for _ in range(max_tracks)]
            x_est = [[] for _ in range(max_tracks)]
            y_est = [[] for _ in range(max_tracks)]
            x_temp = np.zeros(max_tracks)
            y_temp = np.zeros(max_tracks)
            inactive_k = np.ones(max_tracks, dtype=np.int32)
            del_ = np.zeros(max_tracks, dtype=np.int32)
            steps = np.zeros(max_tracks, dtype=np.int32)
            valid = np.ones(max_tracks, dtype=np.int32)
            inactive = np.ones(max_tracks, dtype=np.int32)
            L_before = np.zeros(max_tracks, dtype=np.int32)
            num_involved = np.zeros(max_tracks, dtype=np.int32)
            x_m = np.zeros(max_tracks)
            y_m = np.zeros(max_tracks)
            t_m = np.zeros(max_tracks)
        else:
            n_events = [len(a) for a in ind_clust]
            if not test_mode:
                for i in range(len(ind_clust)):
                    if len(x_m[i]) < 3 or n_events[i]<40:
                        x_clust[i] = np.array([], dtype=int)
                        y_clust[i] = np.array([], dtype=int)
                        t_clust[i] = np.array([], dtype=int)
                        ind_clust[i] = np.array([], dtype=int)
                        x_m[i] = np.array([])
                        y_m[i] = np.array([])
                        t_m[i] = np.array([])
            ind_all = np.concatenate(ind_clust)
            ind_all = ind_all.astype(int)
            mask = np.ones(len(X_global), dtype=bool)
            mask[ind_all] = False
            X_global = X_global[mask]
            Y_global = Y_global[mask]
            T_global = T_global[mask]
            
            x_m_fast = [arr.copy() for arr in x_m]
            y_m_fast = [arr.copy() for arr in y_m]
            t_m_fast = [arr.copy() for arr in t_m]
            x_clust_fast = [arr.copy() for arr in x_clust]
            y_clust_fast = [arr.copy() for arr in y_clust]
            t_clust_fast = [arr.copy() for arr in t_clust]
            ind_clust_fast = [arr.copy() for arr in ind_clust]
            
            if multirun:
                dt = multitimefactor*dt
                N = int(N//multitimefactor)
            
            x_clust = [[] for _ in range(max_tracks)]
            y_clust = [[] for _ in range(max_tracks)]
            t_clust = [[] for _ in range(max_tracks)]
            ind_clust = [[] for _ in range(max_tracks)]
            active_pixels = [[] for _ in range(max_tracks)]
            x_est = [[] for _ in range(max_tracks)]
            y_est = [[] for _ in range(max_tracks)]
            x_temp = np.zeros(max_tracks)
            y_temp = np.zeros(max_tracks)
            inactive_k = np.ones(max_tracks, dtype=np.int32)
            del_ = np.zeros(max_tracks, dtype=np.int32)
            steps = np.zeros(max_tracks, dtype=np.int32)
            valid = np.ones(max_tracks, dtype=np.int32)
            inactive = np.ones(max_tracks, dtype=np.int32)
            L_before = np.zeros(max_tracks, dtype=np.int32)
            num_involved = np.zeros(max_tracks, dtype=np.int32)
            x_m = [np.array([]) for _ in range(max_tracks)]
            y_m = [np.array([]) for _ in range(max_tracks)]
            t_m = [np.array([]) for _ in range(max_tracks)]

x_clust = [np.asarray(a) for a in x_clust]
y_clust = [np.asarray(a) for a in y_clust]
t_clust = [np.asarray(a) for a in t_clust]
ind_clust = [np.asarray(a) for a in ind_clust]

if "x_clust_fast" in locals():
    x_clust_fast = [np.asarray(a) for a in x_clust_fast]
    y_clust_fast = [np.asarray(a) for a in y_clust_fast]
    t_clust_fast = [np.asarray(a) for a in t_clust_fast]
    ind_clust_fast = [np.asarray(a) for a in ind_clust_fast]

if lightweight_mode == False:         
    x_clust = [arr for arr in x_clust if arr.size > 0] + \
             [arr for arr in x_clust_fast if arr.size > 0]
    y_clust = [arr for arr in y_clust if arr.size > 0] + \
             [arr for arr in y_clust_fast if arr.size > 0]
    t_clust = [arr for arr in t_clust if arr.size > 0] + \
             [arr for arr in t_clust_fast if arr.size > 0]
    ind_clust = [arr for arr in ind_clust if arr.size > 0] + \
             [arr for arr in ind_clust_fast if arr.size > 0]
else:
    LLmin = 0  
    x_comb = x_clust + x_clust_fast
    y_comb = y_clust + y_clust_fast
    t_comb = t_clust + t_clust_fast
    x_m_comb = x_m + x_m_fast
    y_m_comb = y_m + y_m_fast
    t_m_comb = t_m + t_m_fast
    
    x_clust, y_clust, t_clust, x_m, y_m, t_m = zip(*[
        (xc, yc, tc, xm, ym, tm)
        for xc, yc, tc, xm, ym, tm in zip(x_comb, y_comb, t_comb, x_m_comb, y_m_comb, t_m_comb)
        if len(xm) > LLmin
    ])

    x_clust, y_clust, t_clust = map(list, (x_clust, y_clust, t_clust))
    x_m, y_m, t_m = map(list, (x_m, y_m, t_m))

if multirun == True:
    dt = int(dt/multitimefactor)
if lightweight_mode==False:
    # for i in range(3):
        x_clust, y_clust, t_clust, steps, ind_clust = repair_tracks(x_clust, y_clust, t_clust, ind_clust, steps,dt)
else:
    # for i in range(3):
        x_clust, y_clust, t_clust, steps, ind_clust, x_m, y_m, t_m = repair_tracks_pseudoframes(x_clust, y_clust, t_clust, ind_clust, steps, dt,x_m,y_m,t_m)

if not test_mode:
    if lightweight_mode == False:
        x_clust = [arr for arr in x_clust if arr.size > 0]
        y_clust = [arr for arr in y_clust if arr.size > 0]
        t_clust = [arr for arr in t_clust if arr.size > 0]
    else:
        LLmin = 2
        x_clust, y_clust, t_clust, x_m, y_m, t_m = zip(*[
            (xc, yc, tc, xm, ym, tm)
            for xc, yc, tc, xm, ym, tm in zip(x_clust, y_clust, t_clust, x_m, y_m, t_m)
            if len(xm) > LLmin
        ])
    
        x_clust, y_clust, t_clust = map(list, (x_clust, y_clust, t_clust))
        x_m, y_m, t_m = map(list, (x_m, y_m, t_m))
        for i in range(len(x_m)):
            x_m[i] = x_m[i]+1
            y_m[i] = y_m[i]+1
            t_m[i] = t_m[i]*1e6

if pseudo_images:
    t_m = [arr[~np.isnan(arr)] for arr in t_m]
    x_m = [arr[~np.isnan(arr)] for arr in x_m]
    y_m = [arr[~np.isnan(arr)] for arr in y_m]    
