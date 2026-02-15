import numpy as np
from numba import jit

#used
@jit(nopython=True,cache=True)
def Kalman_v2_both_optimized(x, y, z, start_id, C, Q, R, P, x_est, Py, y_est, rev):
    n = len(x) - start_id

    y_filtered = np.zeros(n)
    y_velocity = np.zeros(n)
    y_acceleration = np.zeros(n)
    z_filtered = np.zeros(n)
    z_velocity = np.zeros(n)
    z_acceleration = np.zeros(n)

    start_id = int(start_id)

    I = np.eye(3, dtype=np.float64)
    CT = C.T
    for i in range(start_id, len(x)):
        idx = i - start_id
        dt = x[i] - x[i-1]  
        dt = dt* 1e-6

        A = np.zeros((3, 3), dtype=np.float64)
        A[0, 0] = 1.0
        A[0, 1] = dt
        A[0, 2] = 0.5 * dt * dt
        A[1, 1] = 1.0
        A[1, 2] = dt
        A[2, 2] = 1.0

        x_est = np.dot(A, x_est)
        P = np.dot(np.dot(A, P), A.T) + Q

        PC = np.dot(P, CT)
        S = np.dot(C, PC) + R  
        K = PC * (1.0 / S)  

        measurement_residual = y[i] - np.dot(C, x_est)
        x_est = x_est + K * measurement_residual
        P = np.dot((I - np.outer(K, C)), P)

        y_filtered[idx] = x_est[0]
        y_velocity[idx] = x_est[1]
        y_acceleration[idx] = x_est[2]

        y_est = np.dot(A, y_est)
        Py = np.dot(np.dot(A, Py), A.T) + Q

        PC = np.dot(Py, CT)
        S = np.dot(C, PC) + R  
        K = PC * (1.0 / S)  
        
        measurement_residual = z[i] - np.dot(C, y_est)
        y_est = y_est + K * measurement_residual
        Py = np.dot((I - np.outer(K, C)), Py)

        z_filtered[idx] = y_est[0]
        z_velocity[idx] = y_est[1]
        z_acceleration[idx] = y_est[2]

    t_query = np.array([x[start_id], x[-1]])
    x_interp = np.array([y_filtered[0], y_filtered[-1]])
    x_interp_v = np.array([y_velocity[0], y_velocity[-1]])
    x_interp_a = np.array([y_acceleration[0], y_acceleration[-1]])
    y_interp = np.array([z_filtered[0], z_filtered[-1]])
    y_interp_v = np.array([z_velocity[0], z_velocity[-1]])
    y_interp_a = np.array([z_acceleration[0], z_acceleration[-1]])
    return P, x_est, Py, y_est, t_query, x_interp, y_interp, x_interp_v, y_interp_v, x_interp_a, y_interp_a

#used
@jit(nopython=True,cache=True)
def Kalman_v2_both_buffer_optimized_lightweight_afterwards(x, y, z, start_id, C, Q, R, P, x_est, Py, y_est, rev, dtt,factor):
    n = len(x) - start_id

    y_filtered = np.zeros(n)
    y_velocity = np.zeros(n)
    y_acceleration = np.zeros(n)
    z_filtered = np.zeros(n)
    z_velocity = np.zeros(n)
    z_acceleration = np.zeros(n)

    start_id = int(start_id)

    I = np.eye(3, dtype=np.float64)

    CT = C.T
    for i in range(start_id, len(x)):
        idx = i - start_id
        if i == 0:
            dt = x[1] - x[0]
            dt = dt*1e-6
        else:
            dt = x[i] - x[i-1]  
            dt = dt * 1e-6
        A = np.zeros((3, 3), dtype=np.float64)
        A[0, 0] = 1.0
        A[0, 1] = dt
        A[0, 2] = 0.5 * dt * dt
        A[1, 1] = 1.0
        A[1, 2] = dt
        A[2, 2] = 1.0

        x_est = np.dot(A, x_est)
        P = np.dot(np.dot(A, P), A.T) + Q

        PC = np.dot(P, CT)
        S = np.dot(C, PC) + R  
        K = PC * (1.0 / S)  

        measurement_residual = y[i] - np.dot(C, x_est)
        x_est = x_est + K * measurement_residual
        P = np.dot((I - np.outer(K, C)), P)

        y_filtered[idx] = x_est[0]
        y_velocity[idx] = x_est[1]
        y_acceleration[idx] = x_est[2]

        y_est = np.dot(A, y_est)
        Py = np.dot(np.dot(A, Py), A.T) + Q

        PC = np.dot(Py, CT)
        S = np.dot(C, PC) + R  
        K = PC * (1.0 / S)  
        
        measurement_residual = z[i] - np.dot(C, y_est)
        y_est = y_est + K * measurement_residual
        Py = np.dot((I - np.outer(K, C)), Py)

        z_filtered[idx] = y_est[0]
        z_velocity[idx] = y_est[1]
        z_acceleration[idx] = y_est[2]

    t_min = x[start_id]
    t_max = x[-1]
    dt_target = dtt / factor
    total_steps = int((t_max - t_min) / dt_target) + 1
    t_query = np.zeros(total_steps)

    x_interp = np.zeros(total_steps)
    x_interp_v = np.zeros(total_steps)
    x_interp_a = np.zeros(total_steps)
    y_interp = np.zeros(total_steps)
    y_interp_v = np.zeros(total_steps)
    y_interp_a = np.zeros(total_steps)

    last_time = x[start_id]
    t_query[0] = last_time
    x_interp[0] = y_filtered[0]
    x_interp_v[0] = y_velocity[0]
    x_interp_a[0] = y_acceleration[0]
    y_interp[0] = z_filtered[0]
    y_interp_v[0] = z_velocity[0]
    y_interp_a[0] = z_acceleration[0]

    idx = 1
    for i in range(1, len(x) - start_id):
        dt_meas = (x[start_id + i] - x[start_id + i - 1]) * 1e-6
        sub_steps = int(np.round(dt_meas / (dt_target * 1e-6)))

        x_state = np.array([y_filtered[i - 1], y_velocity[i - 1], y_acceleration[i - 1]])
        y_state = np.array([z_filtered[i - 1], z_velocity[i - 1], z_acceleration[i - 1]])

        for k in range(sub_steps):
            if idx >= total_steps:
                break
            last_time += dt_target
            A_sub = np.array([
                [1.0, dt_target * 1e-6, 0.5 * (dt_target * 1e-6) ** 2],
                [0.0, 1.0, dt_target * 1e-6],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)

            x_state = np.dot(A_sub, x_state)
            y_state = np.dot(A_sub, y_state)

            t_query[idx] = last_time
            x_interp[idx] = x_state[0]
            x_interp_v[idx] = x_state[1]
            x_interp_a[idx] = x_state[2]
            y_interp[idx] = y_state[0]
            y_interp_v[idx] = y_state[1]
            y_interp_a[idx] = y_state[2]
            idx += 1
    t_query = t_query[:idx]
    x_interp = x_interp[:idx]
    y_interp = y_interp[:idx]
    x_interp_v = x_interp_v[:idx]
    y_interp_v = y_interp_v[:idx]
    x_interp_a = x_interp_a[:idx]
    y_interp_a = y_interp_a[:idx]

    return (
        P, x_est, Py, y_est,
        t_query, x_interp, y_interp,
        x_interp_v, y_interp_v,
        x_interp_a, y_interp_a
    )

#used
@jit(nopython=True,cache=True)
def Kalman_v2_both_buffer_optimized_afterwards(x, y, z, start_id, C, Q, R, P, x_est, Py, y_est, rev, dtt):
    n = len(x) - start_id

    y_filtered = np.zeros(n)
    y_velocity = np.zeros(n)
    y_acceleration = np.zeros(n)
    z_filtered = np.zeros(n)
    z_velocity = np.zeros(n)
    z_acceleration = np.zeros(n)

    start_id = int(start_id)
    rev = int(rev)

    I = np.eye(3, dtype=np.float64)
    CT = C.T
    
    if rev == 0:  
        for i in range(start_id, len(x)):
            idx = i - start_id
            dt = x[i] - x[i-1]  
            dt = dt * 1e-6

            A = np.zeros((3, 3), dtype=np.float64)
            A[0, 0] = 1.0
            A[0, 1] = dt
            A[0, 2] = 0.5 * dt * dt
            A[1, 1] = 1.0
            A[1, 2] = dt
            A[2, 2] = 1.0

            x_est = np.dot(A, x_est)
            P = np.dot(np.dot(A, P), A.T) + Q

            PC = np.dot(P, CT)
            S = np.dot(C, PC) + R  
            K = PC * (1.0 / S)  

            measurement_residual = y[i] - np.dot(C, x_est)
            x_est = x_est + K * measurement_residual
            P = np.dot((I - np.outer(K, C)), P)
            
            y_filtered[idx] = x_est[0]
            y_velocity[idx] = x_est[1]
            y_acceleration[idx] = x_est[2]

            y_est = np.dot(A, y_est)
            Py = np.dot(np.dot(A, Py), A.T) + Q

            PC = np.dot(Py, CT)
            S = np.dot(C, PC) + R  
            K = PC * (1.0 / S)  
            
            measurement_residual = z[i] - np.dot(C, y_est)
            y_est = y_est + K * measurement_residual
            Py = np.dot((I - np.outer(K, C)), Py)

            z_filtered[idx] = y_est[0]
            z_velocity[idx] = y_est[1]
            z_acceleration[idx] = y_est[2]
        step = np.maximum(1,int(np.ceil((max(x)-min(x))/(dtt/4))))
        indices = np.empty(step, dtype=np.int64)
        for i in range(step):
            indices[i] = int(i * n / step)
        t_query = np.zeros(len(indices))
        x_interp = np.zeros(len(indices))
        x_interp_v = np.zeros(len(indices))
        x_interp_a = np.zeros(len(indices))
        y_interp = np.zeros(len(indices))
        y_interp_v = np.zeros(len(indices))
        y_interp_a = np.zeros(len(indices))
        
        for j in range(len(indices)):
            idx = indices[j]
            t_query[j] = x[start_id-1 + idx]
            x_interp[j] = y_filtered[idx]
            x_interp_v[j] = y_velocity[idx]
            x_interp_a[j] = y_acceleration[idx]
            y_interp[j] = z_filtered[idx]
            y_interp_v[j] = z_velocity[idx]
            y_interp_a[j] = z_acceleration[idx]
    
    else:  
        x_est[1] = -x_est[1]
        x_est[2] = -x_est[2]
        y_est[1] = -y_est[1]
        y_est[2] = -y_est[2]
        
        for i in range(len(x)-1, start_id-1, -1):
            idx = i - start_id
            dt = x[i] - x[i-1]

            A = np.zeros((3, 3), dtype=np.float64)
            A[0, 0] = 1.0
            A[0, 1] = dt
            A[0, 2] = 0.5 * dt * dt
            A[1, 1] = 1.0
            A[1, 2] = dt
            A[2, 2] = 1.0

            x_est = np.dot(A, x_est)
            P = np.dot(np.dot(A, P), A.T) + Q
            
            PC = np.dot(P, CT)
            S = np.dot(C, PC) + R  
            K = PC * (1.0 / S)  
            
            measurement_residual = y[i] - np.dot(C, x_est)
            x_est = x_est + K * measurement_residual
            P = np.dot((I - np.outer(K, C)), P)
            
            y_filtered[idx] = x_est[0]
            y_velocity[idx] = -x_est[1]  
            y_acceleration[idx] = -x_est[2] 
            
            y_est = np.dot(A, y_est)
            Py = np.dot(np.dot(A, Py), A.T) + Q
            
            PC = np.dot(Py, CT)
            S = np.dot(C, PC) + R  
            K = PC * (1.0 / S)  
            
            measurement_residual = z[i] - np.dot(C, y_est)
            y_est = y_est + K * measurement_residual
            Py = np.dot((I - np.outer(K, C)), Py)
            
            z_filtered[idx] = y_est[0]
            z_velocity[idx] = -y_est[1]  
            z_acceleration[idx] = -y_est[2] 
        step = max(1, n // (rev * 2))

        indices = np.arange(0, n, step)
        t_query = np.zeros(len(indices))
        x_interp = np.zeros(len(indices))
        x_interp_v = np.zeros(len(indices))
        x_interp_a = np.zeros(len(indices))
        y_interp = np.zeros(len(indices))
        y_interp_v = np.zeros(len(indices))
        y_interp_a = np.zeros(len(indices))
        
        for j in range(len(indices)):
            idx = indices[j]
            t_query[j] = x[start_id + idx]
            x_interp[j] = y_filtered[idx]
            x_interp_v[j] = y_velocity[idx]
            x_interp_a[j] = y_acceleration[idx]
            y_interp[j] = z_filtered[idx]
            y_interp_v[j] = z_velocity[idx]
            y_interp_a[j] = z_acceleration[idx]
    t_query = t_query.reshape(-1, 1)
    
    return P, x_est, Py, y_est, t_query, x_interp, y_interp, x_interp_v, y_interp_v, x_interp_a, y_interp_a


#used
@jit(nopython=True,cache=True)
def Kalman_v2_both_optimized_lightweight_inner(x, y, z, start_id, C, Q, R, P, x_est, Py, y_est, rev,dtt,old_time):
    n = len(x) - start_id
    y_filtered = np.zeros(n)
    y_velocity = np.zeros(n)
    y_acceleration = np.zeros(n)
    z_filtered = np.zeros(n)
    z_velocity = np.zeros(n)
    z_acceleration = np.zeros(n)
    
    start_id = int(start_id)

    I = np.eye(3, dtype=np.float64)
    CT = C.T
    for i in range(start_id, len(x)):
        idx = i - start_id
        if len(x)>1:
            if i == 0:
                dt = dtt
            else:
                dt = x[i] - x[i-1]  
        else:
            dt = dtt
        A = np.zeros((3, 3), dtype=np.float64)
        A[0, 0] = 1.0
        A[0, 1] = dt
        A[0, 2] = 0.5 * dt * dt
        A[1, 1] = 1.0
        A[1, 2] = dt
        A[2, 2] = 1.0

        x_est = np.dot(A, x_est)
        P = np.dot(np.dot(A, P), A.T) + Q

        PC = np.dot(P, CT)
        S = np.dot(C, PC) + R  
        K = PC * (1.0 / S)  

        measurement_residual = y[i] - np.dot(C, x_est)
        x_est = x_est + K * measurement_residual
        P = np.dot((I - np.outer(K, C)), P)

        y_filtered[idx] = x_est[0]
        y_velocity[idx] = x_est[1]
        y_acceleration[idx] = x_est[2]

        y_est = np.dot(A, y_est)
        Py = np.dot(np.dot(A, Py), A.T) + Q

        PC = np.dot(Py, CT)
        S = np.dot(C, PC) + R  
        K = PC * (1.0 / S)  
        
        measurement_residual = z[i] - np.dot(C, y_est)
        y_est = y_est + K * measurement_residual
        Py = np.dot((I - np.outer(K, C)), Py)

        z_filtered[idx] = y_est[0]
        z_velocity[idx] = y_est[1]
        z_acceleration[idx] = y_est[2]
    t_query = x
    x_interp = y_filtered
    x_interp_v = y_velocity
    x_interp_a = y_acceleration
    y_interp = z_filtered
    y_interp_v = z_velocity
    y_interp_a = z_acceleration
    return P, x_est, Py, y_est, t_query, x_interp, y_interp, x_interp_v, y_interp_v, x_interp_a, y_interp_a

@jit(nopython=True,cache=True)
def Kalman_v2_both_buffer_optimized(x, y, z, start_id, C, Q, R, P, x_est, Py, y_est, rev, buffer):
    n = len(x) - start_id

    y_filtered = np.zeros(n)
    y_velocity = np.zeros(n)
    y_acceleration = np.zeros(n)
    z_filtered = np.zeros(n)
    z_velocity = np.zeros(n)
    z_acceleration = np.zeros(n)

    start_id = int(start_id)
    rev = int(rev)
    buffer = int(buffer)

    I = np.eye(3, dtype=np.float64)
    CT = C.T
    if rev == 0:  
        for i in range(start_id, len(x)):
            idx = i - start_id
            dt = x[i] - x[i-1]  
            dt = dt * 1e-6

            A = np.zeros((3, 3), dtype=np.float64)
            A[0, 0] = 1.0
            A[0, 1] = dt
            A[0, 2] = 0.5 * dt * dt
            A[1, 1] = 1.0
            A[1, 2] = dt
            A[2, 2] = 1.0

            x_est = np.dot(A, x_est)
            P = np.dot(np.dot(A, P), A.T) + Q

            PC = np.dot(P, CT)
            S = np.dot(C, PC) + R  
            K = PC * (1.0 / S)  

            measurement_residual = y[i] - np.dot(C, x_est)
            x_est = x_est + K * measurement_residual
            P = np.dot((I - np.outer(K, C)), P)

            y_filtered[idx] = x_est[0]
            y_velocity[idx] = x_est[1]
            y_acceleration[idx] = x_est[2]

            y_est = np.dot(A, y_est)
            Py = np.dot(np.dot(A, Py), A.T) + Q

            PC = np.dot(Py, CT)
            S = np.dot(C, PC) + R  
            K = PC * (1.0 / S)  
            
            measurement_residual = z[i] - np.dot(C, y_est)
            y_est = y_est + K * measurement_residual
            Py = np.dot((I - np.outer(K, C)), Py)

            z_filtered[idx] = y_est[0]
            z_velocity[idx] = y_est[1]
            z_acceleration[idx] = y_est[2]
        step = max(1, n // (buffer *2))

        indices = np.arange(0, n, step)
        t_query = np.zeros(len(indices))
        x_interp = np.zeros(len(indices))
        x_interp_v = np.zeros(len(indices))
        x_interp_a = np.zeros(len(indices))
        y_interp = np.zeros(len(indices))
        y_interp_v = np.zeros(len(indices))
        y_interp_a = np.zeros(len(indices))
        
        for j in range(len(indices)):
            idx = indices[j]
            t_query[j] = x[start_id-1 + idx]
            x_interp[j] = y_filtered[idx]
            x_interp_v[j] = y_velocity[idx]
            x_interp_a[j] = y_acceleration[idx]
            y_interp[j] = z_filtered[idx]
            y_interp_v[j] = z_velocity[idx]
            y_interp_a[j] = z_acceleration[idx]
    
    else:  
        x_est[1] = -x_est[1]
        x_est[2] = -x_est[2]
        y_est[1] = -y_est[1]
        y_est[2] = -y_est[2]
        
        for i in range(len(x)-1, start_id-1, -1):
            idx = i - start_id
            dt = x[i] - x[i-1]

            A = np.zeros((3, 3), dtype=np.float64)
            A[0, 0] = 1.0
            A[0, 1] = dt
            A[0, 2] = 0.5 * dt * dt
            A[1, 1] = 1.0
            A[1, 2] = dt
            A[2, 2] = 1.0

            x_est = np.dot(A, x_est)
            P = np.dot(np.dot(A, P), A.T) + Q
            
            PC = np.dot(P, CT)
            S = np.dot(C, PC) + R  
            K = PC * (1.0 / S)  
            
            measurement_residual = y[i] - np.dot(C, x_est)
            x_est = x_est + K * measurement_residual
            P = np.dot((I - np.outer(K, C)), P)
            
            y_filtered[idx] = x_est[0]
            y_velocity[idx] = -x_est[1]  
            y_acceleration[idx] = -x_est[2]  
            
            y_est = np.dot(A, y_est)
            Py = np.dot(np.dot(A, Py), A.T) + Q
            
            PC = np.dot(Py, CT)
            S = np.dot(C, PC) + R  
            K = PC * (1.0 / S)  
            
            measurement_residual = z[i] - np.dot(C, y_est)
            y_est = y_est + K * measurement_residual
            Py = np.dot((I - np.outer(K, C)), Py)
            
            z_filtered[idx] = y_est[0]
            z_velocity[idx] = -y_est[1]  
            z_acceleration[idx] = -y_est[2]  
        step = max(1, n // (rev * 2))

        indices = np.arange(0, n, step)
        t_query = np.zeros(len(indices))
        x_interp = np.zeros(len(indices))
        x_interp_v = np.zeros(len(indices))
        x_interp_a = np.zeros(len(indices))
        y_interp = np.zeros(len(indices))
        y_interp_v = np.zeros(len(indices))
        y_interp_a = np.zeros(len(indices))
        
        for j in range(len(indices)):
            idx = indices[j]
            t_query[j] = x[start_id + idx]
            x_interp[j] = y_filtered[idx]
            x_interp_v[j] = y_velocity[idx]
            x_interp_a[j] = y_acceleration[idx]
            y_interp[j] = z_filtered[idx]
            y_interp_v[j] = z_velocity[idx]
            y_interp_a[j] = z_acceleration[idx]

    t_query = t_query.reshape(-1, 1)
    return P, x_est, Py, y_est, t_query, x_interp, y_interp, x_interp_v, y_interp_v, x_interp_a, y_interp_a

@jit(nopython=True,cache=True)
def Kalman_v2_both_buffer_optimized_lightweight(x, y, z, start_id, C, Q, R, P, x_est, Py, y_est, rev, buffer):
    n = len(x) - start_id
    y_filtered = np.zeros(n)
    y_velocity = np.zeros(n)
    y_acceleration = np.zeros(n)
    z_filtered = np.zeros(n)
    z_velocity = np.zeros(n)
    z_acceleration = np.zeros(n)

    start_id = int(start_id)
    rev = int(rev)
    buffer = int(buffer)

    I = np.eye(3, dtype=np.float64)
    CT = C.T
    
    if rev == 0:  
        for i in range(start_id, len(x)):
            idx = i - start_id
            if i == 0:
                dt = x[1] - x[0]
            else:
                dt = x[i] - x[i-1]  
                dt = dt 
            A = np.zeros((3, 3), dtype=np.float64)
            A[0, 0] = 1.0
            A[0, 1] = dt
            A[0, 2] = 0.5 * dt * dt
            A[1, 1] = 1.0
            A[1, 2] = dt
            A[2, 2] = 1.0

            x_est = np.dot(A, x_est)
            P = np.dot(np.dot(A, P), A.T) + Q

            PC = np.dot(P, CT)
            S = np.dot(C, PC) + R  
            K = PC * (1.0 / S)  

            measurement_residual = y[i] - np.dot(C, x_est)
            x_est = x_est + K * measurement_residual
            P = np.dot((I - np.outer(K, C)), P)

            y_filtered[idx] = x_est[0]
            y_velocity[idx] = x_est[1]
            y_acceleration[idx] = x_est[2]

            y_est = np.dot(A, y_est)
            Py = np.dot(np.dot(A, Py), A.T) + Q

            PC = np.dot(Py, CT)
            S = np.dot(C, PC) + R  
            K = PC * (1.0 / S)
            
            measurement_residual = z[i] - np.dot(C, y_est)
            y_est = y_est + K * measurement_residual
            Py = np.dot((I - np.outer(K, C)), Py)

            z_filtered[idx] = y_est[0]
            z_velocity[idx] = y_est[1]
            z_acceleration[idx] = y_est[2]  
        t_query = x
        x_interp = y_filtered
        x_interp_v = y_velocity
        x_interp_a = y_acceleration
        y_interp = z_filtered
        y_interp_v = z_velocity
        y_interp_a = z_acceleration
    
    else:  
        x_est[1] = -x_est[1]
        x_est[2] = -x_est[2]
        y_est[1] = -y_est[1]
        y_est[2] = -y_est[2]
        
        for i in range(len(x)-1, start_id-1, -1):
            idx = i - start_id
            dt = 20000

            A = np.zeros((3, 3), dtype=np.float64)
            A[0, 0] = 1.0
            A[0, 1] = dt
            A[0, 2] = 0.5 * dt * dt
            A[1, 1] = 1.0
            A[1, 2] = dt
            A[2, 2] = 1.0

            x_est = np.dot(A, x_est)
            P = np.dot(np.dot(A, P), A.T) + Q
            
            PC = np.dot(P, CT)
            S = np.dot(C, PC) + R  
            K = PC * (1.0 / S)  
            
            measurement_residual = y[i] - np.dot(C, x_est)
            x_est = x_est + K * measurement_residual
            P = np.dot((I - np.outer(K, C)), P)
            
            y_filtered[idx] = x_est[0]
            y_velocity[idx] = -x_est[1]  
            y_acceleration[idx] = -x_est[2]  
            
            y_est = np.dot(A, y_est)
            Py = np.dot(np.dot(A, Py), A.T) + Q
            
            PC = np.dot(Py, CT)
            S = np.dot(C, PC) + R  
            K = PC * (1.0 / S)  
            
            measurement_residual = z[i] - np.dot(C, y_est)
            y_est = y_est + K * measurement_residual
            Py = np.dot((I - np.outer(K, C)), Py)
            
            z_filtered[idx] = y_est[0]
            z_velocity[idx] = -y_est[1]  
            z_acceleration[idx] = -y_est[2]  
        t_query = x
        x_interp = y_filtered
        x_interp_v = y_velocity
        x_interp_a = y_acceleration
        y_interp = z_filtered
        y_interp_v = z_velocity
        y_interp_a = z_acceleration
    return P, x_est, Py, y_est, t_query, x_interp, y_interp, x_interp_v, y_interp_v, x_interp_a, y_interp_a

#used
def Kalman_v2_both_optimized_lightweight(x, y, z, start_id, C, Q, R, P, x_est, Py, y_est, rev, dt, old_time):
    if np.isscalar(x):
        x = np.array([x], dtype=np.float64)
        y = np.array([y], dtype=np.float64)
        z = np.array([z], dtype=np.float64)
    else:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
    return Kalman_v2_both_optimized_lightweight_inner(
        x, y, z, start_id, C, Q, R, P, x_est, Py, y_est, rev, dt, old_time
    )

#used
def dummy_Kalman():
    x_clust = np.array([1.0, 2.0, 3.0], dtype=np.int32)
    y_clust = np.array([1.0, 0.0, 0.0], dtype=np.int32)
    t_clust = np.array([1.0, 0.0, 0.0], dtype=np.int32)
    x_est = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    y_est = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    C = np.array([1.0, 0.0, 0.0], dtype=np.float64)  
    Q = np.diag([1e-6, 1e-4, 1e-2]).astype(np.float64)  
    R = np.float64(100.0)
    P = np.eye(3, dtype=np.float64) 
    
    Kalman_v2_both_optimized(t_clust, x_clust, y_clust, 2, C, Q, R, P, x_est, P, y_est, 0)
    Kalman_v2_both_buffer_optimized(t_clust, x_clust, y_clust, 2, C, Q, R, P, x_est, P, y_est, 0,1)
    print("Numba JIT-Kompilierung abgeschlossen")
    return

#used
def dummy_Kalman_lightweight():
    x_m = np.array([1.0], dtype=np.int32)
    y_m = np.array([1.0], dtype=np.int32)
    t_m = np.array([1.0], dtype=np.int32)
    x_clust = np.array([1.0, 2.0, 3.0], dtype=np.int32)
    y_clust = np.array([1.0, 0.0, 0.0], dtype=np.int32)
    t_clust = np.array([1.0, 0.0, 0.0], dtype=np.int32)
    x_est = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    y_est = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    C = np.array([1.0, 0.0, 0.0], dtype=np.float64) 
    Q = np.diag([1e-6, 1e-4, 1e-2]).astype(np.float64)  
    R = np.float64(100.0)
    P = np.eye(3, dtype=np.float64) 
    
    Kalman_v2_both_optimized_lightweight(t_m, x_m, y_m, 0, C, Q, R, P, x_est, P, y_est, 0,1,0)
    Kalman_v2_both_buffer_optimized_lightweight(t_clust, x_clust, y_clust, 2, C, Q, R, P, x_est, P, y_est, 0,1)
    print("Numba JIT-Kompilierung abgeschlossen")
    return