import netCDF4 as nc
import xarray as xr
#import xesmf as xe
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import LinearSegmentedColormap
mpl.use('Agg')

#####################
# INPUTS
# Work directory
work_dir           = "/work/cmcc/ag15419/basin_modes_new/basin_modes_sa_mod6t/"
# Num of modes to be analyzed
mode_num           = 120
# The code starts to look for modes around the following period [h]
reference_period   = 10
# Order the modes from the smallest or from the greatest ('SM' or 'LM')
eig_order          = 'LM'
# Min val of the modes periods [h]
Tmin               = 2
# Max val of the modes periods [h]
Tmax               = 40
# Amplitude palette limits [-Plot_max,Plot_max] [%]
Plot_max           = 100
# Filter out modes with low amplitude, e.g. 0.10 means that all the modes with amplitude<10% of the total amplitude are rm (to avoid the filtering set amplitude_threshold_ratio = 0)
amplitude_threshold_ratio=0.0001

# NEMO Mesh
mesh_mask_file     = "/work/cmcc/ag15419/VAA_paper/DATA0/mesh_mask.nc"
# NEMO Bathymetry
bathy_meter_file   = "/work/cmcc/ag15419/VAA_paper/DATA0/bathy_meter.nc"

# Outfiles 
outfile_R          = work_dir+'med_modes_'+str(mode_num)+'.nc'

# To compute pure gravitational modes. e.g. with f=0 set flag_RM_coriolis = 1
flag_RM_coriolis = 0

# If you want to compute the mode flag_compute_modes = 1 
flag_compute_modes = 1

# To apply the bdy conditions on eta => flag_eta_bdy=1
flag_eta_bdy = 1

# To plot only the Adriatic Sea area set flag_only_adriatic = 1
flag_only_adriatic = 0

# To use the original GEBCO bathy instead of the MedFS bathy with the 4000m cut (interpolation on the MedFS grid is required)
flag_gebco_bathy   = 0
gebco_bathy        = "/work/cmcc/ag15419/VAA_paper/DATA0/gebco_2024_n46.5_s30.0_w-19.0_e37.0.nc"
gebco_bathy_int    = work_dir+'bathy_gebco_int.nc' 

# To compute and plot the vorticity set vorticity_flag = 1
vorticity_flag = 1

#####################
def prepare_fields(meshmask_path, bathy_path):

    # Open input files
    ds_mask = xr.open_dataset(meshmask_path, decode_times=False)
    ds_bathy = xr.open_dataset(bathy_path, decode_times=False)

    # Land/sea mask
    mask_t = ds_mask['tmask'].isel(t=0, z=0).values.astype(bool)
    mask_u = ds_mask['umask'].isel(t=0, z=0).values.astype(bool)
    mask_v = ds_mask['vmask'].isel(t=0, z=0).values.astype(bool)
    mask_f = (ds_mask['fmask'].isel(t=0, z=0).values > 0)

    mask=mask_t
    lat_nemo = ds_mask['nav_lat'].isel().values
    lon_nemo = ds_mask['nav_lon'].isel().values

    # Calcolo lat/lon F-grid come media dei 4 T-point adiacenti
    lat_f = 0.25 * (lat_nemo[:-1, :-1] + lat_nemo[1:, :-1] + lat_nemo[:-1, 1:] + lat_nemo[1:, 1:])
    lon_f = 0.25 * (lon_nemo[:-1, :-1] + lon_nemo[1:, :-1] + lon_nemo[:-1, 1:] + lon_nemo[1:, 1:])

    # Bathymetry
    if flag_gebco_bathy == 0:
       bathy = ds_bathy['Bathymetry'].isel(time_counter=0).values
       bathy = np.where(mask, bathy, np.nan)

    elif flag_gebco_bathy == 1:
        # 1) Leggo GEBCO
        ds_gebco = xr.open_dataset(gebco_bathy)
        bath_gebco = ds_gebco['elevation'].values
        lat_gebco = ds_gebco['lat'].values
        lon_gebco = ds_gebco['lon'].values
        # Se lat decrescente, inverti
        if lat_gebco[0] > lat_gebco[-1]:
            lat_gebco = lat_gebco[::-1]
            bath_gebco = bath_gebco[::-1, :]
        # 2) Leggo mesh_mask NEMO
        lat_nemo = ds_mask['nav_lat'].isel().values
        lon_nemo = ds_mask['nav_lon'].isel().values
        tmask = ds_mask['tmask'].isel(t=0, z=0).values
        # 3) Interpolatore bilineare
        interp_func = RegularGridInterpolator(
            (lat_gebco, lon_gebco),
            bath_gebco,
            bounds_error=False,
            fill_value=np.nan
        )
        # 4) Interpolazione sulla griglia NEMO
        points = np.array([lat_nemo.flatten(), lon_nemo.flatten()]).T
        bath_interp_flat = interp_func(points)
        bath_new = bath_interp_flat.reshape(lat_nemo.shape)
        # 5) Applico maschera: solo oceano
        bath_new_masked = np.where(tmask == 1, -bath_new, np.nan)
        # 6) Salvo la batimetria interpolata
        ds_out = xr.Dataset(
            {"Bathymetry": (("y", "x"), bath_new_masked)},
            coords={"lon": (("y", "x"), lon_nemo),
                    "lat": (("y", "x"), lat_nemo)}
        )
        ds_out.to_netcdf(gebco_bathy_int)
        # 7) Bathy diventa la nuova 
        bathy = bath_new_masked

    # Lat
    # Coriolis f su griglia F
    Omega = 7.292115e-5  # rad/s

    # Pure gravitational modes case:
    if flag_RM_coriolis == 1:
       Omega = 0.0 

    coriolis = 2 * Omega * np.sin(np.deg2rad(lat_f)) 
    

    # Grid (dx, dy)
    dxt = ds_mask['e1t'].isel(t=0).values  # m
    dyt = ds_mask['e2t'].isel(t=0).values  # m
    dxu = ds_mask['e1u'].isel(t=0).values  # m
    dyu = ds_mask['e2u'].isel(t=0).values  # m
    dxv = ds_mask['e1v'].isel(t=0).values  # m
    dyv = ds_mask['e2v'].isel(t=0).values  # m

    return mask_t, mask_u, mask_v, bathy, coriolis, dxu, dyu, dxv, dyv, dxt, dyt, lon_nemo, lat_nemo 

def plot_input_fields(mask, bathy, coriolis, dx, dy, filename="input_fields.png", dpi=150):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    im0 = axs[0, 0].imshow(mask, cmap='gray')
    axs[0, 0].set_title("Land/Sea Mask (1=Sea, 0=Land)")
    plt.colorbar(im0, ax=axs[0, 0], orientation='horizontal')

    im1 = axs[0, 1].imshow(bathy, cmap='viridis')
    axs[0, 1].set_title("Bathymetry (m)")
    plt.colorbar(im1, ax=axs[0, 1], orientation='horizontal')

    im2 = axs[0, 2].imshow(coriolis, cmap='coolwarm')
    axs[0, 2].set_title("Coriolis parameter (1/s)")
    plt.colorbar(im2, ax=axs[0, 2], orientation='horizontal')

    im3 = axs[1, 0].imshow(dx, cmap='plasma')
    axs[1, 0].set_title("Grid spacing dx (m)")
    plt.colorbar(im3, ax=axs[1, 0], orientation='horizontal')

    im4 = axs[1, 1].imshow(dy, cmap='plasma')
    axs[1, 1].set_title("Grid spacing dy (m)")
    plt.colorbar(im4, ax=axs[1, 1], orientation='horizontal')

    axs[1, 2].axis('off')

    for ax in axs.flat:
        ax.set_xlabel("i")
        ax.set_ylabel("j")
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()

# Rotation case (K X = lambda X,  X = (u,v,eta) )
def build_operator_K(mask_t, mask_u, mask_v, bathy, coriolis, e1u, e2v, e1t, e2t, g=9.81):
    
    #g = 0 # TMP for tests

    # Build the K structure based on t, u and v grids
    # K Matrix structure
    #     K_11 K_12 K_13
    # K = K_21 K_22 K_23
    #     K_31 K_32 K_33

    # Save also deta/dx and deta/dy in case we want to apply the bdy condition also on eta
    # these are based on the t grid
    deta_dx_rows, deta_dx_cols, deta_dx_data = [], [], []
    deta_dy_rows, deta_dy_cols, deta_dy_data = [], [], []

    # Read grids parameters
    ny_t, nx_t = mask_t.shape
    ny_u, nx_u = mask_u.shape
    ny_v, nx_v = mask_v.shape

    mapping_t = {}
    invmap_t = {}
    idx_t = 0

    for j in range(ny_t):
        for i in range(nx_t):
            if mask_t[j, i]:
                mapping_t[(i,j)] = idx_t
                invmap_t[idx_t] = (i,j)
                idx_t += 1
    N_t = idx_t  # numero di punti mare griglia t

    mapping_u = {}
    invmap_u = {}
    idx_u = 0

    for j in range(ny_u):
        for i in range(nx_u):
            if mask_u[j, i]:
                mapping_u[(i,j)] = idx_u
                invmap_u[idx_u] = (i,j)
                idx_u += 1

    N_u = idx_u  # numero di punti mare griglia u

    mapping_v = {}
    invmap_v = {}
    idx_v = 0

    for j in range(ny_v):
        for i in range(nx_v):
            if mask_v[j, i]:
                mapping_v[(i,j)] = idx_v
                invmap_v[idx_v] = (i,j)
                idx_v += 1
    N_v = idx_v  # numero di punti mare griglia v

    # offset globali per u, v, eta
    offset_u = 0
    offset_v = N_u
    offset_eta = N_u + N_v

    # --- costruzione blocchi ---

    # Blocco K_12 = f * v
    rows_k12, cols_k12, data_k12 = [], [], []
    
    for k_u in range(N_u):
        i, j = invmap_u[k_u]
        if not mask_u[j, i]:
            continue
    
        # f interpolata al punto u
        f_u = 0.5 * (coriolis[j, i] + coriolis[j-1, i])
    
        # v vicini (4 punti) attorno a u
        v_indices = [
            (i, j),
            (i+1, j),
            (i, j-1),
            (i+1, j-1)
        ]
    
        # Conta solo i punti validi e distribuisci peso corretto
        valid_v = []
        for vi, vj in v_indices:
            if 0 <= vi < nx_v and 0 <= vj < ny_v and mask_v[vj, vi]:
                k_v = mapping_v[(vi, vj)]
                valid_v.append(k_v)
    
        n = len(valid_v)
        if n > 0:
            for k_v in valid_v:
                rows_k12.append(offset_u + k_u)
                cols_k12.append(offset_v + k_v)
                data_k12.append(f_u / n)   # peso uniforme sui punti validi

    # Blocco K_21 = - f * u
    rows_k21, cols_k21, data_k21 = [], [], []
    
    for k_v in range(N_v):
        i, j = invmap_v[k_v]
        if not mask_v[j, i]:
            continue
    
        # f interpolata al punto v
        #f_v = 0.5 * (coriolis[j, i] + coriolis[j, i-1])
        f_v = coriolis[j, i]
    
        # u vicini (4 punti) attorno a v
        u_indices = [
            (i, j),
            (i, j+1),
            (i-1, j),
            (i-1, j+1)
        ]
    
        valid_u = []
        for ui, uj in u_indices:
            if 0 <= ui < nx_u and 0 <= uj < ny_u and mask_u[uj, ui]:
                k_u = mapping_u[(ui, uj)]
                valid_u.append(k_u)
    
        n = len(valid_u)
        if n > 0:
            for k_u in valid_u:
                rows_k21.append(offset_v + k_v)
                cols_k21.append(offset_u + k_u)
                data_k21.append(-f_v / n)

    # Blocco K_13 = -g * (d eta / dx) con convenzione finale
    rows_k13, cols_k13, data_k13 = [], [], []

    for k_u in range(N_u):
        i, j = invmap_u[k_u]
        if not mask_u[j, i]:
            continue

        # punti eta a destra e al punto stesso
        if i+1 < nx_t and mask_t[j, i] and mask_t[j, i+1]:
            k_eta_right = mapping_t[(i+1, j)]
            k_eta_left = mapping_t[(i, j)]
            coeff_right = - g / e1u[j, i]
            coeff_left  = + g / e1u[j, i]
            rows_k13.append(offset_u + k_u)
            cols_k13.append(offset_eta + k_eta_right)
            data_k13.append(coeff_right)
            rows_k13.append(offset_u + k_u)
            cols_k13.append(offset_eta + k_eta_left)
            data_k13.append(coeff_left)

    # Blocco K_23 = -g * (d eta / dy) con convenzione finale
    rows_k23, cols_k23, data_k23 = [], [], []

    for k_v in range(N_v):
        i, j = invmap_v[k_v]
        if not mask_v[j, i]:
            continue

        # punti t a nord e al punto stesso
        if j+1 < ny_t and mask_t[j, i] and mask_t[j+1, i]:
            k_eta_north = mapping_t[(i, j+1)]
            k_eta_south = mapping_t[(i, j)]
            coeff_north = - g / e2v[j, i]
            coeff_south = + g / e2v[j, i]
            rows_k23.append(offset_v + k_v)
            cols_k23.append(offset_eta + k_eta_north)
            data_k23.append(coeff_north)
            rows_k23.append(offset_v + k_v)
            cols_k23.append(offset_eta + k_eta_south)
            data_k23.append(coeff_south)

    # Blocco K_31 = - d (H*u) / dx
    rows_k31, cols_k31, data_k31 = [], [], []

    for k_eta in range(N_t):
        i, j = invmap_t[k_eta]
        if not mask_t[j, i]:
            continue

        # due punti u vicini a t: u a destra e u a sinistra
        u_indices = [(i, j), (i-1, j)]
        for idx, (ui, uj) in enumerate(u_indices):
            if 0 <= ui < nx_u and 0 <= uj < ny_u and mask_u[uj, ui]:
                k_u = mapping_u[(ui, uj)]

                # H interpolata tra i due punti t adiacenti lungo x
                if idx == 0:  # u a destra
                    H_val = 0.5 * (bathy[j, i] + bathy[j, i+1])
                    coeff = - H_val / e1t[j, i]
                else:        # u a sinistra
                    H_val = 0.5 * (bathy[j, i] + bathy[j, i-1])
                    coeff = + H_val / e1t[j, i]

                rows_k31.append(offset_eta + k_eta)    # riga = t
                cols_k31.append(offset_u + k_u)        # colonna = u
                data_k31.append(coeff)

    # Blocco K_32 = - d (H*v) / dy
    rows_k32, cols_k32, data_k32 = [], [], []

    for k_eta in range(N_t):
        i, j = invmap_t[k_eta]
        if not mask_t[j, i]:
            continue

        # due punti v vicini a t: v a nord e v a sud
        v_indices = [(i, j), (i, j-1)]
        for idx, (vi, vj) in enumerate(v_indices):
            if 0 <= vi < nx_v and 0 <= vj < ny_v and mask_v[vj, vi]:
                k_v = mapping_v[(vi, vj)]

                # H interpolata tra i due punti t adiacenti lungo y
                if idx == 0:  # v a nord
                    H_val = 0.5 * (bathy[j, i] + bathy[j+1, i])
                    coeff = - H_val / e2t[j, i]
                else:        # v a sud
                    H_val = 0.5 * (bathy[j, i] + bathy[j-1, i])
                    coeff = + H_val / e2t[j, i]

                rows_k32.append(offset_eta + k_eta)    # riga = t
                cols_k32.append(offset_v + k_v)        # colonna = v
                data_k32.append(coeff)

    # ---Unisco i blocchi per creare la matrice K completa---
    rows_all = rows_k12 + rows_k21 + rows_k13 + rows_k23 + rows_k31 + rows_k32
    cols_all = cols_k12 + cols_k21 + cols_k13 + cols_k23 + cols_k31 + cols_k32
    data_all = data_k12 + data_k21 + data_k13 + data_k23 + data_k31 + data_k32

    N = N_u + N_v + N_t
    K = sp.csr_matrix((data_all, (rows_all, cols_all)), shape=(N, N))

    return K, mapping_u, mapping_v, mapping_t, invmap_t, invmap_u, invmap_v


# Rot case
def compute_barotropic_modes_K_eta_only(K, mask_t, mask_u, mask_v, e1u, e2v,  e1t, e2t, bathy, k=10, which='LM', reference_period=24, tol_imag=1e-6, g=9.81):

    #g = 0 # TMP for tests

    N_t = mask_t.sum()  # numero di punti mare sulla griglia eta
    N_u = mask_u.sum()  # numero di punti mare sulla griglia u
    N_v = mask_v.sum()  # numero di punti mare sulla griglia v

    # --- eigenvalue target ---
    Tref_sec = reference_period * 3600
    omega_ref = 2 * np.pi / Tref_sec
    sigma = +1j * omega_ref  # lambda = i omega

    # Risolvo K X = lambda X
    eigvals, eigvecs = spla.eigs(K, k=k, sigma=sigma, which=which)
    #print("Eigenvalues:", eigvals)

    # --- FILTRO sugli autovalori ---
    offset_eta = mask_u.sum() + mask_v.sum()  # offset globale per eta
    offset_u = 0
    offset_v = mask_u.sum()
    eta_modes = eigvecs[offset_eta:offset_eta + N_t, :]  # Estraggo eta
    u_modes = eigvecs[offset_u:offset_u + N_u, :]  # Estraggo u
    v_modes = eigvecs[offset_v:offset_v + N_v, :]  # Estraggo v

    omega = -1j * eigvals  # Calcolo omega = -i lambda

    # Tengo solo omega positiva (ottengo coppie di valori +- omega)
    n_before_pos = eta_modes.shape[1]
    mask_pos = omega > 0
    omega = omega[mask_pos]
    eta_modes = eta_modes[:, mask_pos]
    u_modes = u_modes[:, mask_pos]
    v_modes = v_modes[:, mask_pos]
    n_after_pos = eta_modes.shape[1]
    print(f"Filtro omega positiva: rimossi {n_before_pos - n_after_pos} modi su {n_before_pos}")

    # BDY CONDITIONS
    n_before_bdy = eta_modes.shape[1]

    # Cerco i punti di costa a est ed ovest per u
    coast_u_east_west = []
    coast_u_east_west_idx = []
    for j in range(mask_u.shape[0]):
        for i in range(mask_u.shape[1]-1):
          # Se e' un punto terra 
          if not mask_u[j,i]:
             t_west = mask_t[j, i]
             t_east = mask_t[j, i+1]
             # se la costa e' a est
             if t_west:
                coast_u_east_west.append(mapping_t[(i,j)])
                coast_u_east_west_idx.append((j,i))
             # se la costa e' a ovest
             elif t_east:
                coast_u_east_west.append(mapping_t[(i+1,j)]) 
                coast_u_east_west_idx.append((j,i))

    # Cerco i punti di costa a nord e sud per v
    coast_v_north_south = []
    coast_v_north_south_idx = []
    for j in range(mask_v.shape[0]-1):
        for i in range(mask_v.shape[1]):
          # Se v e' un punto terra
          if not mask_v[j,i]:
            t_south = mask_t[j, i]
            t_north = mask_t[j+1, i]
            # se la costa e' a nord
            if t_south:
               coast_v_north_south.append(mapping_t[(i,j)])
               coast_v_north_south_idx.append((j,i))
            # se la costa e' a sud
            if t_north:
               coast_v_north_south.append(mapping_t[(i,j+1)])
               coast_v_north_south_idx.append((j,i))

    # Condizioni  al contorno su ETA lungo la costa
    n_before_eta_bdy = eta_modes.shape[1]
    valid_mask_eta = np.ones(n_before_eta_bdy, dtype=bool)

    # La soglia su residual_eta
    #soglia_eta_bdy_max = 1e-6
    soglia_eta_bdy_u = 2.5e-7
    soglia_eta_bdy_v = 3.5e-7

    # Loop su tutti i modi
    for k_mode in range(n_before_eta_bdy):
     eta_field = eta_modes[:, k_mode]
     u_field = u_modes[:, k_mode]
     v_field = v_modes[:, k_mode]
 
     # 1) Punti di bordo sulla griglia u (est/ovest)
     for j_coast, i_coast in coast_u_east_west_idx:
 
         # Estrazione dei punti eta vicini al punto u
         eta_right = eta_field[mapping_t[(i_coast+1, j_coast)]] if (i_coast+1, j_coast) in mapping_t else 0.0
         eta_left  = eta_field[mapping_t[(i_coast, j_coast)]] if (i_coast, j_coast) in mapping_t else 0.0
 
         # Derivata d eta/dx calcolata al punto u
         deta_dx = (eta_right - eta_left) / e1u[j_coast, i_coast]
 
         # Derivata d eta/dy media sui punti v vicini (central difference)
         deta_dy = 0.0
          
         # contributo colonna i
         eta_top    = eta_field[mapping_t[(i_coast, j_coast+1)]] if (i_coast, j_coast+1) in mapping_t else 0.0
         eta_bottom = eta_field[mapping_t[(i_coast, j_coast-1)]] if (i_coast, j_coast-1) in mapping_t else 0.0
         deta_dy += (eta_top - eta_bottom) / (2 * e2t[j_coast, i_coast])
          
         # contributo colonna i+1
         eta_top    = eta_field[mapping_t[(i_coast+1, j_coast+1)]] if (i_coast+1, j_coast+1) in mapping_t else 0.0
         eta_bottom = eta_field[mapping_t[(i_coast+1, j_coast-1)]] if (i_coast+1, j_coast-1) in mapping_t else 0.0
         deta_dy += (eta_top - eta_bottom) / (2 * e2t[j_coast, i_coast+1])
          
         # divido per 1/2 perche' prendo la media tra est ed ovest
         deta_dy /= 2.0

         # Coriolis su glriglia u
         f_u = 0.5 * (coriolis[j_coast, i_coast] + coriolis[j_coast-1, i_coast])
         
         # Residuo della condizione al bdy su eta da minimizzare
         residuo_u = np.abs(1j * omega[k_mode] * deta_dx + f_u * deta_dy)

         # Impongo la soglia su eta 
         if residuo_u > soglia_eta_bdy_u :
             valid_mask_eta[k_mode] = False
             break  # non serve controllare altri punti di bordo per questo modo
 
     # 2) Punti di bordo sulla griglia v (nord/sud)
     if valid_mask_eta[k_mode]:  # solo se non è già scartato
        for j_coast, i_coast in coast_v_north_south_idx:
 
             # Estrazione dei punti eta vicini al punto v
             eta_north = eta_field[mapping_t[(i_coast, j_coast+1)]] if (i_coast, j_coast+1) in mapping_t else 0.0
             eta_south = eta_field[mapping_t[(i_coast, j_coast)]] if (i_coast, j_coast) in mapping_t else 0.0
 
             # Derivata d eta/dy al punto v
             deta_dy = (eta_north - eta_south) / e2v[j_coast, i_coast]
 
             # Derivata d eta/dx media sui punti u vicini come in K_12 (central difference)
             deta_dx = 0.0
             
             # contributo riga j
             eta_right = eta_field[mapping_t[(i_coast+1, j_coast)]] if (i_coast+1, j_coast) in mapping_t else 0.0
             eta_left  = eta_field[mapping_t[(i_coast-1, j_coast)]] if (i_coast-1, j_coast) in mapping_t else 0.0
             deta_dx += (eta_right - eta_left) / (2 * e1t[j_coast, i_coast])
             
             # contributo riga j+1
             eta_right = eta_field[mapping_t[(i_coast+1, j_coast+1)]] if (i_coast+1, j_coast+1) in mapping_t else 0.0
             eta_left  = eta_field[mapping_t[(i_coast-1, j_coast+1)]] if (i_coast-1, j_coast+1) in mapping_t else 0.0
             deta_dx += (eta_right - eta_left) / (2 * e1t[j_coast+1, i_coast])
             
             # divido per 1/2 perche' prendo la media tra est ed ovest
             deta_dx /= 2.0
             
             # Coriolis su griglia v
             #f_v = 0.5 * (coriolis[j_coast, i_coast] + coriolis[j_coast, i_coast-1])
             f_v = coriolis[j_coast, i_coast]

             # Residuo della condizione al bdy su eta da minimizzare
             residuo_v = np.abs(1j * omega[k_mode] * deta_dy - f_v * deta_dx)

             # Impongo la soglia su eta
             if residuo_v > soglia_eta_bdy_v :
                 valid_mask_eta[k_mode] = False
                 break  # non serve controllare altri punti di bordo per questo modo
 
    # Applica il filtro
    if flag_eta_bdy == 1:
       print ("Applying eta bdy conditions")
       omega     = omega[valid_mask_eta]
       period    = 2 * np.pi / omega / 3600
       eta_modes = eta_modes[:, valid_mask_eta]
       u_modes   = u_modes[:, valid_mask_eta]
       v_modes   = v_modes[:, valid_mask_eta]
 
       n_after_eta_bdy = eta_modes.shape[1]
       print(f"Filtro bdy condition su eta: rimossi {n_before_eta_bdy - n_after_eta_bdy} modi su {n_before_eta_bdy}")

    else:
       print(f"Filtro bdy condition su eta NOT applied")

    # Tengo solo la parte reale
    omega = np.real(omega)

    # Periodi in ore
    period = 2 * np.pi / omega / 3600
    #print("all periods:", period)

    # Per la fase calcolo tan-1(-eta_imm/eta_re)
    theta_modes = np.arctan2(-np.imag(eta_modes), np.real(eta_modes))
    # Per l'ampiezza calcolo il modulo sqrt(Re*2+Im*2)
    eta_modes =  np.sqrt(np.real(eta_modes)**2 + np.imag(eta_modes)**2)
    u_modes =  np.sqrt(np.real(u_modes)**2 + np.imag(u_modes)**2)
    v_modes =  np.sqrt(np.real(v_modes)**2 + np.imag(v_modes)**2)

    return omega, period, eta_modes, theta_modes, u_modes, v_modes

# Controllo vorticita' dei modi
def compute_modes_vorticity(eta_modes, u_modes, v_modes, mask_t, mask_u, mask_v, e1u, e2v, coriolis, bathy, threshold_rot=0.3, threshold_grav=0.1):

    n_modes = eta_modes.shape[1]
    vorticity_rms = np.zeros(n_modes)
    vorticity_max = np.zeros(n_modes)
    vorticity_mean = np.zeros(n_modes)
    Pvorticity_rms = np.zeros(n_modes)
    Pvorticity_max = np.zeros(n_modes)
    Pvorticity_mean = np.zeros(n_modes)

    for k_mode in range(n_modes):

        # Ricostruisco u/v 2D dall 1D
        u_field = np.zeros(mask_u.shape)
        v_field = np.zeros(mask_v.shape)
        u_field[mask_u] = u_modes[:, k_mode]
        v_field[mask_v] = v_modes[:, k_mode]

        # Calcolo vorticità relativa sulla F-grid
        # zeta_ij = dv/dx - du/dy
        dv_dx = (v_field[1:, :] - v_field[:-1, :]) / e1u[:-1, :]
        du_dy = (u_field[:, 1:] - u_field[:, :-1]) / e2v[:, :-1]
        zeta = dv_dx[:, :-1] - du_dy[:-1, :]

        # H su griglia F (centrata tra i punti T)
        H_F = 0.25 * (bathy[:-1, :-1] + bathy[1:, :-1] + bathy[:-1, 1:] + bathy[1:, 1:])

        # PV sulla F-grid
        mask_F = H_F != 0
        PV = np.full_like(H_F, np.nan, dtype=float)
        PV[mask_F] = (zeta[mask_F] + coriolis[mask_F]) / H_F[mask_F]

        # Vorticita' relativa
        rms_zeta = np.sqrt(np.nanmean(zeta**2))
        vorticity_rms[k_mode] = rms_zeta
        vorticity_max[k_mode] = np.nanmax(zeta)
        vorticity_mean[k_mode] = np.nanmean(zeta)
        # Vorticita' Potenziale
        Pvorticity_rms[k_mode] = np.sqrt(np.nanmean(PV**2))
        Pvorticity_max[k_mode] = np.nanmax(PV)
        Pvorticity_mean[k_mode] = np.nanmean(PV)

    return PV, Pvorticity_rms, Pvorticity_max, Pvorticity_mean, vorticity_rms, vorticity_max, vorticity_mean

def reconstruct_modes(modes, invmap, shape):
    Nmodes = modes.shape[1]
    ny, nx = shape
    modes_2D = np.full((Nmodes, ny, nx), np.nan)

    for m in range(Nmodes):
        for idx, (i, j) in invmap.items():
            modes_2D[m, j, i] = modes[idx, m]

    return modes_2D

def save_modes_to_netcdf(filename, modes_2D, periods, mask=None):

    k, ny, nx = modes_2D.shape

    with nc.Dataset(filename, 'w') as ds:
        # Dimensioni
        ds.createDimension('mode', k)
        ds.createDimension('y', ny)
        ds.createDimension('x', nx)

        # Variabili
        modes_var = ds.createVariable('modes', 'f4', ('mode', 'y', 'x'))
        periods_var = ds.createVariable('periods', 'f4', ('mode',))
        
        # Attributi
        modes_var.units = "%"
        periods_var.units = "hours"

        # Scrittura dati
        modes_var[:] = modes_2D
        periods_var[:] = periods

        if mask is not None:
            mask_var = ds.createVariable('mask', 'i1', ('y', 'x'))
            mask_var[:] = mask
            mask_var.description = "1 = sea, 0 = land"

    print(f"Salvato file NetCDF: {filename}")

def load_modes_from_netcdf(filename,meshmask_path):

    # Open mesh_mask file
    ds_mask = xr.open_dataset(meshmask_path, decode_times=False)
    # Land/sea mask
    mask = ds_mask['tmask'].isel(t=0, z=0).values.astype(bool)
    lat_nemo = ds_mask['nav_lat'].isel().values
    lon_nemo = ds_mask['nav_lon'].isel().values

    # Open modes nc file
    with nc.Dataset(filename) as ds:
        modes_2D = ds.variables['modes'][:]
        periods = ds.variables['periods'][:]
        mask = ds.variables['mask'][:] if 'mask' in ds.variables else None
    return modes_2D, periods, mask, lon_nemo, lat_nemo

def plot_mode(mode_2d, mask, lon_nemo, lat_nemo, title="", filename="mode.png", filename_abs="mode_abs.png", cmap="RdBu_r", cmap_abs="gist_stern_r", dpi=150, n_levels=41): 

    # Abs plot
    if flag_only_adriatic == 1:
       plt.figure(figsize=(5, 4))
    else:
       plt.figure(figsize=(10, 4))

    # Stampa l'ampiezza massima, media e RMS prima della normalizzazione
    #print(f" Max Amp:  {np.nanmax(np.abs(mode_2d)):.3e}")
    #print(f" Mean Amp: {np.nanmean(np.abs(mode_2d)):.3e}")
    #print(f"RMS Amp: {np.sqrt(np.nanmean(np.abs(mode_2d)**2)):.3e}")


    # Normalizza a 100 sul valore massimo o sul 99th percentile
    #norm_mode = np.abs(mode_2d) / np.nanmax(np.abs(mode_2d)) * 100
    p99 = np.nanpercentile(np.abs(mode_2d), 99)
    norm_mode = np.abs(mode_2d) / p99 * 100

    # Applica la maschera
    masked = np.ma.masked_where(~mask.astype(bool), norm_mode)

    # Costruisce palette discreta centrata in zero
    levels = np.linspace(0, Plot_max, n_levels)
    norm = mpl.colors.BoundaryNorm(levels, ncolors=256)

    # Plot
    cmap_abs = mpl.cm.get_cmap(cmap_abs)
    cmap_abs = truncate_colormap(cmap_abs, 0.05, 0.95)
    cmap_abs.set_bad("white")

    # Usa pcolormesh con lon/lat
    im = plt.pcolormesh(lon_nemo, lat_nemo, masked, cmap=cmap_abs, norm=norm, shading='auto')
    cbar = plt.colorbar(im) #, orientation='vertical')
    cbar.set_label("Mode Amplitude (%)")

    contour_levels = np.arange(0, Plot_max, 10)
    CS = plt.contour(lon_nemo, lat_nemo, masked, levels=contour_levels, colors='k',
                     linestyles='dashed', linewidths=0.5)
    if flag_only_adriatic == 1:
       plt.clabel(CS, inline=True, fontsize=5, fmt='%d%%', inline_spacing=0.2, manual=False)
    else:
       plt.clabel(CS, inline=True, fontsize=7, fmt='%d%%')

    # Linea di costa (contorno della maschera)
    plt.contour(lon_nemo, lat_nemo, masked.mask, levels=[0.5], colors='black', linewidths=0.8)
    plt.contourf(lon_nemo, lat_nemo, masked.mask, levels=[0.5, 1.5], colors='gray')

    plt.contourf(lon_nemo, lat_nemo, masked, 
                 levels=[-amplitude_threshold_ratio, amplitude_threshold_ratio], colors='white')

    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    if flag_only_adriatic == 1:
         plt.xlim(12,20)
         plt.ylim(39,46)
    else:
         plt.xlim(-6.000, lon_nemo.max())
         plt.ylim(lat_nemo.min(), lat_nemo.max())

    plt.tight_layout()
    plt.savefig(filename_abs, dpi=dpi)
    plt.close()

# Plot di mean e max amplitude per ogni modo
def plot_all_modes_amplitude(modes_2D, period, work_dir="", flag_only_adriatic=0, dpi=150,Pvorticity_rms=None, Pvorticity_max=None, Pvorticity_mean=None, vorticity_rms=None, vorticity_max=None, vorticity_mean=None):
    
    n_modes  = len(modes_2D)
    max_amp  = np.zeros(n_modes)
    mean_amp = np.zeros(n_modes)
    rms_amp  = np.zeros(n_modes)
    
    # Calcola max e mean per ogni modo
    for i in range(n_modes):
        max_amp[i]  = np.nanmax(np.abs(modes_2D[i]))
        mean_amp[i] = np.nanmean(np.abs(modes_2D[i]))
        rms_amp[i]  = np.sqrt(np.nanmean(np.abs(modes_2D[i])**2))
    
    # Plot max amplitude
    plt.figure(figsize=(12,4))
    plt.plot(range(n_modes), max_amp, 'o-', label="Max amplitude", color='tab:orange')
    plt.xticks(range(n_modes), [f"{p:.2f}h" for p in period], rotation=45)
    plt.xlabel("Mode (Period)")
    plt.ylabel("Max amplitude (m)")
    plt.title("Maximum amplitude per mode")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{work_dir}/all_modes_max_amplitude.png", dpi=dpi)
    plt.close()
    
    # Plot mean amplitude
    plt.figure(figsize=(12,4))
    plt.plot(range(n_modes), mean_amp, 'o-', label="Mean amplitude", color='tab:blue')
    plt.xticks(range(n_modes), [f"{p:.2f}h" for p in period], rotation=45)
    plt.xlabel("Mode (Period)")
    plt.ylabel("Mean amplitude (m)")
    plt.title("Mean amplitude per mode")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{work_dir}/all_modes_mean_amplitude.png", dpi=dpi)
    plt.close()

    # Plot rms amplitude
    plt.figure(figsize=(12,4))
    plt.plot(range(n_modes), rms_amp, 'o-', label="RMS amplitude", color='tab:red')
    plt.xticks(range(n_modes), [f"{p:.2f}h" for p in period], rotation=45)
    plt.xlabel("Mode (Period)")
    plt.ylabel("Amplitude RMS (m)")
    plt.ylim(0.0,0.003)
    plt.title("Amplitude RMS per mode")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{work_dir}/all_modes_rms_amplitude.png", dpi=dpi)
    plt.close()
    
    # Plot RMS, Mean and Max vorticity (se ho calcolato la vorticity)
    if vorticity_rms is not None:
        plt.figure(figsize=(12,4))
        plt.plot(range(n_modes), vorticity_rms, 'o-', label="RMS vorticity", color='tab:red')
        plt.xticks(range(n_modes), [f"{p:.2f}h" for p in period], rotation=45)
        plt.xlabel("Mode (Period)")
        plt.ylabel("RMS vorticity (1/s)")
        plt.title("RMS Vorticity per mode")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{work_dir}/all_modes_rms_vorticity.png", dpi=dpi)
        plt.close()

    if vorticity_max is not None:
        plt.figure(figsize=(12,4))
        plt.plot(range(n_modes), vorticity_max, 'o-', label="Max vorticity", color='tab:orange')
        plt.xticks(range(n_modes), [f"{p:.2f}h" for p in period], rotation=45)
        plt.xlabel("Mode (Period)")
        plt.ylabel("Max vorticity (1/s)")
        plt.title("Max Vorticity per mode")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{work_dir}/all_modes_max_vorticity.png", dpi=dpi)
        plt.close()

    if vorticity_mean is not None:
        plt.figure(figsize=(12,4))
        plt.plot(range(n_modes), vorticity_max, 'o-', label="Mean vorticity")
        plt.xticks(range(n_modes), [f"{p:.2f}h" for p in period], rotation=45)
        plt.xlabel("Mode (Period)")
        plt.ylabel("Mean vorticity (1/s)")
        plt.title("Mean Vorticity per mode")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{work_dir}/all_modes_mean_vorticity.png", dpi=dpi)
        plt.close()

    if Pvorticity_mean is not None:
        plt.figure(figsize=(12,4))
        plt.plot(range(n_modes), Pvorticity_max, 'o-', label="Mean vorticity", color='tab:blue')
        plt.xticks(range(n_modes), [f"{p:.2f}h" for p in period], rotation=45)
        plt.xlabel("Mode (Period)")
        plt.ylabel("Mean vorticity (1/s)")
        plt.title("Mean Vorticity per mode")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{work_dir}/all_modes_mean_Pvorticity.png", dpi=dpi)
        plt.close()    

    if Pvorticity_rms is not None:
        plt.figure(figsize=(12,4))
        plt.plot(range(n_modes), Pvorticity_rms, 'o-', label="RMS vorticity", color='tab:red')
        plt.xticks(range(n_modes), [f"{p:.2f}h" for p in period], rotation=45)
        plt.xlabel("Mode (Period)")
        plt.ylabel("RMS vorticity (1/s)")
        plt.title("RMS Vorticity per mode")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{work_dir}/all_modes_rms_Pvorticity.png", dpi=dpi)
        plt.close()

    if Pvorticity_max is not None:
        plt.figure(figsize=(12,4))
        plt.plot(range(n_modes), Pvorticity_max, 'o-', label="Max vorticity", color='tab:orange')
        plt.xticks(range(n_modes), [f"{p:.2f}h" for p in period], rotation=45)
        plt.xlabel("Mode (Period)")
        plt.ylabel("Max vorticity (1/s)")
        plt.title("Max Vorticity per mode")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{work_dir}/all_modes_max_Pvorticity.png", dpi=dpi)
        plt.close()

# Truncate the colormap to exclude the lightest part (e.g. bottom 20%)
def truncate_colormap(cmap, minval=0.2, maxval=1.0, n=256):
    new_cmap = LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

################### MAIN #############
# Prepare input fields
print ('Preparing input fields..')
mask_t, mask_u, mask_v, bathy, coriolis, dxu, dyu, dxv, dyv, dxt, dyt, lon_nemo, lat_nemo = prepare_fields(mesh_mask_file, bathy_meter_file)
print ('Done!')

# Compute and plot the Med modes

# Mask the Antlantic
print ('Mask the Atlantic Box')
def mask_atlantic(mask):
    ny, nx = mask.shape
    J, I = np.indices((ny, nx))
    atlantic_mask = (I < 300) | ((I < 415) & (J > 250))
    mask[atlantic_mask] = 0
    return mask

mask_t = mask_atlantic(mask_t)
mask_u = mask_atlantic(mask_u)
mask_v = mask_atlantic(mask_v)
bathy  = mask_atlantic(bathy)

# Plot input fields
print ('Plotting input fields..')
plot_input_fields(np.squeeze(mask_t), np.squeeze(bathy), np.squeeze(coriolis), np.squeeze(dxt), np.squeeze(dyt), filename=work_dir+"/input_fields"+str(mode_num)+".png")
print ('Done!')

########### Real modes ##############3

if flag_compute_modes != 0 :

   print ('Compute K operator')
   K, mapping_u, mapping_v, mapping_t, invmap_t, invmap_u, invmap_v  = build_operator_K(mask_t, mask_u, mask_v, bathy, coriolis, dxu, dyv, dxt, dyt)
   print ('Done!')

   print ('Compute the modes')
   omega, period, eta_modes, theta_modes, u_modes, v_modes = compute_barotropic_modes_K_eta_only(K, mask_t, mask_u, mask_v, dxu, dyv, dxt, dyt, bathy, k=mode_num, which=eig_order,reference_period=reference_period)
   modes=eta_modes
   print ('Done!')

   print ('Select modes with periods in the following range [h]:',Tmin,Tmax)
   valid = np.where((period >= Tmin) & (period <= Tmax))[0]
   omega, period = omega[valid], period[valid]
   modes = modes[:, valid]
   k = len(omega)
   print (k,' modes selected')
   print ('Done!')

   print ('Build 2d modes')
   shape = mask_t.shape
   modes_2D = reconstruct_modes(modes, invmap_t, shape)
   print ('Done!')

   print("Compute vorticity for each mode")
   if vorticity_flag == 1 :
      vorticity, Pvorticity_rms, Pvorticity_max, Pvorticity_mean, vorticity_rms, vorticity_max, vorticity_mean  = compute_modes_vorticity( eta_modes, u_modes, v_modes, mask_t, mask_u, mask_v, dxu, dyv, coriolis, bathy)

   print ('Save the R modes')
   save_modes_to_netcdf(outfile_R, modes_2D, period, mask=mask_t)
   print ('Done!')

else:
   print ('Load the R modes')
   modes_2D, period, mask_t, lon_nemo, lat_nemo = load_modes_from_netcdf(outfile_R,mesh_mask_file)
   print ('Done!')

print ('Plot the R modes amplitude')

# Ordina gli indici dei modi per periodo decrescente
sorted_indices = np.argsort(-period)
renum = 0  
for m in sorted_indices:
    max_amp = np.nanmax(modes_2D[m])
    this_period = period[m]
    print ('Mode:',renum,f' Period: {this_period:.2f} h') 
            
    title = f"Mode with Period: {this_period:.2f} h"
    if flag_only_adriatic == 1:
       filename = f"{work_dir}/mode_{renum:02d}_{mode_num}_{this_period:.2f}h_AdrSea.png"
       filename_abs = f"{work_dir}/mode_abs_{renum:02d}_{mode_num}_{this_period:.2f}h_AdrSea.png"
    else:
       filename = f"{work_dir}/mode_{renum:02d}_{mode_num}_{this_period:.2f}h.png"
       filename_abs = f"{work_dir}/mode_abs_{renum:02d}_{mode_num}_{this_period:.2f}h.png"
            
    plot_mode(modes_2D[m], mask_t, lon_nemo, lat_nemo, title=title, filename=filename, filename_abs=filename_abs)
    renum += 1


# Creo un plot con ampiezze massime, medie e distribuzione dell'ampiezza per ogni modo
modes_sorted = [modes_2D[m] for m in sorted_indices]
period_sorted = [period[m] for m in sorted_indices]
if vorticity_flag == 1 :
   vorticity_rms_sorted = [vorticity_rms[m] for m in sorted_indices]
   vorticity_max_sorted = [vorticity_max[m] for m in sorted_indices]
   vorticity_mean_sorted = [vorticity_mean[m] for m in sorted_indices]
   Pvorticity_rms_sorted = [Pvorticity_rms[m] for m in sorted_indices]
   Pvorticity_max_sorted = [Pvorticity_max[m] for m in sorted_indices]
   Pvorticity_mean_sorted = [Pvorticity_mean[m] for m in sorted_indices]
else: 
   vorticity_rms_sorted = None
   vorticity_max_sorted = None
   vorticity_mean_sorted = None
   Pvorticity_rms_sorted = None
   Pvorticity_max_sorted = None
   Pvorticity_mean_sorted = None

plot_all_modes_amplitude(modes_sorted, period_sorted, work_dir=work_dir, Pvorticity_rms=Pvorticity_rms_sorted, Pvorticity_max=Pvorticity_max_sorted, Pvorticity_mean=Pvorticity_mean_sorted, vorticity_rms=vorticity_rms_sorted, vorticity_max=vorticity_max_sorted, vorticity_mean=vorticity_mean_sorted)
