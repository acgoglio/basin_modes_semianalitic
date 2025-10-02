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
work_dir           = "/work/cmcc/ag15419/basin_modes/basin_modes_sa_fg_new_4/"
# Num of modes to be analyzed
mode_num           = 10
# The code starts to look for modes around the following period [h]
reference_period   = 30
# Order the modes from the smallest or from the greatest ('SM' or 'LM')
eig_order          = 'LM'
# Min val of the modes periods [h]
Tmin               = 2
# Max val of the modes periods [h]
Tmax               = 40
# Consider only the modes whose amplitude is higher than Perc_Amp % of the highest amplitude value in a num of grid points higher than Counts_min 
#Perc_Amp           = 0.001
#Counts_min         = 1
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

# If you want to compute the mode flag_compute_modes = 1 
flag_compute_modes = 1

# To plot only the Adriatic Sea area set flag_only_adriatic = 1
flag_only_adriatic = 0

# To set f term (1=rot+grav modes, 0=only gravitational contribution, 2=f cost+grav modes)
flag_f             = 1

# To use the original GEBCO bathy instead of the MedFS bathy with the 4000m cut (interpolation on the MedFS grid is required)
flag_gebco_bathy   = 0
gebco_bathy        = "/work/cmcc/ag15419/VAA_paper/DATA0/gebco_2024_n46.5_s30.0_w-19.0_e37.0.nc"
gebco_bathy_int    = work_dir+'bathy_gebco_int.nc' 

#####################
def prepare_fields(meshmask_path, bathy_path):

    # Open input files
    ds_mask = xr.open_dataset(meshmask_path, decode_times=False)
    ds_bathy = xr.open_dataset(bathy_path, decode_times=False)

    # Land/sea mask
    mask_t = ds_mask['tmask'].isel(t=0, z=0).values.astype(bool)
    mask_u = ds_mask['umask'].isel(t=0, z=0).values.astype(bool)
    mask_v = ds_mask['vmask'].isel(t=0, z=0).values.astype(bool)

    mask=mask_t
    lat_nemo = ds_mask['nav_lat'].isel().values
    lon_nemo = ds_mask['nav_lon'].isel().values

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
    lat = ds_mask['nav_lat'].values
    # Coriolis f
    omega = 7.292115e-5  # rad/s
    omega = 0.0 # TMP for debug!
    if flag_f != 2:
       coriolis = 2 * omega * np.sin(np.deg2rad(lat))
       print ('Coriolis:',coriolis)
    elif flag_f == 2:
       lat_fix=lat*0+37.75
       coriolis = 2 * omega * np.sin(np.deg2rad(lat_fix))
       print ('Constant Coriolis:',coriolis)

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

# Case without rotation
def build_operator_A(mask, bathy, coriolis, e1u, e2v, e1t, e2t, g=9.81):
    ny, nx = mask.shape
    mapping = {}
    invmap = {}
    idx = 0

    for j in range(ny):
        for i in range(nx):
            if mask[j, i]:
                mapping[(i, j)] = idx
                invmap[idx] = (i, j)
                idx += 1

    N = idx
    rows, cols, data = [], [], []

    for k in range(N):
        i, j = invmap[k]
        H = bathy[j, i]
        f = coriolis[j, i]
        diag = 0

        # Direzione x (U-points)
        for di in [-1, 1]:
            ni = i + di
            # Se e' dentro il dominio e se e' punto mare
            if 0 <= ni < nx and mask[j, ni]:
                # Calcolo bathy media tra i due punti consecutivi
                Hij = 0.5 * (bathy[j, i] + bathy[j, ni])
                # calcolo la lunghezza della cella in direzione zonale
                e1u_ij = e1u[j, i] if di > 0 else e1u[j, ni]
                # Calcolo il coeff
                coeff = g * Hij / (e1u_ij * e1t[j, i])
                n_idx = mapping[(ni, j)]

                # La righa dell'operatore corrisponde al punto corrente (k)
                rows.append(k)
                # Il valore nella colonna dell'operatore corrisponde invece al punto con cui interagisce il punto corrente 
                cols.append(n_idx)
                # Scrivo i Valori fuori diagonale
                data.append(coeff)    # Scrivo il termine non rot fuori diag
                # Aggiorno i valori della diagonale (sottraendo dal termine rotazionale)
                #print ('diag prima x',diag)
                diag = diag - coeff         # termine non rot sulla diagonale
                #print ('diag dopo x',diag)

        # Direzione y (V-points) - stessa cosa
        for dj in [-1, 1]:
            nj = j + dj
            if 0 <= nj < ny and mask[nj, i]:
                Hij = 0.5 * (bathy[j, i] + bathy[nj, i])
                e2v_ij = e2v[j, i] if dj > 0 else e2v[nj, i]
                coeff = g * Hij / (e2v_ij * e2t[j, i])
                n_idx = mapping[(i, nj)]

                rows.append(k)
                cols.append(n_idx)
                data.append(coeff)    # Scrivo il termine non rot fuori diag
                # Aggiorno i valori della diagonale (sottraendo dal termine rotazionale)
                #print ('diag prima y',diag)
                diag = diag - coeff        # termine non rot sulla diagonale
                #print ('diag dopo y',diag)

        # Scrivo i valori calcolati per la diagonale
        rows.append(k)
        cols.append(k)
        data.append(diag)

    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    #print("A symmetric?", (A - A.T).nnz == 0)
    return A, mapping, invmap

# Rot case (K X = lambda X,  X = (u,v,eta) )
def build_operator_K(mask_t, mask_u, mask_v, bathy, coriolis, e1u, e2v, e1t, e2t, g=9.81):

    # Build the K structure based on t, u and v grids

    # K Matrix structure
    #     K_11 K_12 K_13
    # K = K_21 K_22 K_23
    #     K_31 K_32 K_33
    import scipy.sparse as sp

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

        # f interpolata al punto u (media lungo x)
        f_u = 0.5 * (coriolis[j, i] + coriolis[j, i+1])

        # v vicini (4 punti) a nord/sud e sx/dx del punto u
        v_indices = [
            (i, j),       # nord-ovest
            (i+1, j),     # nord-est
            (i, j-1),     # sud-ovest
            (i+1, j-1)    # sud-est
        ]

        for vi, vj in v_indices:
            if 0 <= vi < nx_v and 0 <= vj < ny_v and mask_v[vj, vi]:
                k_v = mapping_v[(vi, vj)]
                rows_k12.append(offset_u + k_u)      # riga = u
                cols_k12.append(offset_v + k_v)      # colonna = v
                data_k12.append(f_u * 0.25)

    # Blocco K_21 = - f * u
    rows_k21, cols_k21, data_k21 = [], [], []

    for k_v in range(N_v):
        i, j = invmap_v[k_v]
        if not mask_v[j, i]:
            continue

        # f interpolata al punto v (media lungo y)
        f_v = 0.5 * (coriolis[j, i] + coriolis[j+1, i])

        # u vicini (4 punti intorno a v)
        u_indices = [
            (i, j),       # sud-ovest
            (i, j+1),     # nord-ovest
            (i-1, j),     # sud-est
            (i-1, j+1)    # nord-est
        ]

        for ui, uj in u_indices:
            if 0 <= ui < nx_u and 0 <= uj < ny_u and mask_u[uj, ui]:
                k_u = mapping_u[(ui, uj)]
                rows_k21.append(offset_v + k_v)      # riga = v
                cols_k21.append(offset_u + k_u)      # colonna = u
                data_k21.append(-0.25 * f_v)

    # Blocco K_13 = -g * (d eta / dx) con convenzione finale
    rows_k13, cols_k13, data_k13 = [], [], []

    for k_u in range(N_u):
        i, j = invmap_u[k_u]
        if not mask_u[j, i]:
            continue

        # punti η a destra e al punto stesso
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

        # punti η a nord e al punto stesso
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

        # due punti u vicini a η: u a destra e u a sinistra
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

                rows_k31.append(offset_eta + k_eta)    # riga = η
                cols_k31.append(offset_u + k_u)        # colonna = u
                data_k31.append(coeff)

    # Blocco K_32 = - d (H*v) / dy
    rows_k32, cols_k32, data_k32 = [], [], []

    for k_eta in range(N_t):
        i, j = invmap_t[k_eta]
        if not mask_t[j, i]:
            continue

        # due punti v vicini a η: v a nord e v a sud
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

                rows_k32.append(offset_eta + k_eta)    # riga = η
                cols_k32.append(offset_v + k_v)        # colonna = v
                data_k32.append(coeff)

    # ---Unisco i blocchi per creare la matrice K completa---
    rows_all = rows_k12 + rows_k21 + rows_k13 + rows_k23 + rows_k31 + rows_k32
    cols_all = cols_k12 + cols_k21 + cols_k13 + cols_k23 + cols_k31 + cols_k32
    data_all = data_k12 + data_k21 + data_k13 + data_k23 + data_k31 + data_k32

    N = N_u + N_v + N_t
    K = sp.csr_matrix((data_all, (rows_all, cols_all)), shape=(N, N))

    return K, mapping_u, mapping_v, mapping_t, invmap_t, invmap_u, invmap_v


# Case without rotation 
def compute_barotropic_modes(A, k=10, which='LM', reference_period=24):
    # Compute sigma (target eigenvalue)
    Tref_sec = reference_period * 3600
    omega_ref = 2 * np.pi / Tref_sec
    sigma = - omega_ref**2  

    # Solve perche' l'eq. e' A eta = lambda eta
    eigvals, eigvecs = eigsh(A, k=k, sigma=sigma, which=which, mode='normal')
    #eigvals, eigvecs =eigsh(A, k=k, which=which)
    #print("Eigenvalues:", eigvals)

    # Consider only physical eigenvalues
    valid = eigvals < 0
    eigvals = eigvals[valid]
    eigvecs = eigvecs[:, valid]
    #print("Eigenvalues with physical relevance:", eigvals)

    # Frequenze omega = sqrt(lambda)
    omega = np.sqrt(-eigvals)
    #omega = np.sqrt(np.abs(eigvals))

    # Periodi in ore
    period = 2 * np.pi / omega / 3600
    print('Selected periods:', period)

    return omega, period, eigvecs


# Rot case
def compute_barotropic_modes_K_eta_only(K, mask_t, mask_u, mask_v, k=10, which='LM', reference_period=24, tol_imag=1e-6):

    N_t = mask_t.sum()  # numero di punti mare sulla griglia eta

    # --- eigenvalue target ---
    Tref_sec = reference_period * 3600
    omega_ref = 2 * np.pi / Tref_sec
    sigma = +1j * omega_ref  # lambda = i omega

    # Risolvo K X = lambda X
    eigvals, eigvecs = spla.eigs(K, k=k, sigma=sigma, which=which)
    print("Eigenvalues:", eigvals)

    # --- FILTRO sugli autovalori ---
    offset_eta = mask_u.sum() + mask_v.sum()  # offset globale per eta
    eta_modes = eigvecs[offset_eta:offset_eta + N_t, :]  # Estraggo SOLO eta
    #offset_u = 0
    #N_u = mask_u.sum()
    #eta_modes = eigvecs[offset_u:offset_u + N_u, :] # Estraggo SOLO u
    #offset_v = mask_u.sum()
    #N_v = mask_v.sum()
    #eta_modes = eigvecs[offset_v:offset_v + N_v, :] # Estraggo SOLO v

    omega = -1j * eigvals  # Calcolo omega = -i lambda

    # Tengo solo valori con parte immaginaria piccola rispetto alla parte reale
    #n_before_im = eta_modes.shape[1]
    #valid = np.abs(np.imag(omega)) < tol_imag * np.abs(np.real(omega))
    #omega = np.real(omega[valid])
    #eta_modes = eta_modes[:, valid]
    #n_after_im = eta_modes.shape[1]
    #print(f"Filtro parte immaginaria: rimossi {n_before_im - n_after_im} modi su {n_before_im}")

    # Tengo solo omega positiva
    #n_before_pos = eta_modes.shape[1]
    #mask_pos = omega > 0
    #omega = omega[mask_pos]
    #eta_modes = eta_modes[:, mask_pos]
    #n_after_pos = eta_modes.shape[1]
    #print(f"Filtro omega positiva: rimossi {n_before_pos - n_after_pos} modi su {n_before_pos}")

    # Periodi in ore
    period = 2 * np.pi / omega / 3600
    print("all periods:", period)

    # --- FILTRO sugli autovettori ---
    # Anche per l'ampiezza tengo solo le parti reali
    eta_modes = np.real(eta_modes)

    return omega, period, eta_modes

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
    plt.figure(figsize=(10, 4))

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
    cbar = plt.colorbar(im, orientation='vertical')
    cbar.set_label("Mode Amplitude (%)")

    contour_levels = np.arange(0, Plot_max, 10)
    CS = plt.contour(lon_nemo, lat_nemo, masked, levels=contour_levels, colors='k',
                     linestyles='dashed', linewidths=0.5)
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
         plt.xlim(lon_nemo.min(), lon_nemo.max())
         plt.ylim(lat_nemo.min(), lat_nemo.max())
    else:
         plt.xlim(-6.000, lon_nemo.max())
         plt.ylim(lat_nemo.min(), lat_nemo.max())

    plt.tight_layout()
    plt.savefig(filename_abs, dpi=dpi)
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

# Plot input fields
print ('Plotting input fields..')
plot_input_fields(np.squeeze(mask_t), np.squeeze(bathy), np.squeeze(coriolis), np.squeeze(dxt), np.squeeze(dyt), filename=work_dir+"/input_fields"+str(mode_num)+".png")
print ('Done!')

########### Real modes ##############3

if flag_compute_modes != 0 :

   print ('Compute A or K operator')
   if flag_f == 0:
      A, mapping, invmap_t = build_operator_A(mask_t, bathy, coriolis, dxu, dyv, dxt, dyt)
   else:
      K, mapping_u, mapping_v, mapping_t, invmap_t, invmap_u, invmap_v  = build_operator_K(mask_t, mask_u, mask_v, bathy, coriolis, dxu, dyv, dxt, dyt)
   print ('Done!')

   print ('Compute the modes')
   if flag_f == 0:
      omega, period, modes = compute_barotropic_modes(A, k=mode_num, which=eig_order,reference_period=reference_period)
   else:
      omega, period, eta_modes = compute_barotropic_modes_K_eta_only(K, mask_t, mask_u, mask_v, k=mode_num, which=eig_order,reference_period=reference_period)
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

   print ('Save the R modes')
   save_modes_to_netcdf(outfile_R, modes_2D, period, mask=mask_t)
   print ('Done!')

else:
   print ('Load the R modes')
   modes_2D, period, mask_t, lon_nemo, lat_nemo = load_modes_from_netcdf(outfile_R,mesh_mask_file)
   print ('Done!')

print ('Plot the R modes amplitude')
print ('Print and plot only the modes with a relevant amplitude')

# Ordina gli indici dei modi per periodo decrescente
sorted_indices = np.argsort(-period)
renum = 0  
for m in sorted_indices:
    max_amp = np.nanmax(modes_2D[m])
    #if max_amp > Th_Amp:
    #    counts = np.sum(modes_2D[m] > Th_Amp)
    #    if counts > Counts_min:
    this_period = period[m]
    print ('Mode:',renum,f' Period: {this_period:.2f} h') 
            
    #title = f"Barotropic mode {renum} - Period {this_period:.2f} h"
    title = f"Mode with Period: {this_period:.2f} h"
    filename = f"{work_dir}/mode_{renum:02d}_{mode_num}_{this_period:.2f}h.png"
    filename_abs = f"{work_dir}/mode_abs_{renum:02d}_{mode_num}_{this_period:.2f}h.png"
            
    plot_mode(modes_2D[m], mask_t, lon_nemo, lat_nemo, title=title, filename=filename, filename_abs=filename_abs)
    renum += 1

