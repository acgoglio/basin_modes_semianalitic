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
work_dir           = "/work/cmcc/ag15419/basin_modes_new/basin_modes_sa_sign/"
# Num of modes to be analyzed (e.g. 120)
mode_num           = 120
# The code starts to look for modes around the following period [h], e.g. 10
reference_period   = 10
# Order the modes from the smallest or from the greatest ('SM' or 'LM'), e.g. LM
eig_order          = 'LM'
# Min val of the modes periods [h], e.g. 3
Tmin               = 3
# Max val of the modes periods [h], e.g. 40
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
soglia_eta_bdy_u = 2.5e-7
soglia_eta_bdy_v = 3.5e-7

# To plot only the Adriatic Sea area set flag_only_adriatic = 1
flag_only_adriatic = 0

# To use the original GEBCO bathy instead of the MedFS bathy with the 4000m cut (interpolation on the MedFS grid is required)
flag_gebco_bathy   = 0
gebco_bathy        = "/work/cmcc/ag15419/VAA_paper/DATA0/gebco_2024_n46.5_s30.0_w-19.0_e37.0.nc"
gebco_bathy_int    = work_dir+'bathy_gebco_int.nc'

# To compute and plot the vorticity set vorticity_flag = 1
vorticity_flag = 0

# To compute and plot the PE/KE ratio set flag_PE_KE_ratio = 1
flag_PE_KE_ratio = 1

# Classificazione oscillatori / rotazionali
# Periodi [h] dei modi che hanno un corrispettivo tra le soluzioni con f = 0
# Usati SOLO per colorare il plot RMS.
flag_color_by_f0 = 1
oscillatory_periods = [26.65, 21.22, 13.32, 11.49, 10.73, 9.37, 8.41, 7.08,
                       6.71, 5.96, 5.70, 5.64, 5.26, 5.11, 4.98, 4.91, 4.25,
                       4.20, 4.09, 3.86, 3.85, 3.82, 3.72, 3.70, 3.67, 3.66,
                       3.63, 3.52, 3.47, 3.44, 3.41, 3.35, 3.32, 3.29]
oscillatory_tol = 0.05   # tolleranza [h] per il match


# --- Palette color-blind safe (Okabe & Ito) ---
COL_OSC = 'tab:red'   #  modi oscillatori
COL_ROT = 'tab:blue'   #  modi rotazionali

# Tick a valori tondi sull'asse dei periodi
RMS_XTICK_STEP = 5 

# Dimensioni dei font
FS_LABEL  = 15
FS_TICK   = 13
FS_LEGEND = 13

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

    # Build the K structure based on t, u and v grids
    # K Matrix structure
    #     K_11 K_12 K_13
    # K = K_21 K_22 K_23
    #     K_31 K_32 K_33

    # CORREZIONE: rimosse le liste deta_dx_rows/cols/data e deta_dy_rows/cols/data,
    # che erano inizializzate ma mai popolate ne' usate. Le condizioni al contorno
    # su eta (Eq. 11-12) non entrano in K: la no-normal-flow e' imposta
    # strutturalmente dalla maschera sui punti u/v di costa.

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

        # f al punto v.
        # NOTA: f dipende solo dalla latitudine, quindi la media lungo i sarebbe
        # un no-op. L'asimmetria rispetto a f_u e' quindi irrilevante.
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
def compute_barotropic_modes_K_eta_only(K, mask_t, mask_u, mask_v, e1u, e2v, e1t, e2t,
                                        bathy, coriolis, mapping_t,
                                        k=10, which='LM', reference_period=24,
                                        tol_imag=1e-6, g=9.81):

    N_t = mask_t.sum()  # numero di punti mare sulla griglia eta
    N_u = mask_u.sum()  # numero di punti mare sulla griglia u
    N_v = mask_v.sum()  # numero di punti mare sulla griglia v

    # --- eigenvalue target ---
    Tref_sec = reference_period * 3600
    omega_ref = 2 * np.pi / Tref_sec
    sigma = +1j * omega_ref  # lambda = i omega

    # Risolvo K X = lambda X
    # NOTA: ARPACK restituisce autovettori a norma 2 unitaria sul vettore
    # completo X = (u, v, eta). Gli autovettori sono quindi definiti a meno di
    # una costante moltiplicativa e le ampiezze non hanno scala assoluta.
    eigvals, eigvecs = spla.eigs(K, k=k, sigma=sigma, which=which)

    # Verifica della normalizzazione
    print(f"Norma del primo autovettore: {np.linalg.norm(eigvecs[:, 0]):.6f}")

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
    mask_pos = np.real(omega) > 0
    omega = omega[mask_pos]
    eta_modes = eta_modes[:, mask_pos]
    u_modes = u_modes[:, mask_pos]
    v_modes = v_modes[:, mask_pos]
    n_after_pos = eta_modes.shape[1]
    print(f"Filtro omega positiva: rimossi {n_before_pos - n_after_pos} modi su {n_before_pos}")

    # BDY CONDITIONS

    # Cerco i punti di costa a est ed ovest per u
    coast_u_east_west_idx = []
    for j in range(mask_u.shape[0]):
        for i in range(mask_u.shape[1]-1):
          # Se e' un punto terra
          if not mask_u[j,i]:
             t_west = mask_t[j, i]
             t_east = mask_t[j, i+1]
             # se la costa e' a est
             if t_west:
                coast_u_east_west_idx.append((j,i))
             # se la costa e' a ovest
             elif t_east:
                coast_u_east_west_idx.append((j,i))

    # Cerco i punti di costa a nord e sud per v
    coast_v_north_south_idx = []
    for j in range(mask_v.shape[0]-1):
        for i in range(mask_v.shape[1]):
          # Se v e' un punto terra
          if not mask_v[j,i]:
            t_south = mask_t[j, i]
            t_north = mask_t[j+1, i]
            # se la costa e' a nord
            if t_south:
               coast_v_north_south_idx.append((j,i))
            # se la costa e' a sud
            if t_north:
               coast_v_north_south_idx.append((j,i))

    coast_v_north_south_idx = sorted(set(coast_v_north_south_idx))
    coast_u_east_west_idx   = sorted(set(coast_u_east_west_idx))

    # Condizioni  al contorno su ETA lungo la costa
    n_before_eta_bdy = eta_modes.shape[1]
    valid_mask_eta = np.ones(n_before_eta_bdy, dtype=bool)

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
     if valid_mask_eta[k_mode]:  # solo se non e' gia' scartato
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

    # Per la fase calcolo tan-1(-eta_imm/eta_re)
    theta_modes = np.arctan2(-np.imag(eta_modes), np.real(eta_modes))
    # Per l'ampiezza calcolo il modulo sqrt(Re*2+Im*2)
    eta_modes =  np.sqrt(np.real(eta_modes)**2 + np.imag(eta_modes)**2)
    u_modes =  np.sqrt(np.real(u_modes)**2 + np.imag(u_modes)**2)
    v_modes =  np.sqrt(np.real(v_modes)**2 + np.imag(v_modes)**2)

    return omega, period, eta_modes, theta_modes, u_modes, v_modes


# Controllo vorticita' dei modi
def compute_modes_vorticity(eta_modes, u_modes, v_modes, mask_t, mask_u, mask_v,
                            e1u, e2v, coriolis, bathy,
                            threshold_rot=0.3, threshold_grav=0.1):

    n_modes = eta_modes.shape[1]
    vorticity_rms = np.zeros(n_modes)
    vorticity_max = np.zeros(n_modes)
    vorticity_mean = np.zeros(n_modes)
    Pvorticity_rms = np.zeros(n_modes)
    Pvorticity_max = np.zeros(n_modes)
    Pvorticity_mean = np.zeros(n_modes)

    # H su griglia F (centrata tra i punti T) -> shape (ny-1, nx-1), come coriolis
    H_F = 0.25 * (bathy[:-1, :-1] + bathy[1:, :-1] + bathy[:-1, 1:] + bathy[1:, 1:])
    mask_F = np.isfinite(H_F) & (H_F > 0)

    for k_mode in range(n_modes):

        # Ricostruisco u/v 2D dall 1D
        u_field = np.zeros(mask_u.shape)
        v_field = np.zeros(mask_v.shape)
        u_field[mask_u] = u_modes[:, k_mode]
        v_field[mask_v] = v_modes[:, k_mode]

        # Calcolo vorticita' relativa sulla F-grid
        # zeta = dv/dx - du/dy
        dv_dx = (v_field[:, 1:] - v_field[:, :-1]) / e1u[:, :-1]   # (ny, nx-1)
        du_dy = (u_field[1:, :] - u_field[:-1, :]) / e2v[:-1, :]   # (ny-1, nx)
        zeta = dv_dx[:-1, :] - du_dy[:, :-1]                       # (ny-1, nx-1)

        # PV sulla F-grid
        PV = np.full_like(H_F, np.nan, dtype=float)
        PV[mask_F] = (zeta[mask_F] + coriolis[mask_F]) / H_F[mask_F]

        # Vorticita' relativa
        zeta_masked = np.where(mask_F, zeta, np.nan)
        vorticity_rms[k_mode] = np.sqrt(np.nanmean(zeta_masked**2))
        vorticity_max[k_mode] = np.nanmax(zeta_masked)
        vorticity_mean[k_mode] = np.nanmean(zeta_masked)
        # Vorticita' Potenziale
        Pvorticity_rms[k_mode] = np.sqrt(np.nanmean(PV**2))
        Pvorticity_max[k_mode] = np.nanmax(PV)
        Pvorticity_mean[k_mode] = np.nanmean(PV)

    return PV, Pvorticity_rms, Pvorticity_max, Pvorticity_mean, vorticity_rms, vorticity_max, vorticity_mean


# Calcolo PE/KE per i modi
def compute_modes_PE_KE(eta_modes, u_modes, v_modes, mask_t, mask_u, mask_v,
                        e1u, e2v, e1t, e2t, bathy, g=9.81):

    n_modes = eta_modes.shape[1]
    PE_KE_ratio = np.zeros(n_modes)
    PE_modes = np.zeros(n_modes)
    KE_modes = np.zeros(n_modes)

    # Maschera mare valida
    mask_sea = mask_t & np.isfinite(bathy) & (bathy > 0)

    # Area celle
    cell_area = e1t * e2t

    for k_mode in range(n_modes):

        # Ricostruisco u/v/eta 2D dall'1D
        u_field = np.zeros(mask_u.shape)
        v_field = np.zeros(mask_v.shape)
        eta_field = np.zeros(mask_t.shape)
        u_field[mask_u] = u_modes[:, k_mode]
        v_field[mask_v] = v_modes[:, k_mode]
        eta_field[mask_t] = eta_modes[:, k_mode]

        u_T = np.zeros_like(u_field)
        u_T[:, 1:] = 0.5 * (u_field[:, :-1] + u_field[:, 1:])
        v_T = np.zeros_like(v_field)
        v_T[1:, :] = 0.5 * (v_field[:-1, :] + v_field[1:, :])

        # --- Energia cinetica e potenziale ---
        KE = 0.5 * (bathy[mask_sea] + eta_field[mask_sea]) * \
             (u_T[mask_sea]**2 + v_T[mask_sea]**2) * cell_area[mask_sea]
        PE = 0.5 * g * (eta_field[mask_sea]**2) * cell_area[mask_sea]

        # --- Integrazione sul dominio ---
        KE_total = np.nansum(KE)
        PE_total = np.nansum(PE)

        KE_modes[k_mode] = KE_total
        PE_modes[k_mode] = PE_total
        PE_KE_ratio[k_mode] = PE_total / KE_total if KE_total != 0 else np.nan

    return PE_KE_ratio, PE_modes, KE_modes


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

def plot_mode(mode_2d, mask, lon_nemo, lat_nemo, title="", filename="mode_sgn.png", filename_abs="mode_abs.png", cmap="RdBu_r", cmap_abs="gist_stern_r", dpi=150, n_levels=41):

    # 1) Abs plot
    if flag_only_adriatic == 1:
       plt.figure(figsize=(5, 4))
    else:
       plt.figure(figsize=(10, 4))

    # Normalizza a 100 sul valore massimo o sul 99th percentile
    p99 = np.nanpercentile(np.abs(mode_2d), 99)
    norm_mode = np.abs(mode_2d) / p99 * 100

    # Applica la maschera
    masked = np.ma.masked_where(~mask.astype(bool), norm_mode)

    # Costruisce palette discreta centrata in zero
    levels = np.linspace(0, Plot_max, n_levels)
    norm = mpl.colors.BoundaryNorm(levels, ncolors=256)

    # Plot
    cmap_abs = plt.get_cmap(cmap_abs)
    cmap_abs = truncate_colormap(cmap_abs, 0.05, 0.95)
    cmap_abs.set_bad("white")

    # Usa pcolormesh con lon/lat
    im = plt.pcolormesh(lon_nemo, lat_nemo, masked, cmap=cmap_abs, norm=norm, shading='auto')
    cbar = plt.colorbar(im)
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

    # 2) Plot  con il segno dell'ampiezza
    # To be implemented (the phase is needed!)

# Helper: costruisce il flag oscillatorio/rotazionale a partire dai periodi dei
# modi che hanno un corrispettivo tra le soluzioni con f = 0.
def build_oscillatory_flag(period_sorted, oscillatory_periods, tol=0.05):

    period_sorted = np.asarray(period_sorted, dtype=float)
    oscillatory_periods = np.asarray(oscillatory_periods, dtype=float)

    flag = np.zeros(len(period_sorted), dtype=bool)
    for i, p in enumerate(period_sorted):
        flag[i] = np.any(np.abs(oscillatory_periods - p) < tol)

    print(f"build_oscillatory_flag: {flag.sum()} modi oscillatori su {len(flag)}")

    # Controllo: ogni periodo dichiarato oscillatorio deve trovare un match
    for p in oscillatory_periods:
        if not np.any(np.abs(period_sorted - p) < tol):
            print(f"  ATTENZIONE: nessun modo trovato per il periodo {p:.2f} h "
                  f"(tolleranza {tol})")

    return flag


# ----------------------------------------------------------------------
# Helper: scatter di una grandezza per-modo contro il periodo, con lo
# stile concordato (asse lineare decrescente, colori tab, marker diversi
# per le due classi, legenda in basso a destra).
# Usato per RMS, R_eta, PE, KE e PE/KE cosi' che tutte le figure
# diagnostiche siano confrontabili a colpo d'occhio.
# ----------------------------------------------------------------------
def plot_per_mode(period, values, is_oscillatory, ylabel, title, filename,
                  work_dir="", dpi=150, yscale='linear', ylim=None,
                  legend_loc='lower right'):

    period = np.asarray(period, dtype=float)
    values = np.asarray(values, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    if is_oscillatory is not None:
        is_osc = np.asarray(is_oscillatory, dtype=bool)

        if len(is_osc) != len(period):
            raise ValueError(
                f"is_oscillatory ha lunghezza {len(is_osc)} ma ci sono "
                f"{len(period)} modi. Deve essere nello stesso ordine di period."
            )

        ax.plot(period[is_osc], values[is_osc],
                linestyle='none', marker='o', markersize=9,
                color=COL_OSC, markeredgecolor='black', markeredgewidth=0.6,
                label='Oscillatory modes')

        ax.plot(period[~is_osc], values[~is_osc],
                linestyle='none', marker='^', markersize=9,
                color=COL_ROT, markeredgecolor='black', markeredgewidth=0.6,
                label='Rotational modes')

        ax.legend(loc=legend_loc, fontsize=FS_LEGEND, frameon=True)

    else:
        ax.plot(period, values, linestyle='none', marker='o', markersize=9,
                color='0.4', markeredgecolor='black', markeredgewidth=0.6)

    # Asse dei periodi: lineare, decrescente da sinistra a destra
    pmin, pmax = np.nanmin(period), np.nanmax(period)
    ax.set_xlim(pmax + 2, max(pmin - 2, 0))

    ticks = np.arange(0, pmax + RMS_XTICK_STEP, RMS_XTICK_STEP)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0f}" for t in ticks], fontsize=FS_TICK)

    ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(*ylim)
    elif yscale == 'linear':
        vmax = np.nanmax(values)
        if np.isfinite(vmax) and vmax > 0:
            # Margine sotto lo zero: senza, i marker dei modi con valore
            # ~0 (i rotazionali) risultano tagliati a meta' dall'asse.
            ax.set_ylim(-0.04 * vmax, 1.15 * vmax)
    ax.tick_params(axis='y', labelsize=FS_TICK)

    ax.set_xlabel('Period (h)', fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    if title:
        ax.set_title(title, fontsize=FS_LABEL + 1)
    ax.grid(True, alpha=0.35)

    fig.tight_layout()
    fig.savefig(f"{work_dir}/{filename}", dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# Plot di mean e max amplitude per ogni modo
def plot_all_modes_amplitude(modes_2D, period, work_dir="", flag_only_adriatic=0, dpi=150,
                             is_oscillatory=None, R_eta=None,
                             Pvorticity_rms=None, Pvorticity_max=None, Pvorticity_mean=None,
                             vorticity_rms=None, vorticity_max=None, vorticity_mean=None,
                             PE_KE_ratio=None, PE_frac=None, PE=None, KE=None):

    n_modes  = len(modes_2D)
    max_amp  = np.zeros(n_modes)
    mean_amp = np.zeros(n_modes)
    rms_amp  = np.zeros(n_modes)

    # Calcola max e mean per ogni modo
    for i in range(n_modes):
        max_amp[i]  = np.nanmax(np.abs(modes_2D[i]))
        mean_amp[i] = np.nanmean(np.abs(modes_2D[i]))
        rms_amp[i]  = np.sqrt(np.nanmean(np.abs(modes_2D[i])**2))

    period = np.array(period)
    x      = np.arange(len(period))

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

    # ======================================================================
    # Plot RMS in percentuale del massimo su tutti i modi.
    # NOTA: la normalizzazione degli autovettori e' arbitraria (ARPACK), quindi
    # questo valore indica solo la posizione relativa del modo rispetto al piu'
    # ampio; NON dice che l'ampiezza assoluta sia piccola. Per l'affermazione
    # sulla firma trascurabile in SSH si usa R_eta (plot successivo).
    # ======================================================================
    period = np.asarray(period, dtype=float)

    rms_ref = np.nanmax(rms_amp)
    if rms_ref > 0:
        rms_amp_norm = rms_amp / rms_ref * 100.0
    else:
        rms_amp_norm = rms_amp.copy()
        print("ATTENZIONE: massimo RMS nullo, normalizzazione non applicata")

    plot_per_mode(period, rms_amp_norm, is_oscillatory,
                  ylabel='RMS amplitude (% of max)',
                  title='Amplitude RMS per mode',
                  filename='all_modes_rms_amplitude.png',
                  work_dir=work_dir, dpi=dpi, ylim=(-4, 105))

    # ======================================================================
    # Plot R_eta: frazione della norma della soluzione portata da eta.
    #   R_eta = |eta|^2 / (|u|^2 + |v|^2 + |eta|^2)
    # Essendo un rapporto, e' INDIPENDENTE dalla normalizzazione arbitraria
    # degli autovettori: valori bassi identificano i modi quasi-orizzontali,
    # la cui firma e' contenuta soprattutto nel campo di velocita' e non nel
    # livello del mare.
    # ======================================================================
    if R_eta is not None:
        plot_per_mode(period, np.asarray(R_eta, dtype=float) * 100.0,
                      is_oscillatory,
                      ylabel=r'Fraction of solution norm in $\hat{\eta}$ (%)',
                      title=r'Sea level share of the mode norm',
                      filename='all_modes_R_eta.png',
                      work_dir=work_dir, dpi=dpi)

        # Stessa grandezza in scala logaritmica, utile se le due classi
        # differiscono di ordini di grandezza
        plot_per_mode(period, np.asarray(R_eta, dtype=float) * 100.0,
                      is_oscillatory,
                      ylabel=r'Fraction of solution norm in $\hat{\eta}$ (%)',
                      title=r'Sea level share of the mode norm (log scale)',
                      filename='all_modes_R_eta_log.png',
                      work_dir=work_dir, dpi=dpi, yscale='log',
                      legend_loc='best')

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
        plt.plot(range(n_modes), vorticity_mean, 'o-', label="Mean vorticity")
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
        plt.plot(range(n_modes), Pvorticity_mean, 'o-', label="Mean PV", color='tab:blue')
        plt.xticks(range(n_modes), [f"{p:.2f}h" for p in period], rotation=45)
        plt.xlabel("Mode (Period)")
        plt.ylabel("Mean PV (1/(m s))")
        plt.title("Mean Potential Vorticity per mode")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{work_dir}/all_modes_mean_Pvorticity.png", dpi=dpi)
        plt.close()

    if Pvorticity_rms is not None:
        plt.figure(figsize=(12,4))
        plt.plot(range(n_modes), Pvorticity_rms, 'o-', label="RMS PV", color='tab:red')
        plt.xticks(range(n_modes), [f"{p:.2f}h" for p in period], rotation=45)
        plt.xlabel("Mode (Period)")
        plt.ylabel("RMS PV (1/(m s))")
        plt.title("RMS Potential Vorticity per mode")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{work_dir}/all_modes_rms_Pvorticity.png", dpi=dpi)
        plt.close()

    if Pvorticity_max is not None:
        plt.figure(figsize=(12,4))
        plt.plot(range(n_modes), Pvorticity_max, 'o-', label="Max PV", color='tab:orange')
        plt.xticks(range(n_modes), [f"{p:.2f}h" for p in period], rotation=45)
        plt.xlabel("Mode (Period)")
        plt.ylabel("Max PV (1/(m s))")
        plt.title("Max Potential Vorticity per mode")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{work_dir}/all_modes_max_Pvorticity.png", dpi=dpi)
        plt.close()

    if PE_KE_ratio is not None:
        plot_per_mode(period, PE_KE_ratio, is_oscillatory,
                      ylabel='PE / KE',
                      title='PE/KE ratio per mode',
                      filename='all_modes_PE_KE_ratio.png',
                      work_dir=work_dir, dpi=dpi,
                      legend_loc='best')

    # Frazione di energia potenziale: stessa informazione di PE/KE ma
    # limitata tra 0 e 1 (equipartizione -> 0.5, quasi-orizzontali -> ~0).
    # Confrontabile direttamente con R_eta, che ha lo stesso intervallo.
    if PE_frac is not None:
        plot_per_mode(period, PE_frac, is_oscillatory,
                      ylabel='PE / (PE + KE)',
                      title='Potential energy share per mode',
                      filename='all_modes_PE_frac.png',
                      work_dir=work_dir, dpi=dpi, ylim=(-0.04, 1.05),
                      legend_loc='best')

    if PE is not None:
        plot_per_mode(period, PE, is_oscillatory,
                      ylabel=r'PE [m$^5$/s$^2$]',
                      title='Potential Energy per unit density per mode',
                      filename='all_modes_PE.png',
                      work_dir=work_dir, dpi=dpi)

    if KE is not None:
        plot_per_mode(period, KE, is_oscillatory,
                      ylabel=r'KE [m$^5$/s$^2$]',
                      title='Kinetic Energy per unit density per mode',
                      filename='all_modes_KE.png',
                      work_dir=work_dir, dpi=dpi)


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

########### Real modes ##############

if flag_compute_modes != 0 :

   print ('Compute K operator')
   K, mapping_u, mapping_v, mapping_t, invmap_t, invmap_u, invmap_v  = build_operator_K(mask_t, mask_u, mask_v, bathy, coriolis, dxu, dyv, dxt, dyt)
   print ('Done!')

   print ('Compute the modes')
   omega, period, eta_modes, theta_modes, u_modes, v_modes = compute_barotropic_modes_K_eta_only(
       K, mask_t, mask_u, mask_v, dxu, dyv, dxt, dyt, bathy, coriolis, mapping_t,
       k=mode_num, which=eig_order, reference_period=reference_period)
   modes=eta_modes
   print ('Done!')

   # PRIMA della selezione: cosa ha restituito il solver
   print("Periodi calcolati (tutti):", np.sort(period)[::-1])
   print("Sopra 30 h:", np.sort(period[period > 30])[::-1])

   print ('Select modes with periods in the following range [h]:',Tmin,Tmax)
   valid = np.where((period >= Tmin) & (period <= Tmax))[0]
   omega, period = omega[valid], period[valid]
   modes = modes[:, valid]
   eta_modes   = eta_modes[:, valid]
   u_modes     = u_modes[:, valid]
   v_modes     = v_modes[:, valid]
   theta_modes = theta_modes[:, valid]
   k = len(omega)
   print (k,' modes selected')
   print ('Done!')

   print ('Build 2d modes')
   shape = mask_t.shape
   modes_2D = reconstruct_modes(modes, invmap_t, shape)
   print ('Done!')

   # ------------------------------------------------------------------
   # R_eta: frazione della norma della soluzione portata dalla componente
   # di livello del mare.
   #     R_eta = |eta|^2 / (|u|^2 + |v|^2 + |eta|^2)
   # Gli autovettori sono normalizzati da ARPACK a norma unitaria sul
   # vettore completo (u, v, eta), quindi il singolo valore di ampiezza non
   # ha scala fisica. R_eta, essendo un rapporto, non dipende da quella
   # normalizzazione: misura quanta parte del modo sta nel livello del mare
   # invece che nelle correnti, cioe' esattamente la quasi-orizzontalita'
   # dei modi rotazionali.
   # NB: u, v ed eta hanno unita' diverse, quindi R_eta e' un indicatore
   # adimensionale e non un rapporto energetico.
   # ------------------------------------------------------------------
   print ('Compute R_eta (sea level share of the mode norm)')
   n_eta = np.sum(eta_modes**2, axis=0)
   n_u   = np.sum(u_modes**2,   axis=0)
   n_v   = np.sum(v_modes**2,   axis=0)
   R_eta = n_eta / (n_eta + n_u + n_v)
   print ('Done!')

   if vorticity_flag == 1 :
      print("Compute vorticity for each mode")
      vorticity, Pvorticity_rms, Pvorticity_max, Pvorticity_mean, vorticity_rms, vorticity_max, vorticity_mean  = compute_modes_vorticity( eta_modes, u_modes, v_modes, mask_t, mask_u, mask_v, dxu, dyv, coriolis, bathy)

   if flag_PE_KE_ratio == 1:
      print("Compute PE/KE for each mode")
      PE_KE_ratio, PE, KE  = compute_modes_PE_KE( eta_modes, u_modes, v_modes, mask_t, mask_u, mask_v, dxu, dyv, dxt, dyt, bathy)
      # Frazione di energia potenziale: PE/(PE+KE), limitata tra 0 e 1.
      # Contiene la stessa informazione di PE/KE (r/(1+r)) ma essendo
      # limitata e' piu' leggibile. Equipartizione -> 0.5; modi
      # quasi-orizzontali (rotazionali) -> ~0.
      with np.errstate(divide='ignore', invalid='ignore'):
          PE_frac = np.where((PE + KE) > 0, PE / (PE + KE), np.nan)

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

# inizializzazione
PE_KE_ratio_sorted = None
PE_frac_sorted = None
PE_sorted = None
KE_sorted = None

# R_eta esiste solo se i modi sono stati ricalcolati (non nel ramo di load)
try:
    R_eta_sorted = [R_eta[m] for m in sorted_indices]
except NameError:
    R_eta_sorted = None
    print("R_eta non disponibile (modi caricati da file)")

if flag_PE_KE_ratio == 1:
   PE_KE_ratio_sorted = [PE_KE_ratio[m] for m in sorted_indices]
   PE_frac_sorted = [PE_frac[m] for m in sorted_indices]
   PE_sorted = [PE[m] for m in sorted_indices]
   KE_sorted = [KE[m] for m in sorted_indices]

# Classificazione oscillatori/rotazionali per la colorazione della Fig. 1
if flag_color_by_f0 == 1:
   is_osc_sorted = build_oscillatory_flag(period_sorted, oscillatory_periods,
                                          tol=oscillatory_tol)
else:
   is_osc_sorted = None

plot_all_modes_amplitude(modes_sorted, period_sorted, work_dir=work_dir,
                         is_oscillatory=is_osc_sorted, R_eta=R_eta_sorted,
                         Pvorticity_rms=Pvorticity_rms_sorted, Pvorticity_max=Pvorticity_max_sorted,
                         Pvorticity_mean=Pvorticity_mean_sorted, vorticity_rms=vorticity_rms_sorted,
                         vorticity_max=vorticity_max_sorted, vorticity_mean=vorticity_mean_sorted,
                         PE_KE_ratio=PE_KE_ratio_sorted, PE_frac=PE_frac_sorted,
                         PE=PE_sorted, KE=KE_sorted )
