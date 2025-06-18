import netCDF4 as nc
import xarray as xr
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

#####################
# Inputs
work_dir           = "/work/cmcc/ag15419/basin_modes_sa/"
mode_num           = 100
reference_period   = 12
Tmin               = 1
Tmax               = 40

mesh_mask_file     = "/work/cmcc/ag15419/VAA_paper/DATA0/mesh_mask.nc"
bathy_meter_file   = "/work/cmcc/ag15419/VAA_paper/DATA0/bathy_meter.nc"

outfile_R          = work_dir+'med_modes_'+str(mode_num)+'.nc'
outfile_C          = work_dir+'med_modes_'+str(mode_num)+'_C.nc'

flag_compute_modes = 1
flag_only_adriatic = 1

#####################
def prepare_fields(meshmask_path, bathy_path):
    # Open input files
    print ('Open file',meshmask_path)
    ds_mask = xr.open_dataset(meshmask_path, decode_times=False)
    print ('Done!')
    print ('Open file',bathy_path)
    ds_bathy = xr.open_dataset(bathy_path, decode_times=False)
    print ('Done!')

    # Land/sea mask
    print ('Reading land/sea mask..') 
    mask = ds_mask['tmask'].isel(t=0, z=0).values.astype(bool)
    print ('Done!')

    # Bathymetry
    print ('Reading Bathymetry..')
    bathy = ds_bathy['Bathymetry'].values
    bathy = np.where(mask, bathy, np.nan)
    print ('Done!')

    print ('Computing f Coriolis..')
    # Lat
    lat = ds_mask['nav_lat'].values
    # Coriolis f
    omega = 7.292115e-5  # rad/s
    coriolis = 2 * omega * np.sin(np.deg2rad(lat))
    print ('Done!')

    # Grid (dx, dy)
    print ('Reading the horizontal grid..')
    dx = ds_mask['e1t'].values  # m
    dy = ds_mask['e2t'].values  # m
    print ('Done!')

    return mask, bathy, coriolis, dx, dy

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

def build_operator_A(mask, bathy, coriolis, dx, dy, g=9.81):
    """
    Costruisce l'operatore A in forma sparsa usando dx, dy variabili su griglia 2D.

    Parametri:
    - mask: 2D array binario (1 = mare, 0 = terra)
    - bathy: 2D array con profondità H (m)
    - coriolis: 2D array con f (s⁻¹)
    - dx, dy: 2D array con passo griglia in x e y (m)
    - g: accelerazione gravitazionale (default = 9.81 m/s²)

    Ritorna:
    - A: matrice sparsa (Nwet x Nwet)
    - mapping: dizionario (i,j) → indice vettoriale
    - invmap: dizionario indice vettoriale → (i,j)
    """
    ny, nx = mask.shape
    mapping = {}
    invmap = {}
    idx = 0

    # Mappa (i,j) <-> indice vettoriale
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
        H = bathy[0, j, i]
        f = coriolis[j, i]
        diag = -f**2 * H  # termine rotazionale

        # Punti griglia vicini in x e y dir
        for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < nx and 0 <= nj < ny and mask[nj, ni]:
                Hij = 0.5 * (bathy[0, j, i] + bathy[0, nj, ni])

                if di != 0:  # dir x
                    dxij = 0.5 * (dx[0, j, i] + dx[0,nj, ni])
                    dxy = dxij ** 2
                else:        # dir y
                    dyij = 0.5 * (dy[0, j, i] + dy[0, nj, ni])
                    dxy = dyij ** 2

                coeff = -g * Hij / dxy
                n_idx = mapping[(ni, nj)]

                rows.append(k)
                cols.append(n_idx)
                data.append(coeff)

                diag += g * Hij / dxy

        # termine diagonale
        rows.append(k)
        cols.append(k)
        data.append(diag)

    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    return A, mapping, invmap

def compute_barotropic_modes(A, k=10, which='LM'):
    """
    Calcola i primi k modi normali barotropici risolvendo:
        A η = ω**2 η

    Parametri:
    - A: matrice sparsa (costruita da build_operator_A)
    - k: numero di modi da calcolare (default 10)
    - which: quali autovalori ('SM' = Smallest Magnitude; 'LM' = Largest Magnitude)
    oppure:
   - sigma: calcolato in base al parametro fornito in input 

    Ritorna:
    - omega: array 1D di frequenze (rad/s)
    - modes: matrice 2D (Nwet x k) con i modi normalizzati
    """
    # Compute sigma
    Tref_sec = reference_period * 3600
    omega_ref = 2 * np.pi / Tref_sec
    sigma=-omega_ref**2

    # Solve A η = λ η, with λ = ω**2
    #eigvals, eigvecs = eigsh(A, k=k, which=which)
    eigvals, eigvecs = eigsh(A, k=k, sigma=sigma, which='LM', mode='normal')

    # Frequenze in rad/s (ω = sqrt(λ))
    omega = np.sqrt(np.abs(eigvals))  # abs to avoid numerical issues
    # Periods in h
    period = 2 * np.pi / omega / 3600  # hours

    print ('periods:',period)

    return omega, period, eigvecs

def reconstruct_modes(modes, invmap, shape):
    """
    Ricostruisce i modi 2D a partire dagli autovettori 1D.

    Parametri:
    - modes: array 2D (Nwet x Nmodes)
    - invmap: dizionario indice → (i, j)
    - shape: tuple (ny, nx) del dominio originale

    Ritorna:
    - modes_2D: array 3D (Nmodes, ny, nx)
    """
    Nmodes = modes.shape[1]
    ny, nx = shape
    modes_2D = np.full((Nmodes, ny, nx), np.nan)

    for m in range(Nmodes):
        for idx, (i, j) in invmap.items():
            modes_2D[m, j, i] = modes[idx, m]

    return modes_2D

def save_modes_to_netcdf(filename, modes_2D, periods, mask=None):
    """
    Salva i modi 2D in un file NetCDF.
    
    Parametri:
    - filename: nome del file .nc
    - modes_2D: array (k, ny, nx) dei modi
    - periods: array (k,) dei periodi in ore
    - mask: array (ny, nx) opzionale
    """
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

def load_modes_from_netcdf(filename):
    with nc.Dataset(filename) as ds:
        modes_2D = ds.variables['modes'][:]
        periods = ds.variables['periods'][:]
        mask = ds.variables['mask'][:] if 'mask' in ds.variables else None
    return modes_2D, periods, mask

def plot_mode(mode_2d, mask, title="", filename="mode.png", cmap="RdBu_r", dpi=150, n_levels=51):
    """
    Plotta e salva un singolo modo barotropico in 2D, normalizzato in modo che
    il massimo valore assoluto sia 100.

    Parametri:
    - mode_2d: array 2D (ny, nx) con valori del modo
    - mask: 2D array (1 = mare, 0 = terra)
    - title: titolo del grafico
    - filename: nome del file immagine da salvare
    - cmap: colormap (default = "RdBu_r")
    - dpi: risoluzione in output
    - n_levels: numero di livelli discreti nella colormap
    """
    plt.figure(figsize=(10, 6))

    # Normalizza a 100
    norm_mode = mode_2d / np.nanmax(np.abs(mode_2d)) * 100

    # Applica maschera
    masked = np.ma.masked_where(~mask.astype(bool), norm_mode)

    # Costruisce palette discreta centrata in zero
    levels = np.linspace(-100, 100, n_levels)
    norm = mpl.colors.BoundaryNorm(levels, ncolors=256)

    # Linea di costa (contorno della maschera)
    plt.contour(mask, levels=[0.5], colors='k', linewidths=0.5)

    # Plot
    im = plt.imshow(masked, cmap=cmap, norm=norm)
    cbar = plt.colorbar(im, orientation='horizontal', ticks=levels[::2])
    cbar.set_label("Mode amplitude (%)")

    plt.title(title)
    plt.xlabel("i")
    plt.ylabel("j")
    plt.xlim(300,1307)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()

def build_complex_mode(eta1, eta2):
    """
    Costruisce il modo complesso η = η1 + i η2
    e restituisce ampiezza e fase spaziali.
    """
    eta_complex = eta1 + 1j * eta2
    amplitude = np.abs(eta_complex)
    phase = np.angle(eta_complex)  # in radianti
    return amplitude, phase

def plot_amplitude_phase(amplitude, phase, mask, prefix="mode", dpi=150):
    import matplotlib.pyplot as plt
    import numpy as np

    amp_masked = np.ma.masked_where(~mask, amplitude)
    pha_masked = np.ma.masked_where(~mask, phase)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    im1 = axs[0].imshow(amp_masked, cmap='viridis')
    axs[0].set_title("Ampiezza |\u03b7|")
    axs[0].invert_yaxis()
    plt.colorbar(im1, ax=axs[0], orientation='horizontal')

    im2 = axs[1].imshow(pha_masked, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axs[1].set_title("Fase \u03b1 (rad)")
    axs[1].invert_yaxis()
    plt.colorbar(im2, ax=axs[1], orientation='horizontal')

    plt.tight_layout()
    plt.savefig(f"{work_dir}/{prefix}_amp_phase.png", dpi=dpi)
    plt.close()

def save_amp_phase_to_netcdf(filename, amplitudes, phases, periods):
    """
    Salva ampiezze, fasi e periodi dei modi complessi in un file NetCDF.
    """
    Nmodes, ny, nx = amplitudes.shape
    with nc.Dataset(filename, 'w') as ds:
        ds.createDimension('mode', Nmodes)
        ds.createDimension('y', ny)
        ds.createDimension('x', nx)

        amp_var = ds.createVariable('amplitude', 'f4', ('mode', 'y', 'x'))
        pha_var = ds.createVariable('phase', 'f4', ('mode', 'y', 'x'))
        per_var = ds.createVariable('period', 'f4', ('mode',))

        amp_var.units = 'm'
        pha_var.units = 'radians'
        per_var.units = 'hours'

        amp_var[:] = amplitudes
        pha_var[:] = phases
        per_var[:] = periods

    print(f"Salvato file NetCDF: {filename}")

def load_complex_modes_from_netcdf(filename):
    """
    Carica ampiezze, fasi e periodi da un file NetCDF con modi complessi.

    Parametri:
    - filename: percorso del file .nc da leggere

    Ritorna:
    - amplitudes: array 3D (Nmodes, ny, nx)
    - phases:     array 3D (Nmodes, ny, nx)
    - periods:    array 1D (Nmodes,) in ore
    """
    with nc.Dataset(filename, 'r') as ds:
        amplitudes = ds.variables['amplitude'][:]
        phases = ds.variables['phase'][:]
        periods = ds.variables['period'][:]
    return amplitudes, phases, periods

################### MAIN #############
# Prepare input fields
print ('Preparing input fields..')
mask, bathy, coriolis, dx, dy = prepare_fields(mesh_mask_file, bathy_meter_file)
print ('Done!')

# Compute and plot the Med modes
print ('Mask the Atlantic Box')
ny, nx = mask.shape
J, I = np.indices((ny, nx))
atlantic_mask = (I < 300) | ((I < 415) & (J > 250))
mask[atlantic_mask] = 0

# Mask also all the subregions except from the Adriatic Sea (temporary for test)
if flag_only_adriatic == 1 :
   # cut the Thyrrenian box
   J, I = np.indices((ny, nx))
   box_mask = (I < 820) & (J < 285)
   mask[box_mask] = 0
   # Select the Adriatic Sea
   adriatic_mask = np.zeros_like(mask)
   i_min, i_max = 720, 920
   j_min, j_max = 220, 380
   adriatic_mask[j_min:j_max, i_min:i_max] = mask[j_min:j_max, i_min:i_max]
   mask = adriatic_mask

# Plot input fields
print ('Plotting input fields..')
plot_input_fields(np.squeeze(mask), np.squeeze(bathy), np.squeeze(coriolis), np.squeeze(dx), np.squeeze(dy), filename=work_dir+"input_fields"+str(mode_num)+".png")
print ('Done!')

########### Real modes ##############3

print ('Done!')
if flag_compute_modes != 0 :
   print ('Compute A operator..')
   A, mapping, invmap = build_operator_A(mask, bathy, coriolis, dx, dy)
   print ('Done!')
   print ('Compute the modes')
   omega, period, modes = compute_barotropic_modes(A, k=mode_num)
   print ('Done!')
   print ('Select modes with periods in the following range [h]:',Tmin,Tmax)
   valid = np.where((period >= Tmin) & (period <= Tmax))[0]
   omega, period = omega[valid], period[valid]
   modes = modes[:, valid]
   k = len(omega)
   print (k,' modes selected')
   print ('Done!')
   print ('Build 2d modes')
   shape = mask.shape
   modes_2D = reconstruct_modes(modes, invmap, shape)
   print ('Done!')
   print ('Save the R modes')
   save_modes_to_netcdf(outfile_R, modes_2D, period, mask=mask)
   print ('Done!')

else:
   print ('Load the R modes')
   modes_2D, period, mask = load_modes_from_netcdf(outfile_R)

print ('Plot the R modes amplitude')
for m in range(mode_num):
    title    = f"Barotropic mode {m+1} - Period {period[m]:0.2f} h"
    filename = f"/work/cmcc/ag15419/basin_modes_sa/mode_{m+1:02d}_{mode_num}.png"
    plot_mode(modes_2D[m], mask, title=title, filename=filename)
print ('Done!')

########### Complex modes ############
## Compute or load complex modes
#if flag_compute_modes != 0:
#    print('Compute the C modes')
#    tol = 0.01  # tolleranza in ore per accoppiare periodi
#    used = set()
#    amp_list = []
#    pha_list = []
#    per_list = []
#
#    for i in range(mode_num):
#        if i in used:
#            continue
#        for j in range(i+1, mode_num):
#            if j in used:
#                continue
#            if abs(period[i] - period[j]) < tol:
#                print(f"Modo {i+1} e {j+1} accoppiati: T = {period[i]:.4f} h")
#                eta1 = modes_2D[i]
#                eta2 = modes_2D[j]
#                amplitude, phase = build_complex_mode(eta1, eta2)
#                amp_list.append(amplitude)
#                pha_list.append(phase)
#                per_list.append(period[i])  # o media
#                used.add(i)
#                used.add(j)
#                break
#
#    if amp_list:
#        amp_arr = np.array(amp_list)
#        pha_arr = np.array(pha_list)
#        per_arr = np.array(per_list)
#        save_amp_phase_to_netcdf(outfile_C, amp_arr, pha_arr, per_arr)
#else:
#    print('Load the C modes')
#    amp_arr, pha_arr, per_arr = load_complex_modes_from_netcdf(outfile_C)
#
## Plot sempre, indipendentemente dal flag
#print('Plot the C modes')
#for m in range(len(per_arr)):
#    prefix = f"complex_mode_{m+1:02d}"
#    plot_amplitude_phase(amp_arr[m], pha_arr[m], mask, prefix=prefix)
#print('Done!')
