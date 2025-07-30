import netCDF4 as nc
import xarray as xr
#import xesmf as xe
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import LinearSegmentedColormap
mpl.use('Agg')

#####################
# INPUTS
# Work directory
work_dir           = "/work/cmcc/ag15419/basin_modes_sa_Ag/"
# Num of modes to be analyzed
mode_num           = 1
# The code starts to look for modes around the following period [h]
reference_period   = 24
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

# NEMO Mesh
mesh_mask_file     = "/work/cmcc/ag15419/exp/old/fix_mfseas9_longrun_hmslp_2NT_AB/EXP00_10ok/20150101/model/mesh_mask.nc" #"/work/cmcc/ag15419/VAA_paper/DATA0/mesh_mask.nc"
# NEMO Bathymetry
bathy_meter_file   = "/work/cmcc/ag15419/VAA_paper/DATA0/bathy_meter.nc"

# Outfiles 
outfile_R          = work_dir+'med_modes_'+str(mode_num)+'.nc'
#outfile_C          = work_dir+'med_modes_'+str(mode_num)+'_C.nc'

# If you want to compute the mode flag_compute_modes = 1 
flag_compute_modes = 1

# To run the code on the Adriatic Sea area set flag_only_adriatic = 1
flag_only_adriatic = 1

# To set f term (1=rot+grav modes, 0=only gravitational contribution, 2=f cost+grav modes)
flag_f             = 0

# To use the original GEBCO bathy instead of the MedFS bathy with the 4000m cut (interpolation on the MedFS grid is required)
flag_gebco_bathy   = 0
gebco_bathy        = "/work/cmcc/ag15419/VAA_paper/DATA0/gebco_2024_n46.5_s30.0_w-19.0_e37.0.nc"
gebco_bathy_int    = work_dir+'bathy_gebco_int.nc' 

#####################
def prepare_fields(meshmask_path, bathy_path):

    # Open input files
    ds_mask = xr.open_dataset(meshmask_path, decode_times=False, drop_variables=["x", "y"])
    ds_bathy = xr.open_dataset(bathy_path, decode_times=False, drop_variables=["x", "y"])

    # Land/sea mask
    print ('Land Sea Mask')
    mask = ds_mask['tmask'].isel(time_counter=0, nav_lev=0).values.astype(bool)

    # Bathymetry
    print ('Bathymetry')
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
        lat_nemo = ds_bathy['nav_lat'].isel().values
        lon_nemo = ds_bathy['nav_lon'].isel().values
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

    print ('Lat')
    # Lat
    lat = ds_bathy['nav_lat'].values
    print ('Coriolis')
    # Coriolis f
    omega = 7.292115e-5  # rad/s
    if flag_f != 2:
       coriolis = 2 * omega * np.sin(np.deg2rad(lat))
    elif flag_f == 2:
       lat_fix=lat*0+37.75
       coriolis = 2 * omega * np.sin(np.deg2rad(lat_fix))
       print ('Constant Coriolis:',coriolis)
 
    print ('H Grid')
    # Grid (dx, dy, dz)
    dxt = ds_mask['e1t'].isel(time_counter=0).values  # m
    dyt = ds_mask['e2t'].isel(time_counter=0).values  # m
    dxu = ds_mask['e1u'].isel(time_counter=0).values  # m
    dyu = ds_mask['e2u'].isel(time_counter=0).values  # m
    dxv = ds_mask['e1v'].isel(time_counter=0).values  # m
    dyv = ds_mask['e2v'].isel(time_counter=0).values  # m
    print ('V Grid')
    dzt = ds_mask['e3t_0'].isel(time_counter=0, nav_lev=0).values

    return mask, bathy, coriolis, dxu, dyu, dxv, dyv, dxt, dyt, dzt 

def plot_input_fields(mask, bathy, dz, coriolis, dx, dy, filename="input_fields.png", dpi=150):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # Plot land/sea mask
    im0 = axs[0, 0].imshow(mask, cmap='gray')
    axs[0, 0].set_title("Land/Sea Mask (1=Sea, 0=Land)")
    plt.colorbar(im0, ax=axs[0, 0], orientation='horizontal')

    # Plot bathymetry
    im1 = axs[0, 1].imshow(bathy, cmap='viridis')
    axs[0, 1].set_title("Bathymetry (m)")
    plt.colorbar(im1, ax=axs[0, 1], orientation='horizontal')

    # Plot dz - bathy difference
    dz_diff = dz - bathy
    im2 = axs[0, 2].imshow(dz_diff, cmap='seismic', vmin=-np.nanmax(abs(dz_diff)), vmax=np.nanmax(abs(dz_diff)))
    axs[0, 2].set_title("dz - Bathymetry (m)")
    plt.colorbar(im2, ax=axs[0, 2], orientation='horizontal')

    # Plot dx
    im3 = axs[1, 0].imshow(dx, cmap='plasma')
    axs[1, 0].set_title("Grid spacing dx (m)")
    plt.colorbar(im3, ax=axs[1, 0], orientation='horizontal')

    # Plot dy
    im4 = axs[1, 1].imshow(dy, cmap='plasma')
    axs[1, 1].set_title("Grid spacing dy (m)")
    plt.colorbar(im4, ax=axs[1, 1], orientation='horizontal')

    # Plot coriolis in lower right
    im5 = axs[1, 2].imshow(coriolis, cmap='coolwarm')
    axs[1, 2].set_title("Coriolis parameter (1/s)")
    plt.colorbar(im5, ax=axs[1, 2], orientation='horizontal')

    # Label axes
    for ax in axs.flat:
        ax.set_xlabel("i")
        ax.set_ylabel("j")
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()

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
        if flag_f != 0:
           diag = - f**2  # termine rotazionale solo sulla diag
        elif flag_f == 0:
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
                diag = diag - coeff         # termine non rot sulla diagonale

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
                diag = diag - coeff        # termine non rot sulla diagonale

        # Scrivo i valori calcolati per la diagonale
        rows.append(k)
        cols.append(k)
        data.append(diag)

    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    print("A symmetric?", (A - A.T).nnz == 0)
    print ('Prova',A)
    return A, mapping, invmap

def compute_barotropic_modes(A, k=10, which='LM', reference_period=24):
    # Compute sigma (target eigenvalue)
    Tref_sec = reference_period * 3600
    omega_ref = 2 * np.pi / Tref_sec
    sigma = - omega_ref**2  

    # Solve perche' l'eq. e' A eta = lambda eta
    eigvals, eigvecs = eigsh(A, k=k, sigma=sigma, which=which, mode='normal')
    #eigvals, eigvecs =eigsh(A, k=k, which=which)
    print("Eigenvalues:", eigvals)

    # Consider only physical eigenvalues
    valid = eigvals < 0
    eigvals = eigvals[valid]
    eigvecs = eigvecs[:, valid]
    print("Eigenvalues with physical relevance:", eigvals)

    # Frequenze omega = sqrt(lambda)
    omega = np.sqrt(-eigvals)
    #omega = np.sqrt(np.abs(eigvals))

    # Periodi in ore
    period = 2 * np.pi / omega / 3600
    print('Selected periods:', period)

    return omega, period, eigvecs

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

def load_modes_from_netcdf(filename):
    with nc.Dataset(filename) as ds:
        modes_2D = ds.variables['modes'][:]
        periods = ds.variables['periods'][:]
        mask = ds.variables['mask'][:] if 'mask' in ds.variables else None
    return modes_2D, periods, mask

def plot_mode(mode_2d, mask, title="", filename="mode.png", filename_abs="mode_abs.png", cmap="RdBu_r", cmap_abs="gist_stern_r", dpi=150, n_levels=51): 

    plt.figure(figsize=(10, 6))

    # Normalizza a 100
    norm_mode = mode_2d / np.nanmax(np.abs(mode_2d)) * 100

    # Applica maschera
    masked = np.ma.masked_where(~mask.astype(bool), norm_mode)

    # Costruisce palette discreta centrata in zero
    levels = np.linspace(-Plot_max, Plot_max, n_levels)
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
    if flag_only_adriatic == 1 :
       plt.xlim(720, 920)
       plt.ylim(380, 200)
    else:
       plt.xlim(300,1307)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()

    #########
    # Abs plot
    plt.figure(figsize=(10, 6))

    # Prendo abs 
    masked=np.abs(masked)    

    # Costruisce palette discreta centrata in zero
    levels = np.linspace(0, Plot_max, n_levels)
    norm = mpl.colors.BoundaryNorm(levels, ncolors=256)

    # Linea di costa (contorno della maschera)
    plt.contour(mask, levels=[0.5], colors='k', linewidths=0.5)

    # Plot
    cmap_abs = mpl.cm.get_cmap(cmap_abs)
    cmap_abs = truncate_colormap(cmap_abs, 0.05, 0.95)
    cmap_abs.set_bad("white")

    im = plt.imshow(masked, cmap=cmap_abs, norm=norm)
    cbar = plt.colorbar(im, orientation='horizontal', ticks=levels[::2])
    cbar.set_label("Mode amplitude (%)")

    contour_levels = np.arange(0, Plot_max + 1, 10)  
    CS = plt.contour(masked, levels=contour_levels, colors='k', linewidths=0.4)
    plt.clabel(CS, inline=True, fontsize=6, fmt='%d%%')

    plt.title(title)
    plt.xlabel("i")
    plt.ylabel("j")
    if flag_only_adriatic == 1 :
       plt.xlim(720, 920)
       plt.ylim(380, 200)
    else:
       plt.xlim(300,1307)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename_abs, dpi=dpi)
    plt.close()

#def build_complex_mode(eta1, eta2):
#
#    eta_complex = eta1 + 1j * eta2
#    amplitude = np.abs(eta_complex)
#    phase = np.angle(eta_complex)  # in radianti
#    return amplitude, phase
#
#def plot_amplitude_phase(amplitude, phase, mask, prefix="mode", dpi=150):
#
#    amp_masked = np.ma.masked_where(~mask, amplitude)
#    pha_masked = np.ma.masked_where(~mask, phase)
#
#    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
#
#    im1 = axs[0].imshow(amp_masked, cmap='viridis')
#    axs[0].set_title("Ampiezza |\u03b7|")
#    axs[0].invert_yaxis()
#    plt.colorbar(im1, ax=axs[0], orientation='horizontal')
#
#    im2 = axs[1].imshow(pha_masked, cmap='twilight', vmin=-np.pi, vmax=np.pi)
#    axs[1].set_title("Fase \u03b1 (rad)")
#    axs[1].invert_yaxis()
#    plt.colorbar(im2, ax=axs[1], orientation='horizontal')
#
#    plt.tight_layout()
#    plt.savefig(f"{work_dir}/{prefix}_amp_phase.png", dpi=dpi)
#    plt.close()
#
#def save_amp_phase_to_netcdf(filename, amplitudes, phases, periods):
#
#    Nmodes, ny, nx = amplitudes.shape
#    with nc.Dataset(filename, 'w') as ds:
#        ds.createDimension('mode', Nmodes)
#        ds.createDimension('y', ny)
#        ds.createDimension('x', nx)
#
#        amp_var = ds.createVariable('amplitude', 'f4', ('mode', 'y', 'x'))
#        pha_var = ds.createVariable('phase', 'f4', ('mode', 'y', 'x'))
#        per_var = ds.createVariable('period', 'f4', ('mode',))
#
#        amp_var.units = 'm'
#        pha_var.units = 'radians'
#        per_var.units = 'hours'
#
#        amp_var[:] = amplitudes
#        pha_var[:] = phases
#        per_var[:] = periods
#
#    print(f"Salvato file NetCDF: {filename}")
#
#def load_complex_modes_from_netcdf(filename):
#    with nc.Dataset(filename, 'r') as ds:
#        amplitudes = ds.variables['amplitude'][:]
#        phases = ds.variables['phase'][:]
#        periods = ds.variables['period'][:]
#    return amplitudes, phases, periods

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
mask, bathy, coriolis, dxu, dyu, dxv, dyv, dxt, dyt, dzt = prepare_fields(mesh_mask_file, bathy_meter_file)
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
   box_mask = (I < 775) & (J < 285) #825 285
   mask[box_mask] = 0
   box_mask = (I < 825) & (J < 260) #825 285
   mask[box_mask] = 0
   # Select the Adriatic Sea
   adriatic_mask = np.zeros_like(mask)
   i_min, i_max = 720, 920
   j_min, j_max = 200, 380
   adriatic_mask[j_min:j_max, i_min:i_max] = mask[j_min:j_max, i_min:i_max]
   mask = adriatic_mask

# Plot input fields
print ('Plotting input fields..')
plot_input_fields(np.squeeze(mask), np.squeeze(bathy), np.squeeze(dzt), np.squeeze(coriolis), np.squeeze(dxt), np.squeeze(dyt), filename=work_dir+"/input_fields"+str(mode_num)+".png")
print ('Done!')

########### Real modes ##############3

if flag_compute_modes != 0 :

   print ('Compute A operator')
   A, mapping, invmap = build_operator_A(mask, bathy, coriolis, dxu, dyv, dxt, dyt)
   print ('Done!')

   print ('Compute the modes')
   omega, period, modes = compute_barotropic_modes(A, k=mode_num, which=eig_order,reference_period=reference_period)
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

   #print ('Select modes with a relevant amplitude')
   #Amax=np.nanmax(modes_2D)
   #print ('Max amplitude is:',Amax)
   #Th_Amp=Amax*Perc_Amp/100.0
   #print ('Amplitude threshold is:',Th_Amp)
   #print ('Done!')

   print ('Save the R modes')
   save_modes_to_netcdf(outfile_R, modes_2D, period, mask=mask)
   print ('Done!')

else:
   print ('Load the R modes')
   modes_2D, period, mask = load_modes_from_netcdf(outfile_R)
   print ('Done!')

print ('Plot the R modes amplitude')
print ('Print and plot only the modes with a relevant amplitude')
#for m in range(k):
#    if np.nanmax(modes_2D[m]) > Th_Amp:
#       counts = np.sum(modes_2D[m] > Th_Amp)
#       if counts > Counts_min: 
#          print ('Mode:',m,f'{period[m]:0.2f} h')
#          title    = f"Barotropic mode {m+1} - Period {period[m]:0.2f} h"
#          filename = f"/work/cmcc/ag15419/basin_modes_sa/mode_{m+1:02d}_{mode_num}.png"
#          filename_abs = f"/work/cmcc/ag15419/basin_modes_sa/mode_abs_{m+1:02d}_{mode_num}.png"
#          plot_mode(modes_2D[m], mask, title=title, filename=filename,filename_abs=filename_abs)
#print ('Done!')

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
            
    title = f"Barotropic mode {renum} - Period {this_period:.2f} h"
    filename = f"{work_dir}/mode_{renum:02d}_{mode_num}_{this_period:.2f}h.png"
    filename_abs = f"{work_dir}/mode_abs_{renum:02d}_{mode_num}_{this_period:.2f}h.png"
            
    plot_mode(modes_2D[m], mask, title=title, filename=filename, filename_abs=filename_abs)
    renum += 1


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
