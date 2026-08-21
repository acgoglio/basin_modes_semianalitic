# basin_modes_semianalitic
#
# semi-analitical_med_modes.py -> Most updated version of the code 
# 
# other_versions/analitical_med_modes.py -> Most updated version with f=0
# other_versions/analitical_med_modes_f.py -> First version with rotation (f!=0)
# other_versions/all_code_newmesh.py  -> works with barotropic mesh_mask.nc (but you need to create it!)
# other_versions/all_code.py          -> works with baroclinic mesh_mask.nc (but does not include the check on e3t-bathy.. ) 
# old/analitical_med_modes_f_bdycond.py -> version with f!=0 and the bdy condition imposed inside the K matrix


########
semi-analitical_med_mode_original.py -> uguale alla vecchia versione completa
semi-analitical_med_modes_paper.py -> versione creata dalla semi-analitical_med_mode_original.py correggendo qualche bug (calcolo PE/KE)
semi-analitical_med_mode_2runs.py -> estensione di semi-analitical_med_modes_paper.py fino a 40h
semi-analitical_med_mode_3runs.py -> estensione di semi-analitical_med_mode_2runs.py che calcola anche autonomamente i modi oscillatori da f=0
                                     (ATTENZIONE: questo e' ancora da sistemare perche' vanno gestiti i casi in cui f=0 genera uno scift di frequenza )

