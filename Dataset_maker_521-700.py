#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 12:37:47 2022

@author: jeffo
"""

import numpy as np
import pandas as pd
import ants
import antspynet

Deskar_Label_dict=dict({0:'background',
    4: 'left_lateral_ventricle',
    5: 'left_inferior_lateral_ventricle',
    6: 'left_cerebellem_exterior',
    7: 'left_cerebellum_white_matter',
    10: 'left_thalamus_proper',
    11: 'left_caudate',
    12: 'left_putamen',
    13: 'left_pallidium',
    14: '3rd_ventricle',
    15: '4th_ventricle',
    16: 'brain_stem',
    17: 'left_hippocampus',
    18: 'left_amygdala',
    24: 'CSF',
    25: 'left_lesion',
    26: 'left_accumbens_area',
    28: 'left_ventral_DC',
    30: 'left_vessel',
    43: 'right_lateral_ventricle',
    44: 'right_inferior_lateral_ventricle',
    45: 'right_cerebellum_exterior',
    46: 'right_cerebellum_white_matter',
    49: 'right_thalamus_proper',
    50: 'right_caudate',
    51: 'right_putamen',
    52: 'right_palladium',
    53: 'right_hippocampus',
    54: 'right_amygdala',
    57: 'right_lesion',
    58: 'right_accumbens_area',
    60: 'right_ventral_DC',
    62: 'right_vessel',
    72: '5th_ventricle',
    85: 'optic_chasm',
    91: 'left_basal_forebrain',
    92: 'right_basal_forebrain',
    630: 'cerebellar_vermal_lobules_I-V',
    631: 'cerebellar_vermal_lobules_VI-VII',
    632: 'cerebellar_vermal_lobules_VIII-X',
    1002: 'left_caudal_anterior_cingulate',
    1003: 'left_caudal_middle_frontal',
    1005: 'left_cuneus',
    1006: 'left_entorhinal',
    1007: 'left_fusiform',
    1008: 'left_inferior_parietal',
    1009: 'left_inferior_temporal',
    1010: 'left_isthmus_cingulate',
     1011: 'left_lateral_occipital',
     1012: 'left_lateral_orbitofrontal',
     1013: 'left_lingual',
     1014: 'left_medial_orbitofrontal',
     1015: 'left_middle_temporal',
     1016: 'left_parahippocampal',
     1017: 'left_paracentral',
     1018: 'left_pars_opercularis',
     1019: 'left_pars_orbitalis',
     1020: 'left_pars_triangularis',
     1021: 'left_pericalcarine',
     1022: 'left_postcentral',
     1023: 'left_posterior_cingulate',
     1024: 'left_precentral',
     1025: 'left_precuneus',
     1026: 'left_rostral_anterior_cingulate',
     1027: 'left_rostral_middle_frontal',
     1028: 'left_superior_frontal',
     1029: 'left_superior_parietal',
     1030: 'left_superior_temporal',
     1031: 'left_supramarginal',
     1034: 'left_transverse_temporal',
     1035: 'left_insula',
     2002: 'right_caudal_anterior_cingulate',
     2003: 'right_caudal_middle_frontal',
     2005: 'right_cuneus',
     2006: 'right_entorhinal',
     2007: 'right_fusiform',
     2008: 'right_inferior_parietal',
     2009: 'right_inferior_temporal',
     2010: 'right_isthmus_cingulate',
     2011: 'right_lateral_occipital',
     2012: 'right_lateral_orbitofrontal',
     2013: 'right_lingual',
     2014: 'right_medial_orbitofrontal',
     2015: 'right_middle temporal',
     2016: 'right_parahippocampal',
     2017: 'right_paracentral',
     2018: 'right_pars_opercularis',
     2019: 'right_pars_orbitalis',
     2020: 'right_pars_triangularis',
     2021: 'right_pericalcarine',
     2022: 'right_postcentral',
     2023: 'right_posterior_cingulate',
     2024: 'right_precentral',
     2025: 'right_precuneus',
     2026: 'right_rostral_anterior_cingulate',
     2027: 'right_rostral_middle_frontal',
     2028: 'right_superior_frontal',
     2029: 'right_superior_parietal',
     2030: 'right_superior_temporal',
     2031: 'right_supramarginal',
     2034: 'right_transverse_temporal',
     2035: 'right_insula'})
data=pd.read_pickle('/home/jott2/brainlabs/ALL_DATA.pkl')
for i in range(521,700):
    print(f'Starting brian_{i}')
    brain  = ants.image_read(data['input_file_path'][i],3)
    print('Parcellating Brain')
    #There may be a better way to create this training data
    dkt=antspynet.desikan_killiany_tourville_labeling(brain,do_preprocessing=True,verbose=False) #turn on verbose to see output
    print('Pulling mask')
    img_target = ants.image_read(data['target_file_path'][i],3)
    img_mask_grey=ants.threshold_image(img_target,2,2,1,0) #Get grey matter mask
    print('Propagating')
    final_prop=ants.iMath(img_mask_grey,"PropagateLabelsThroughMask",img_mask_grey *dkt)
    print('Making Tabular Data')
    tab_data=ants.label_stats(img_mask_grey,final_prop)
    tab_data=tab_data.reset_index(drop=True)
    print('Adding Deskar Labels to Tabular Data')
    tab_data['Label']=" "
    final_name=data['filenames'][i]
    for j in range(len(tab_data)):
        name=Deskar_Label_dict[tab_data.iloc[j]['LabelValue']]
        tab_data['Label'][j]=name
    print('Saving Parcellation')
    np.save(f'{final_name}_parcel_{i}',final_prop.numpy())
    print(f'Saved {final_name}_parcel_{i}')
    tab_data.to_pickle(f'{final_name}_tab_data_{i}.pkl')
    print(f'Saved {final_name}_tab_data_{i}')