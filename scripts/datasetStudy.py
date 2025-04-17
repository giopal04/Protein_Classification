import pandas as pd
import pyvista as pv
import numpy as np
import os

# Multiple function here depends on thi variable
root = '/mnt/dataset/shrec-2025-protein-classification/v2-20250331'

def possible_disconnected_mesh(df, cls, get_names=False):
    tmp_df = df[ df['class_id'] == cls]
    cls_mesh = len(tmp_df)

    max_points = tmp_df['number_of_points'].max()
    cut_off = max_points * 0.5
    df_defect = tmp_df[ tmp_df['number_of_points'] < cut_off]

    if get_names:
        return (len(df_defect), cls_mesh), df_defect
    else:
        return (len(df_defect), cls_mesh)
    
def cls_distribution(df, column_name='class_id'):
    dist = {}
    classes = list(np.sort(df[column_name].unique()))

    for cls in classes:
        df_tmp = df[df[column_name] == cls]
        dist[cls] = len(list(df_tmp[column_name]))
    
    return dist

def max_of_dist(dist):
    max = 0

    for key in dist:
        if dist[key] >= max:
            max = dist[key]
    
    return max

def inspect_distribution(dist, l_lim=0, u_lim=None):
    if u_lim == None:
        u_lim = max_of_dist(dist)
    
    count = 0
    for cls in dist:
        if dist[cls] >= l_lim and dist[cls] <= u_lim:
            count += 1

    print(f'Classes which have between {l_lim} and {u_lim} element: {count}/{len(dist)}')

def number_of_point_filter(df, cut_off):
    return df[ df['number_of_points'] >= cut_off ]

def number_of_class_filter(df, l_cut_off, u_cut_off=None):
    cls_dist = cls_distribution(df)

    if u_cut_off is None:
        u_cut_off = max_of_dist(cls_dist)

    tmp_df = df
    for key in cls_dist:
        if cls_dist[key] < l_cut_off or cls_dist[key] > u_cut_off:
            tmp_df = tmp_df[ tmp_df['class_id'] != key ]
    
    return tmp_df

def print_dist(dist, idx):
    try:
        output = dist[idx]
    except KeyError:
        output = 0
    
    return output

def visualize_mesh(df, cls, idx=0):
    assert 'class_id' in df.columns, '"class_id" is not a column of the dataframe'
    assert 'protein_id' in df.columns, '"protein_id" is not a column of the dataframe'

    if cls < df['class_id'].min() or cls > df['class_id'].max():
        raise UserWarning(f'There is no {cls} class')
    
    column = list(df[df['class_id'] == cls]['protein_id'])
    
    try:
        protein_name = column[idx]
    except:
        protein_name = column[-1]
    protein_name += '.vtk'

    mesh = pv.read(os.path.join(root, 'train', protein_name))
    print(mesh)
    mesh.plot()

def create_dataframe(df, class_ids, number_of_proteins=None):
    subset_dfs = []
    for cls in class_ids:
        class_df = df[df['class_id'] == cls]
        if number_of_proteins == None:
            print(f'Using all data for the class {cls}')
            subset = class_df.copy()
        elif len(class_df) < number_of_proteins:
            print(f"Warning: Class {cls} has only {len(class_df)} proteins, but {number_of_proteins} were requested. Using all available proteins.")
            subset = class_df.copy() 
        else:
            subset = class_df.sample(n=number_of_proteins).copy()
        subset_dfs.append(subset)

    dataset = pd.concat(subset_dfs)
    return dataset.sample(frac=1, ignore_index=True)

