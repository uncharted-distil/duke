import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import glob
from inflection import titleize


# filename='plots/heatmap_keep{0}_{1}.png'.format(n_keep, get_timestamp()))
def plot_heatmap(df, image_path='plots/heatmap.png', columns='config', index='dataset', values='positive_match_rate', clear_fig=True):
    ''' Plot heatmap of average values on a grid of columns vs. index ''' 
    
    # produce df with given index and columns,  average values over all non-index and column values
    data = df.groupby([columns, index])[values].mean()     
    data = data.reset_index()
    data = data.pivot(columns=columns, index=index, values=values)

    # take mean over datasets
    mean_values = data.mean()  
    # reorder configs by average (across datasets) pos match rate
    col_sort_inds = np.argsort(mean_values)[::-1]  
    data = data[data.columns[col_sort_inds]]  
    # sort datasets by performance of best config on that dataset
    row_sort_inds = np.argsort(data[data.columns[0]])  
    data = data.reindex(data.index[row_sort_inds])  

    if clear_fig: 
        plt.clf()
    sb.heatmap(data, square=True)
    plt.savefig(image_path)
    

def plot_horizontal_barplot(df, image_path='barplot.png', group_by='config', values='score_gap', clear_fig=True, colors='b'):
    
    gb = df.groupby(group_by)
    
    if clear_fig: 
        plt.clf()
    sb.set(style="whitegrid")
    fig, ax = plt.subplots() # figsize=(15, 15))
    
    if not isinstance(values, list):
        values = [values]
    
    if not isinstance(colors, list):
        colors = [colors]

    assert len(colors) == len(values)

    for color, vals in zip(colors, values):
        data = gb[vals].mean()
        grp_names = np.array(data.keys())
        sort_inds = np.argsort(data.values)[::-1]
        order = grp_names[sort_inds]
        sb.barplot(x=vals, y=group_by, data=df, label=titleize(vals), ci=None, order=order, color=color)

    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(ylabel=titleize(group_by))  # , xlabel="Average Score")
    sb.despine(left=True, bottom=True)
    
    plt.savefig(image_path)
    