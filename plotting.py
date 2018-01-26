import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import glob


def plot_heatmap(df, filename='heatmap.png', columns='config', index='dataset', values='positive_match_rate'):
    ''' Plot heatmap of average values on a grid of columns vs. index ''' 

    # produce df with given index and columns,  average values over all non-index and column values
    heatmap_data = df.groupby([columns, index])[values].mean()     
    heatmap_data = heatmap_data.reset_index()
    heatmap_data = heatmap_data.pivot(columns=columns, index=index, values=values)

    mean_values = heatmap_data.mean()
    col_sort_inds = np.argsort(mean_values)[::-1]
    heatmap_data = heatmap_data[col_sort_inds]  # reorder configs by average (across datasets) pos match rate
    data_sort_inds = np.argsort(heatmap_data[col_sort_inds[0]])  # sort datasets by performance of best config on that dataset
    heatmap_data = heatmap_data.reindex(heatmap_data.index[data_sort_inds])  

    plt.clf()
    sb.heatmap(heatmap_data, square=True)
    plt.savefig(filename)  
    plt.clf()
    
    # 'plots/heatmap_keep{0}_{1}.png'.format(n_keep, get_timestamp()))
    

def plot_results(trial_results=None, n_keep=None):

    if trial_results is None:
        files = glob.glob('trials/*.csv')
        most_recent = sorted(files)[-1]  # assumes timestamp file suffix
        df = pd.read_csv(most_recent)

    elif isinstance(trial_results, str):
        df = pd.read_csv(trial_results)
        
    else:
        assert isinstance(trial_results, pd.DataFrame)
        df = trial_results

    print('\npost-processing data')
    df['config'] = get_config_string_col(df)
    df['score_gap'] = df['avg_positive_score'] - df['avg_negative_score']
    df['-avg_negative_score'] = - df['avg_negative_score']


    df.groupby(['config', 'dataset'])



    config_score_map = df.groupby('config')['score_gap'].mean()     
    config_strings = np.array(list(config_score_map.keys()))
    config_scores = config_score_map.values
    sort_inds = np.argsort(config_scores)[::-1]
    # top_scores = config_scores[sort_inds][:n_top]
    # top_configs = config_strings[sort_inds][:n_top]
    # print('\n\ntop {0} configs, scores:\n{1}\n\n'.format(
    #     n_top,
    #     '\n'.join([str(x) for x in zip(top_configs, top_scores)])
    #     ))

    print('plotting config scores\n')

    df.sort_values(by=['score_gap'], ascending=False, inplace=True)

    sb.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(15, 15))

    order = config_strings[sort_inds]

    sb.set_color_codes("pastel")
    sb.barplot(x="score_gap", y="config", data=df, label="Score Gap", color="b", ci=None, order=order)

    sb.set_color_codes("muted")
    sb.barplot(x="avg_positive_score", y="config", data=df, label="Positive Score", color="b", ci=None, order=order)

    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(ylabel="Model Config", xlabel="Average Score")
    sb.despine(left=True, bottom=True)
    
    plt.savefig('plots/score_gap_{0}.png'.format(get_timestamp()))

    plt.clf()

    config_score_map = df.groupby('config')['positive_match_rate'].mean()     
    config_strings = np.array(list(config_score_map.keys()))
    config_scores = config_score_map.values
    sort_inds = np.argsort(config_scores)[::-1]
    order = config_strings[sort_inds]
    fig, ax = plt.subplots(figsize=(15, 15))
    sb.barplot(x="positive_match_rate", y="config", data=df, label="Positive Match Rate (keep {0})".format(n_keep), color="b", ci=None, order=order)    
    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(ylabel="Model Config", xlabel="Positive Match Rate (keep {0})".format(n_keep))
    sb.despine(left=True, bottom=True)

    plt.savefig('plots/positive_match_rate_keep{0}_{1}.png'.format(n_keep, get_timestamp()))


