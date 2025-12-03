import sys, pickle, os, warnings
warnings.simplefilter('ignore')
import pandas as pd
import numpy as np
import cmlreaders as cml
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
pd.set_option("display.max_columns", None)
import logging
import glob
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pandas_to_pybeh import *


def compute_recall_rate(data):
    word_evs = data[data['type'] == 'WORD']
    return word_evs['recalled'].sum() / len(word_evs)


def compute_first_recall(data, list_len):
    rec_evs = data[data['type'] == 'REC_WORD']
    rec_evs['pos'] = rec_evs.groupby(['session', 'trial']).cumcount()
    first_recall_df = rec_evs.query('pos == 0 and serialpos >= 0')
    first_recall_df = first_recall_df.groupby(
        ['serialpos']).agg(
        {'recalled': 'count'}).reindex(range(1, list_len+1), fill_value=0)
    n_lists = first_recall_df['recalled'].sum()
    return first_recall_df['recalled'].to_numpy(dtype=float) / n_lists


def compute_lag_crp_single_subject_array(data, list_len):
    center = list_len - 1
    min_lag = -center
    max_lag = center + 1
    actual = {lag: 0 for lag in range(min_lag, max_lag)}
    possible = {lag: 0 for lag in range(min_lag, max_lag)}
    for session_id, session_data in data.groupby('session'):
        recalls = session_data[session_data.type == 'REC_WORD']
        words = session_data[session_data.type == 'WORD']
        if recalls.empty or words.empty:
            print(f"session {session_id} has no events")
            continue
        recalls = recalls[(recalls['trial'] != -999)]
        word_to_pos = dict(zip(words['item'], words['serialpos']))
        for trial in recalls['trial'].unique():
            trial_words = words[words['trial'] == trial]['item'].tolist()
            trial_recalls = (recalls[recalls['trial'] == trial]
                             .sort_values('rectime')
                             .drop_duplicates('item'))
            
            if len(trial_recalls) < 2:
                print(f"session {session_id}, trial {trial} doesn't have enough events")
                continue
            trial_recalls = trial_recalls[trial_recalls['item'].isin(trial_words)]
            recall_pos = [word_to_pos[w] for w in trial_recalls['item']]
            for i, cur in enumerate(recall_pos[:-1]):
                lag = recall_pos[i+1] - cur
                if min_lag <= lag <= max_lag and lag != 0:
                    actual[lag] += 1
                for pos in set(range(1, list_len+1)) - set(recall_pos[:i+1]):
                    pl = pos - cur
                    if min_lag <= pl <= max_lag and pl != 0:
                        possible[pl] += 1

    full_len = 2*list_len - 1
    crp = np.full(full_len, np.nan)
    center = list_len - 1
    for lag in range(min_lag, max_lag):
        idx = center + lag
        if 0 <= idx < full_len:
            crp[idx] = (actual[lag] / possible[lag]) if possible[lag] > 0 else np.nan
    crp[center] = 0.0
    return crp


def get_recall_prob_per_subject(in_df):
    pres = in_df.type=='WORD'
    recall_by_sub = in_df[pres].groupby('subject')\
        .agg({'recalled':'mean'})\
        .rename(columns={"recalled":"Recall_Probability"}
    )
    recall_by_sub = recall_by_sub.reset_index()
    recall_by_sub = recall_by_sub.sort_values(by="Recall_Probability", ascending=True)
    return recall_by_sub


def plot_subject_group_pointplot(in_df, subjects, x_col, y_col, graph_configs=None):
    """
    Generalized function to plot individual-subject pointplots and group-level averages.
    
    Parameters
    ----------
    in_df : pd.DataFrame
        Input dataframe containing subject, x, and y columns.
    subjects : list
        List of subject identifiers to plot individually.
    x_col : str
        Column name for the x-axis variable (e.g., 'serialpos' or 'lag').
    y_col : str
        Column name for the y-axis variable (e.g., 'prob' or 'Recall_Probability').
    graph_configs : dict
        Dictionary of optional parameters:
            {
                'size': (width, height),
                'color': base_color (default "#79B"),
                'ci': confidence interval (default 68),
                'capsize': error bar cap size (default .3),
                'ylim': (min, max),
                'title': str,
                'xlabel': str,
                'ylabel': str
            }
    """
    
    # ---- Default settings ----
    defaults = {
        'size': (8, 6),
        'color': "#79B",
        'ci': 68,
        'capsize': .3,
        'ylim': (0, 1.1),
        'title': None,
        'xlabel': x_col,
        'ylabel': y_col
    }
    if graph_configs:
        defaults.update(graph_configs)
    cfg = defaults
    
    # ---- Initialize figure ----
    fig, ax = plt.subplots(figsize=cfg['size'])
    colors = sns.light_palette(cfg['color'], len(subjects))
    
    # ---- Subject-level plots ----
    for i, sub in enumerate(subjects):
        sub_df = in_df.query('subject == @sub')
        sns.pointplot(
            data=sub_df,
            x=x_col,
            y=y_col,
            ci=None,
            ax=ax,
            color=colors[i],
            linestyles="-"
        )
    
    # ---- Group-level average ----
    sns.pointplot(
        data=in_df,
        x=x_col,
        y=y_col,
        ci=cfg['ci'],
        capsize=cfg['capsize'],
        ax=ax,
        color="k"
    )
    
    # ---- Style ----
    ax.set_xlabel(cfg['xlabel'], fontsize=22)
    ax.set_ylabel(cfg['ylabel'], fontsize=22)
    ax.tick_params(labelsize=18)
    ax.yaxis.grid()
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    
    if cfg['title']:
        ax.set_title(cfg['title'], fontsize=22)
    if cfg['ylim']:
        ax.set_ylim(cfg['ylim'])
    
    plt.tight_layout()
    plt.show()
    

def get_spc(in_df):
    spc_df = in_df.query("type=='WORD'").groupby(
        ['subject', 'session', 'serialpos']
    ).agg({'recalled':np.nanmean}).reset_index()

    return spc_df.groupby(["subject","serialpos"]).agg({"recalled":"mean"}).reset_index()


def get_prob_first_recall(in_df):
    recword = in_df.query('type=="REC_WORD"')
    recword['pos'] = recword.groupby(['subject', 'session', 'trial']).cumcount()
    first_recall_df = recword.query('pos == 0 and serialpos >= 0')
    first_recall_df = first_recall_df.groupby(
        ['subject', 'serialpos']).agg(
        {'recalled': 'count'}).unstack(fill_value=0).stack().reset_index()
    first_recall_df['n_lists'] = first_recall_df.groupby(['subject'])['recalled'].transform('sum')
    first_recall_df['prob'] = first_recall_df['recalled'] / first_recall_df['n_lists']
    return first_recall_df


def compute_temporal_clustering(in_df):
    """Compute temporal clustering (TC) per subject."""
    temporal_clustering_df = (
        in_df.groupby(['subject'])
        .apply(
            pd_temp_fact,
            itemno_column='itemno',
            list_index=['subject', 'session', 'trial'],
            skip_first_n=0
        )
        .rename('TC')
        .reset_index()
        .sort_values(by='TC', ascending=True)
    )
    return temporal_clustering_df


def compute_trial_error(in_df):
    """Compute absolute error per trial (valuerecall - actualvalue)."""
    word_evs = in_df[in_df['type'] == 'WORD']
    error_by_trial = (
        word_evs.groupby(['subject', 'session', 'trial'], as_index=False)
        .agg(
            actualvalue=('actualvalue', 'first'),
            valuerecall=('valuerecall', 'first'),
            item_count=('item', 'size'),
            storepointtype=('storepointtype', 'first')
        )
    )
    error_by_trial['abs_error'] = (error_by_trial['valuerecall'] - error_by_trial['actualvalue']).abs()
    return error_by_trial


def merge_predictors(error_by_trial, temporal_clustering_df, recall_by_sub):
    """Merge error, temporal clustering, and recall probability info."""
    merged = error_by_trial.merge(temporal_clustering_df[['subject', 'TC']], on='subject')
    merged = merged.merge(recall_by_sub[['subject', 'Recall_Probability']], on='subject')
    return merged

import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_models_and_get_residuals(merged):
    """Fit mixed model and compute residual-adjusted error."""
    # Mixed effects model summary
    model = smf.mixedlm(
        "abs_error ~ storepointtype + TC + Recall_Probability",
        data=merged,
        groups=merged["subject"]
    )
    result = model.fit(reml=False)
    print(result.summary())

    # Fit simple OLS for recall correction
    recall_model = sm.OLS.from_formula("abs_error ~ Recall_Probability", data=merged).fit()
    merged['error_adj'] = recall_model.resid
    return merged


def compute_error_diff_adj(merged):
    """Compute adjusted Temporal–Random error difference per subject."""
    error_by_trial_adj = (
        merged.groupby(['subject', 'storepointtype'], as_index=False)
        .agg(error_diff=('error_adj', 'mean'))
    )

    error_wide_adj = (
        error_by_trial_adj
        .pivot(index='subject', columns='storepointtype', values='error_diff')
        .reset_index()
        .dropna(subset=['Temporal', 'random'])
    )

    error_wide_adj['error_diff_adj'] = error_wide_adj['Temporal'] - error_wide_adj['random']
    return error_wide_adj[['subject', 'error_diff_adj']]


def get_mixed_model_fitted_error(in_df, recall_by_sub):
    """
    Full pipeline:
      1. Compute temporal clustering (TC)
      2. Compute trial-level absolute error
      3. Merge with recall metrics
      4. Fit mixed model and compute recall-adjusted residuals
      5. Compute adjusted Temporal–Random error difference

    Returns:
      DataFrame with columns ['subject', 'TC', 'error_diff_adj']
    """
    # Step 1
    temporal_clustering_df = compute_temporal_clustering(in_df)

    # Step 2
    error_by_trial = compute_trial_error(in_df)

    # Step 3
    merged = merge_predictors(error_by_trial, temporal_clustering_df, recall_by_sub)

    # Step 4
    merged = fit_models_and_get_residuals(merged)

    # Step 5
    error_wide_adj = compute_error_diff_adj(merged)

    # Combine final results
    return pd.merge(temporal_clustering_df, error_wide_adj, on='subject')


def plot_error_diff_vs_binned_cluster_score(*dfs, labels=None):
    """
    Plot |Error| Difference vs. binned Temporal Clustering Score for one or more dataframes.
    
    Args:
        *dfs: One or more pandas DataFrames, each containing columns 'TC' and 'error_diff_adj'.
        labels (list, optional): List of labels for each dataframe for legend entries.
    """
    if labels and len(labels) != len(dfs):
        raise ValueError("Number of labels must match number of dataframes.")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i, df in enumerate(dfs):
        df = df.copy()
        
        # Bin TC into low/medium/high
        edges = np.linspace(df['TC'].min(), df['TC'].max(), 4)
        df['TC_bin'] = pd.cut(
            df['TC'],
            bins=edges,
            labels=['low', 'medium', 'high'],
            include_lowest=True,
            right=True
        )

        cats = pd.CategoricalDtype(categories=['low','medium','high'], ordered=True)
        df['TC_bin'] = df['TC_bin'].astype(cats)

        # Compute summary stats
        summ = df.groupby('TC_bin', observed=True)['error_diff_adj'].agg(['mean', 'count', 'std']).reset_index()
        summ['se'] = summ['std'] / np.sqrt(summ['count'])
        summ['x'] = summ['TC_bin'].cat.codes

        label = labels[i] if labels else f'Dataset {i+1}'
        ax.errorbar(
            summ['x'],
            summ['mean'],
            yerr=summ['se'],
            fmt='o-',
            capsize=3,
            label=label
        )

    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['low','medium','high'])
    ax.set_xlabel('Temporal Clustering Score', fontsize=22)
    ax.set_ylabel('|Error| Difference', fontsize=22)
    ax.tick_params(labelsize=18)
    ax.yaxis.grid(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if len(dfs) > 1:
        ax.legend(fontsize=16)
        
    plt.tight_layout()
    plt.show()