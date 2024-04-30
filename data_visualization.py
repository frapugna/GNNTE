import matplotlib.pyplot as plt
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns

def compare_models_hist(data: pd.DataFrame | str, bin_criterion: str='a%', bins_name: str='Correct Label', out_pdf: str=None, font_scale: float=0.7) -> pd.DataFrame:
    """Function to plot an histogram to compare performances of different models depending on their range of error 

    Args:
        data (pd.DataFrame | str): data frame containing the results
        bin_criterion (str, optional): parameter to generate the 10 bins, must be with values in [0,1]. Defaults to 'a%'.
        bins_name (str, optional): name of the bins. Defaults to 'AE'.
    """
    if isinstance(data, str):
        data = pd.read_csv(data)

    ranges = f'{bins_name} Range' 
    new_data = {
        ranges:[],
        'Approach':[],
        'MAE':[]
    }
    for i in range(1, 11, 1):
        i /= 10
        prev = round(i-0.1, 2)
        t = data[data[bin_criterion] >= prev]
        if i == 1:
            t = t[t[bin_criterion] <= i]
        else:
            t = t[t[bin_criterion] < i]
        
        #curr =  f'{prev}_{i}'
        curr =  f'[{prev},\n{i}]'
        #curr =  f'{prev},{i}'
        new_data['Approach'].append('Armadillo Gittables')
        new_data[ranges].append(curr)
        try:
            new_data['MAE'].append(round(np.mean(t['AE_armadillo']),2))
        except: 
            new_data['MAE'].append(round(np.mean(t['armadillo_gittables_AE']),2))

        new_data['Approach'].append('Overlap Set Similarity')
        new_data[ranges].append(curr)
        try:
            new_data['MAE'].append(round(np.mean(t['AE_josie']),2))
        except:
            new_data['MAE'].append(round(np.mean(t['o_set_sim_AE']),2))

        new_data['Approach'].append('Jaccard Similarity')
        new_data[ranges].append(curr)
        try:
            new_data['MAE'].append(round(np.mean(t['AE_jsim']),2))
        except:
            new_data['MAE'].append(round(np.mean(t['jsim_AE']),2))
        
        try:
            new_data['Approach'].append('Armadillo Wikitables')
            new_data[ranges].append(curr)
            new_data['MAE'].append(round(np.mean(t['armadillo_wikitables_AE']),2))
        except:
            pass
    
    df = pd.DataFrame(new_data)
    sns.set_theme(font_scale=font_scale, style="whitegrid")
    sns.barplot(data=df, x=ranges, y='MAE', hue='Approach')

    if isinstance(out_pdf, str):
        plt.tight_layout()
        
        plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    return df


def plot_data_distribution(df_path: str | pd.DataFrame, label: str='a%', label_y: str='n_samples') -> None:
    """Given a labelled dataset print the data distribution of its samples

    Args:
        df_path (str | pd.DataFrame): path to the labelled dataframe or the dataframe.
        label (str, optional): label of the column to analyze. Defaults to 'a%'.
        label_y (str, optional): label of the y axis of the chart. Defaults to 'n_samples'.
    """
    if isinstance(df_path, str):
        data = pd.read_csv(df_path)
    else:
        data = df_path
    d = {}
    for i in range(1, 11, 1):
        i /= 10
        prev = round(i-0.1, 2)
        t = data[data[label] >= prev]
        t = t[t[label] < i]
        d[f'{prev}_{i}'] = t.shape[0]
    ##
    t = data[data[label] == 1]
    d['0.9_1.0']+=t.shape[0]
    ## 
    keys = list(d.keys())
    values = list(d.values())
    
    bar_width = 0.5
    
    # Create the bar plot
    plt.bar(keys, values, width=bar_width, color='grey')
    
    for i, v in enumerate(values):
        plt.text(i, v + 0.5, str(v), ha='center', va='bottom')
    
    plt.xticks(ha='center', fontsize=8)  # Ruota le etichette sull'asse x di 45 gradi
    plt.subplots_adjust(bottom=0.2) 
    
    # Adding labels and title
    plt.xlabel(f'{label} Range')
    plt.ylabel(label_y)

    # Show the plot
    plt.show()

def visualize_scatter_plot(exp_data_file: str | dict, logx: bool=True, logy: bool=True, out_pdf: str=None) -> None:
    """visualize embedding generation time on the y axis and table area on the x axis

    Args:
        exp_data_file (str | dict): path to a file containing the data related to a "embed_all_no_paral" test or the dictionary containing the actual data
        logx (bool, opt): if True, the x axis is in logscale. Defaults to True.
        logy (bool, opt): if True, the y axis is in logscale. Defaults to False.
    """
    if isinstance(exp_data_file, str):
        with open(exp_data_file, 'rb') as f:
            data = pickle.load(f)
    else:
        data = exp_data_file

    keys = list(data.keys())

    areas = [data[k]['area'] for k in keys]
    t_execs = [data[k]['t_tot'] for k in keys]

    x = areas
    y = t_execs

    # Definisci la figura e gli assi per lo scatterplot
    fig, (ax_scatter, ax_kde) = plt.subplots(2, 1, figsize=(8, 8), 
                                            gridspec_kw={'height_ratios': [3, 1]})

    # Disegna lo scatterplot
    ax_scatter.scatter(x, y, s=3, c='orange', alpha=0.7, edgecolors='black')

    sns.histplot(
    data=x, ax=ax_kde,
    label='KDE',
    fill=True, common_norm=False,
    alpha=.5, linewidth=0, color='grey'
    )
    # Imposta i titoli e le etichette degli assi per lo scatterplot
    #ax_scatter.set_title('Embedding generation time with increasing table areas')
    ax_scatter.set_ylabel('Total Embedding Time (s)')
    
    if logx:
        ax_kde.set_xscale('log')    
        ax_scatter.set_xscale('log')
    if logy:
        ax_scatter.set_yscale('log')
    ax_kde.set_yscale('log')
    
    # Imposta le etichette degli assi per il KDE plot
    ax_kde.set_xlabel('Table Area')
    ax_kde.set_ylabel('Number Of Samples')

    # Visualizza il grafico
    plt.tight_layout()
    
    if isinstance(out_pdf, str):
        plt.savefig(out_pdf, format="pdf", bbox_inches="tight")

    plt.show()


def plot_dict(d: dict, xlabel: str, ylabel: str) -> None:
    l=[ [k,v] for k,v in d.items()]
    df_occurrencies = pd.DataFrame(l).sort_values(0)
    ax = df_occurrencies.plot(x=0, y=1, kind='bar')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.legend().remove()
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom')

def show_samples_distribution(df:pd.DataFrame, granularity:float=0.1, index: int | str=2, label_x: str='Overlap Ratio', label_y: str='n_samples')->dict:
    """The dataset is divided in bins based on sample's table overlap, a bar diagram is displayed to show visually the data distribution

    Args:
        df (pd.DataFrame): the dataframe to analyze
        granularity (float, optional): the size of the bins. Defaults to 0.1.

    Returns:
        dict: contains the count of elements in every bin
    """
    d = {}
    if isinstance(index, str):
        for i in tqdm(df[index]):#tqdm(df['table_overlap']):
                n = i//granularity/10
                if i == 1:
                    n = 1
                try:
                    d[n]+=1
                except:
                    d[n]=1    
    else:
        for i in tqdm(df.iloc[:,index]):#tqdm(df['table_overlap']):
            n = i//granularity/10
            if i == 1:
                n = 1
            try:
                d[n]+=1
            except:
                d[n]=1

    l=[ [k,v] for k,v in d.items()]

    #l={ [k,v] for k,v in d.items()}

    df_occurrencies = pd.DataFrame(l).sort_values(0)
    ax = df_occurrencies.plot(x=0, y=1, kind='bar')
    plt.xlabel(f'{label_x} Range')
    plt.ylabel(label_y)
    ax.legend().remove()
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom')
    return d

if __name__ == '__main__':
    # git_train = pd.read_csv('/home/francesco.pugnaloni/GNNTE/Datasets/2_WikiTables/1M_wikitables_disjointed/455252_52350_52530/test.csv')
    # show_samples_distribution(git_train)
    dd = pd.read_csv('/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/labelled/Training_data_Gittables/train_raw.csv')
    plot_data_distribution(dd, 'a%')