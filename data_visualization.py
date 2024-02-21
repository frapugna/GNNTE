import matplotlib.pyplot as plt
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np

def plot_data_distribution(df_path: str, label: str, label_y: str='n_samples') -> None:
    data = pd.read_csv(df_path)
    d = {}
    for i in range(1, 11, 1):
        i /= 10
        prev = round(i-0.1, 2)
        t = data[data[label] >= prev]
        t = t[t[label] < i]
        d[f'{prev}_{i}'] = t.shape[0]
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

def visualize_scatter_plot(exp_data_file: str | dict, logx: bool=True, logy: bool=False) -> None:
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

    plt.scatter(areas, t_execs, s=3, c='orange', alpha=0.7, edgecolors='black')

    plt.title('Embedding generation time with increasing table areas')
    plt.xlabel('Table Area')
    plt.ylabel('Total Embedding Time (ms)')
    plt.grid(True)
    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')
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
    git_train = pd.read_csv('/home/francesco.pugnaloni/GNNTE/Datasets/2_WikiTables/1M_wikitables_disjointed/455252_52350_52530/test.csv')
    show_samples_distribution(git_train)