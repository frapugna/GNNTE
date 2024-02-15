import matplotlib.pyplot as plt
import pickle
import pandas as pd
from tqdm import tqdm

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

def show_samples_distribution(df:pd.DataFrame, granularity:float=0.1)->dict:
    """The dataset is divided in bins based on sample's table overlap, a bar diagram is displayed to show visually the data distribution

    Args:
        df (pd.DataFrame): the dataframe to analyze
        granularity (float, optional): the size of the bins. Defaults to 0.1.

    Returns:
        dict: contains the count of elements in every bin
    """
    d = {}
    for i in tqdm(df.iloc[:,2]):#tqdm(df['table_overlap']):
        n = i//granularity/10
        if i == 1:
            n = 1
        try:
            d[n]+=1
        except:
            d[n]=1
    l=[ [k,v] for k,v in d.items()]
    df_occurrencies = pd.DataFrame(l).sort_values(0)
    ax = df_occurrencies.plot(x=0, y=1, kind='bar')
    plt.xlabel('Overlap Ratio')
    plt.ylabel('n_samples')
    ax.legend().remove()
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom')
    return d

if __name__ == '__main__':
    git_train = pd.read_csv('/home/francesco.pugnaloni/GNNTE/Datasets/CoreEvaluationDatasets/gittables_labelled_no_rep_train_test/train_stats_cleaned.csv')
    show_samples_distribution(git_train)