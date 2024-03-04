import matplotlib.pyplot as plt
import pandas as pd

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

if __name__ == '__main__':
    plot_data_distribution('/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/labelled/labelled_200k_groups/train_stats_200000.csv')