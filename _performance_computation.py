import pandas as pd
import numpy as np
from data_visualization import *
import matplotlib.pyplot as plt

def show_mae_per_bin(results_path: str | pd.DataFrame, granularity: float=0.1, plot: bool=True, box: bool=False) -> None:
    data = pd.read_csv(results_path)
    d = {}
    box_plot = {}
    for i in range(1, 11, 1):
        i /= 10
        prev = round(i-0.1, 2)
        t = data[data['overlap_true'] >= prev]
        t = t[t['overlap_true'] < i]
        print(f'Bin: {i}        n_samples:{len(t)}      MAE:{np.mean(t["AE"])}')
        d[f'{prev}_{i}'] = round(np.mean(t['AE']),2)
        box_plot[f'{prev}_{i}'] = t['AE']
    if plot:
        if box:
            df = pd.DataFrame(box_plot)
            plt.figure(figsize=(8, 6))
            df.boxplot()
            plt.title('Boxplot of DataFrame Columns')
            plt.xlabel('Correct Label Range')
            plt.ylabel('MAE')
            plt.show()
        else:
            plot_dict(d,'Correct Label Range','MAE')

if __name__ == '__main__':
    show_mae_per_bin('/home/francesco.pugnaloni/GNNTE/Datasets/CoreEvaluationDatasets/gittables_labelled_no_rep_train_test/results/train_with_predicted_and_AE_wiki_450k.csv')
    