import glob
import bokeh


import sys
import pandas as pd
from bokeh.io import output_file, show, save
from bokeh.layouts import gridplot, column
from bokeh.plotting import figure
from bokeh.palettes import *



def plot(stage='train'):

    if stage == 'train':
        files = glob.glob('log/*/progress.txt')
    else:
        files = glob.glob('log/*/val.txt')

    print(files)

    figs = {}
    colors = Category10[8]
    algos = []

    for idx, file in enumerate(files):
        with open(file, 'r') as f:
            data = pd.read_table(f)
            title = file.strip().split('/')[1]
            algo, game, seed = title.split('-')
            algo_seed = algo + seed
            if algo_seed not in algos:
                algos.append(algo_seed)
            color_idx = algos.index(algo_seed)
            color = colors[color_idx]

            if game not in figs:
                figs[game] = {}

            for col in data.columns:
                if 'Unnamed' not in col:
                    if col in ['RemHrs', 'NumOfEp', 'Epoch', 'TotalEnvInteracts', 'Steps']:
                        continue
                    key = 'TotalEnvInteracts' if stage == 'train' else 'Steps'

                    if col not in figs[game]:
                        figs[game][col] = figure(width=800, height=600, title=col)

                    fig = figs[game][col]
                    fig.circle(data[key].values, data[col].values, legend=title, fill_color=color, color=color)
                    fig.legend.location = 'bottom_right'

    grids = []
    for g, fs in figs.items():
        grids.append(list(fs.values()))
        # output_file(f'log/{g}.html')
    output_file(f'log/{stage}.html')
    p = gridplot(grids)
    save(p)


if __name__ == '__main__':
    plot('train')
    plot('test')
