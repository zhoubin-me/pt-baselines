import glob
import bokeh


import sys
import pandas as pd
from bokeh.io import output_file, show, save
from bokeh.layouts import gridplot, column
from bokeh.plotting import figure
from bokeh.palettes import *
from scipy.signal import savgol_filter



def plot(stage='train', smooth=True):

    if stage == 'train':
        files = glob.glob('log/*/progress.txt')
    else:
        files = glob.glob('log/*/val.txt')

    sorted(files)

    figs = {}
    colors = Category20c[20]
    algos = []

    for idx, fname in enumerate(files):
        if 'SAC' in fname or 'A2C' in fname:
            continue

        with open(fname, 'r') as f:
            try:
                data = pd.read_table(f, sep='\t\t', engine='python')
            except:
                continue
            print('Plotting ', fname)
            title = fname.strip().split('/')[1]
            algo, game, seed = title.split('-')
            if algo not in algos:
                algos.append(algo)

            algo_index = algos.index(algo)
            color = colors[algo_index * 4 + int(seed)]

            if game not in figs:
                figs[game] = {}

            for col in data.columns:
                if 'Unnamed' not in col:
                    if col in ['RemHrs', 'NumOfEp', 'Epoch', 'TotalEnvInteracts', 'Steps']:
                        continue
                    key = 'TotalEnvInteracts' if stage == 'train' else 'Steps'

                    tooltips = [
                        ('index', '$index'),
                        (key, '$x'),
                        (col, '$y'),
                        ('Title', title)
                    ]




                    if col not in figs[game]:
                        figs[game][col] = figure(width=800, height=600, title=col, tools='pan, wheel_zoom, reset') # , tooltips=tooltips)

                    fig = figs[game][col]
                    # 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                    if smooth:
                        if len(data[col].values) < 100:
                            continue
                        y = savgol_filter(data[col].values, 51, 2, mode='nearest')
                    else:
                        y = data[col].values

                    fig.line(data[key].values, y, legend=title, line_color=color, line_width=3)
                    fig.legend.location = 'bottom_right'

    grids = []
    for g, fs in figs.items():
        grids.append(list(fs.values()))
    output_file(f'{stage}.html')
    p = gridplot(grids)
    save(p)


if __name__ == '__main__':
    plot('train')
    plot('test', False)
