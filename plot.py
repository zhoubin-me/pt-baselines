

import glob
import bokeh



import sys
import pandas as pd
from bokeh.io import output_file, show
from bokeh.layouts import gridplot, column
from bokeh.plotting import figure
from bokeh.palettes import *



def main():
    files = glob.glob('log/*/progress.txt')
    print(files)

    figs = {}
    colors = Category10[8]



    for idx, file in enumerate(files):
        with open(file, 'r') as f:
            data = pd.read_table(f)
            title = file.strip().split('/')[1]
            game = title.split('-')[1]


            if game not in figs:
                figs[game] = {}

            for col in data.columns:
                if 'Unnamed' not in col and 'TotalEnv' not in col:
                    if col in ['RemHrs', 'NumOfEp']:
                        continue

                    if col not in figs[game]:
                        figs[game][col] = figure(width=800, height=600, title=col)

                    fig = figs[game][col]
                    color = colors[idx%8]
                    fig.circle(data['TotalEnvInteracts'].values, data[col].values, legend=title, fill_color=color, color=color)
                    fig.legend.location = 'bottom_right'




    grids = []
    for g, fs in figs.items():
        output_file(f'log/{g}.html')
        p = column(list(fs.values()))
        show(p)

if __name__ == '__main__':
    main()
