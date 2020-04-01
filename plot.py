

import glob
import bokeh



import sys
import pandas as pd
from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.plotting import figure
from bokeh.palettes import *

output_file('log/plot.html')

if __name__ == '__main__':
    files = glob.glob('log/*/result.txt')

    fig1 = figure(width=800, height=600, title='Mean Rewards')
    fig2 = figure(width=800, height=600, title='Mean Q')
    colors = Category10[8]

    for idx, file in enumerate(files):
        with open(file, 'r') as f:
            data = pd.read_table(f)
            print(data)
            title = file.strip().split('/')[1]
            fig1.circle(data['Epoch'].values, data[' Mean_Reward'].values, legend=title, fill_color=colors[idx])
            fig1.line(data['Epoch'].values, data[' Mean_Reward'].values, legend=title, line_color=colors[idx])
            fig2.circle(data['Epoch'].values, data[' Mean_Q'].values, legend=title, fill_color=colors[idx])
            fig2.line(data['Epoch'].values, data[' Mean_Q'].values, legend=title, line_color=colors[idx])


    print(files)
    fig1.legend.location = "top_left"
    fig2.legend.location = "top_left"
    p = gridplot([[fig1, fig2]])

