import numpy as np
from flask import Flask, render_template
from flask_charts import GoogleCharts, Chart
import glob
import pandas as pd

app = Flask(__name__)
charts = GoogleCharts(app)

@app.route("/")
def index():


    line_chart = Chart("ScatterChart", "data")
    line_chart.options = {
                            "title": "X-Y",
                            "width": 800,
                            "height": 600
                          }
    line_chart.data.add_column("number", "Steps")
    line_chart.data.add_column("number", "Cosine")
    line_chart.data.add_column("number", "Sine")

    for i in range(9000):
        line_chart.data.add_row([i*0.001, np.cos(i*0.001), np.sin(i*0.001)])

    return render_template("index.html", line_chart=line_chart)

@app.route("/train")
def train():
    fs = glob.glob('log/*/progress.txt')
    sorted(fs)

    algos = []
    charts = {}
    datas = {}
    key = 'TotalEnvInteracts'

    for idx, fname in enumerate(fs):
        with open(fname, 'r') as f:
            try:
                data = pd.read_table(f, sep='\t\t', engine='python')
            except:
                continue
        title = fname.strip().split('/')[1]
        algo, game, seed = title.split('-')
        algo_seed = algo + seed

        if algo_seed not in algos:
            algos.append(algo_seed)

        if game not in datas:
            datas[game] = {}
            charts[game] = {}

        for col in data.columns:
            if 'Unnamed' not in col:
                if col in ['RemHrs', 'NumOfEp', 'Epoch', 'TotalEnvInteracts', 'Steps']:
                    continue

                if col not in datas[game]:
                    datas[game][col] = {}
                    charts[game][col] = Chart("LineChart", "data")
                    charts[game][col].options = {
                        "title": f"{game}-{col}",
                        "width": 800,
                        "height": 600
                    }
                    charts[game][col].data.add_column("number", key)


                if algo_seed not in datas[game][col]:
                    datas[game][col][algo_seed] = {}
                    for i, step in enumerate(data[key].values):
                        datas[game][col][algo_seed][step] = data[col].values[i]
                    charts[game][col].data.add_column("number", algo_seed)

    for game, game_data in datas.items():
        for col, entry_data in game_data.items():
            max_steps = max([x.keys() for x in entry_data])

            for step in range(max_steps):
                entry = [x[step] if step in x else None for x in entry_data]
                charts[game][col].data.add_row([step] + entry)

    return render_template("train.html", charts=charts)

if __name__ == "__main__":
    app.run(debug=True)