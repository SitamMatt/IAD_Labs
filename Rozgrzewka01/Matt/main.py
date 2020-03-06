import plotly.graph_objects as go
import numpy as np

x = np.linspace(-3, 3)


def f1(x):
    return 1 / (1 + np.exp(-x))


def f2(x):
    return 1 / (1 + np.exp(4 * x))


def f3(x):
    return 1 / (1 + np.exp(-100 * x + 55))


fig = go.Figure(
    data=[
        go.Scatter(x=[-0.2, 0.4, 0.6, 1.2, 1.9, 0.5], y=[0, 0, 1, 1, 1, 0], mode='markers', marker=dict(
            color='black'
        )),
        go.Scatter(x=x, y=f1(x), marker= dict(
            color='red'
        )),
        go.Scatter(x=x, y=f2(x), marker= dict(
            color='blue'
        )),
        go.Scatter(x=x, y=f3(x), marker= dict(
            color='green'
        ))
    ]
)
fig.update_xaxes(range=[-3, 3])
fig.update_yaxes(range=[-0.5, 1.5])
fig.write_html('graph.html', auto_open=True)
