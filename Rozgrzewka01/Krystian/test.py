import plotly.graph_objects as go
import numpy as np
np.random.seed(1)
x = np.arange(-3.0, 3.0, 0.01)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=1/(1+np.exp(-x)),
                    line=dict(color='red'),
                    name='f1'))
fig.add_trace(go.Scatter(x=x, y=1/(1+np.exp(4*x)),
                    line=dict(color='blue'),
                    name='f2'))
fig.add_trace(go.Scatter(x=x, y=1/(1+np.exp(-100*x+55)),
                    line=dict(color='green'),
                    name='f3'))
fig.add_trace(go.Scatter(x=[-0.2, 0.4, 0.6, 1.2, 1.9, 0.5], y=[0, 0, 1, 1, 1, 0],
                    mode='markers',
                    marker=dict(color='Black'),
                    name='punkty'))
fig.update_xaxes(range=[-3, 3])
fig.update_yaxes(range=[-0.5, 1.5])
fig.show()