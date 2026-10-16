import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# ── Data Generation ──────────────────────────────────────────────
np.random.seed(42)
time_steps = 1000

def generate_pump_data(time_steps, fault_type=None, fault_start=700):
    t = np.arange(time_steps)
    flow_rate   = 0.065 + np.random.normal(0, 0.002, time_steps)
    pressure    = 35   + np.random.normal(0, 0.5,   time_steps)
    vibration   = 0.5  + np.random.normal(0, 0.05,  time_steps)
    temperature = 25   + np.random.normal(0, 0.3,   time_steps)
    power       = 15   + np.random.normal(0, 0.2,   time_steps)

    if fault_type == 'bearing_wear':
        mask = t >= fault_start
        vibration[mask]   += np.linspace(0, 3,  mask.sum())
        temperature[mask] += np.linspace(0, 15, mask.sum())
    elif fault_type == 'cavitation':
        mask = t >= fault_start
        flow_rate[mask] -= np.linspace(0, 0.02, mask.sum())
        vibration[mask] += np.random.normal(2, 0.5, mask.sum())
    elif fault_type == 'blockage':
        mask = t >= fault_start
        flow_rate[mask] -= np.linspace(0, 0.03, mask.sum())
        pressure[mask]  += np.linspace(0, 10,   mask.sum())
        power[mask]     += np.linspace(0, 5,    mask.sum())

    labels = np.ones(time_steps)
    if fault_type:
        labels[fault_start:] = 0

    return pd.DataFrame({
        'time': t, 'flow_rate': flow_rate, 'pressure': pressure,
        'vibration': vibration, 'temperature': temperature,
        'power': power, 'health': labels,
        'fault_type': fault_type or 'normal'
    })

normal_data    = generate_pump_data(time_steps)
bearing_data   = generate_pump_data(time_steps, 'bearing_wear')
cavitation_data= generate_pump_data(time_steps, 'cavitation')
blockage_data  = generate_pump_data(time_steps, 'blockage')
all_data       = pd.concat([normal_data, bearing_data,
                             cavitation_data, blockage_data],
                            ignore_index=True)

# ── Train Model ───────────────────────────────────────────────────
features = ['flow_rate', 'pressure', 'vibration', 'temperature', 'power']
healthy  = all_data[all_data['health'] == 1]
faulty   = all_data[all_data['health'] == 0]
faulty_up = resample(faulty, replace=True,
                     n_samples=len(healthy), random_state=42)
balanced  = pd.concat([healthy, faulty_up]).sample(frac=1, random_state=42)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(balanced[features])
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, balanced['health'])

# ── App Layout ────────────────────────────────────────────────────
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(style={'backgroundColor': '#0f0f1a', 'minHeight': '100vh',
                              'fontFamily': 'Arial', 'padding': '20px'}, children=[

    # Title
    html.H1(" Digital Twin — Pump Health Monitor",
            style={'color': '#00d4ff', 'textAlign': 'center',
                   'fontSize': '28px', 'marginBottom': '5px'}),
    html.P("Real-time fault detection using LSTM + Random Forest | By Smrity Mallik",
           style={'color': '#888', 'textAlign': 'center', 'marginBottom': '30px'}),

    # Controls
    html.Div(style={'display': 'flex', 'gap': '40px',
                    'justifyContent': 'center', 'marginBottom': '30px'}, children=[

        html.Div([
            html.Label("Select Fault Type", style={'color': '#fff', 'marginBottom': '8px'}),
            dcc.Dropdown(
                id='fault-dropdown',
                options=[
                    {'label': ' Normal Operation', 'value': 'normal'},
                    {'label': ' Bearing Wear',    'value': 'bearing_wear'},
                    {'label': ' Cavitation',       'value': 'cavitation'},
                    {'label': ' Blockage',         'value': 'blockage'},
                ],
                value='normal',
                style={'width': '250px', 'backgroundColor': '#1a1a2e', 'color': '#000'}
            )
        ]),

        html.Div([
            html.Label("Time Range", style={'color': '#fff', 'marginBottom': '8px'}),
            dcc.Slider(id='time-slider', min=100, max=1000, step=100,
                       value=1000,
                       marks={i: {'label': str(i),
                                  'style': {'color': '#fff'}}
                              for i in range(100, 1100, 100)})
        ], style={'width': '400px'})
    ]),

    # Health Score
    html.Div(id='health-score-div',
             style={'textAlign': 'center', 'marginBottom': '20px'}),

    # Sensor Charts
    html.Div(style={'display': 'grid',
                    'gridTemplateColumns': 'repeat(3, 1fr)',
                    'gap': '15px', 'marginBottom': '20px'}, children=[
        dcc.Graph(id='flow-chart'),
        dcc.Graph(id='vibration-chart'),
        dcc.Graph(id='temperature-chart'),
    ]),

    html.Div(style={'display': 'grid',
                    'gridTemplateColumns': 'repeat(2, 1fr)',
                    'gap': '15px', 'marginBottom': '20px'}, children=[
        dcc.Graph(id='pressure-chart'),
        dcc.Graph(id='power-chart'),
    ]),

    # Feature importance
    dcc.Graph(id='importance-chart'),

    html.P("Smrity Mallik | MSc AI & Data Science | Mechanical Engineering",
           style={'color': '#444', 'textAlign': 'center', 'marginTop': '20px'})
])

# ── Callbacks ─────────────────────────────────────────────────────
@app.callback(
    Output('health-score-div',   'children'),
    Output('flow-chart',         'figure'),
    Output('vibration-chart',    'figure'),
    Output('temperature-chart',  'figure'),
    Output('pressure-chart',     'figure'),
    Output('power-chart',        'figure'),
    Output('importance-chart',   'figure'),
    Input('fault-dropdown',      'value'),
    Input('time-slider',         'value')
)
def update_dashboard(fault_type, time_range):
    data = generate_pump_data(time_steps, 
                               None if fault_type == 'normal' else fault_type)
    data = data[data['time'] < time_range]

    # Health score
    X_input  = scaler.transform(data[features])
    preds    = rf_model.predict(X_input)
    health_score = int(preds.mean() * 100)
    color    = '#00ff88' if health_score > 80 else \
               '#ffaa00' if health_score > 50 else '#ff4444'

    score_div = html.Div([
        html.H2(f"Pump Health Score: {health_score}%",
                style={'color': color, 'fontSize': '32px'}),
        html.P(
            " Healthy" if health_score > 80 else
            " Warning — Fault Developing" if health_score > 50 else
            " Critical — Fault Detected",
            style={'color': color, 'fontSize': '18px'}
        )
    ])

    def make_chart(col, title, color_line):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['time'], y=data[col],
            mode='lines', line=dict(color=color_line, width=1.5),
            name=col
        ))
        if fault_type != 'normal':
            fig.add_vline(x=700, line_dash='dash',
                          line_color='red',
                          annotation_text='Fault Start',
                          annotation_font_color='red')
        fig.update_layout(
            title=title, template='plotly_dark',
            paper_bgcolor='#1a1a2e', plot_bgcolor='#1a1a2e',
            font=dict(color='#fff'), height=250,
            margin=dict(l=40, r=20, t=40, b=30)
        )
        return fig

    flow_fig  = make_chart('flow_rate',   'Flow Rate (m³/s)',    '#00d4ff')
    vib_fig   = make_chart('vibration',   'Vibration (mm/s)',    '#ff6b6b')
    temp_fig  = make_chart('temperature', 'Temperature (°C)',    '#ffd93d')
    pres_fig  = make_chart('pressure',    'Pressure (m head)',   '#6bcb77')
    power_fig = make_chart('power',       'Power (kW)',          '#c77dff')

    # Feature importance
    imp_df = pd.DataFrame({
        'Feature':    features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=True)

    imp_fig = px.bar(imp_df, x='Importance', y='Feature',
                     orientation='h',
                     title='Sensor Importance for Fault Detection',
                     color='Importance', color_continuous_scale='Blues',
                     template='plotly_dark')
    imp_fig.update_layout(paper_bgcolor='#1a1a2e',
                          plot_bgcolor='#1a1a2e', height=300)

    return (score_div, flow_fig, vib_fig,
            temp_fig, pres_fig, power_fig, imp_fig)

if __name__ == '__main__':
    app.run(debug=True)
