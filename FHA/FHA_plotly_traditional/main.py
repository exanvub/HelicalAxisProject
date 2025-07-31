import dash
from dash import html, dcc, Output, Input
import config  # will be reloaded on button click using importlib
import importlib

from FHA_visualizer_plotly import plotly_visualize_FHA
import polhemus_import as pi
import vicon_import as vi


def load_and_generate_figure():
    importlib.reload(config)  # Reload the config module to get updated values

    isPolhemus = False
    methods = ['all_FHA', 'incremental_time', 'step_angle', 'incremental_angle']

    if config.data_type == '1':
        q1, q2, loc1, loc2, t = pi.read_data(config.path)
        R1, T1, R2, T2 = pi.quaternions_to_matrices(q1, q2, loc1, loc2)
        isPolhemus = True
    elif config.data_type == '2':
        c3d, marker_data, t = vi.read_data(config.path)
        labels = c3d['parameters']['POINT']['LABELS']['value']
        for m in [config.marker1_j1, config.marker2_j1, config.marker3_j1,
                  config.marker1_j2, config.marker2_j2, config.marker3_j2]:
            if m not in labels:
                raise ValueError(f"Marker {m} not found in c3d file.")

        R1, T1, R2, T2, loc1, loc2 = vi.build_reference_frames(
            marker_data,
            config.marker1_j1, config.marker2_j1, config.marker3_j1,
            config.marker1_j2, config.marker2_j2, config.marker3_j2
        )
    else:
        raise ValueError("Invalid config.data_type. Must be '1' or '2'.")

    if config.method_type not in methods:
        raise ValueError(f"FHA method '{config.method_type}' not in supported methods {methods}")

    step = config.step if config.method_type != 'all_FHA' else 0

    from FHA_visualizer_plotly import generate_FHA  # avoid circular imports
    hax, ang, svec, d, translation_1_list, translation_2_list, time_diff, time_incr, ang_incr, all_angles, ind_incr, ind_step, t = generate_FHA(
        config.method_type, t, int(config.cut1), int(config.cut2), int(step), int(config.nn),
        R1, R2, T1, T2, loc1, loc2
    )

    fig = plotly_visualize_FHA(
        isPolhemus, T1, T2, config.method_type, hax, svec, d, ind_incr, ind_step, int(config.nn),
        translation_1_list, translation_2_list, t, time_incr, ang_incr, int(step), all_angles, time_diff
    )

    return fig


# Initialize Dash app
app = dash.Dash(__name__)
app.title = "FHA Dashboard"

# Initial figure
initial_fig = load_and_generate_figure()

# App layout
app.layout = html.Div([
    html.H2("Scrollable FHA Dashboard", style={"textAlign": "center"}),
    html.Button("Reload Config and Refresh Plot", id="reload-button", n_clicks=0),
    html.Div([dcc.Loading(id="loading-figure",
                type="circle",
                children=dcc.Graph(id='fha-figure', figure=initial_fig, style={'height': '1900px', 'width': '100%'})),
              ],
              style={
                  'height': '2000px',
                  'overflowY': 'scroll',
                  'border': '1px solid #ccc',
                  'padding': '10px',
                  'backgroundColor': '#f5f5f5'  # light gray background
              })
    ],
    style={
        'backgroundColor': '#f5f5f5'  # light gray background
    }
)

# Callback to regenerate figure
@app.callback(
    Output('fha-figure', 'figure'),
    Input('reload-button', 'n_clicks'),
    prevent_initial_call=True
)
def reload_figure(n_clicks):
    fig = load_and_generate_figure()
    return fig


if __name__ == '__main__':
    app.run(debug=True)