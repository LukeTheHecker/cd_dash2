import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import time
import os
import pickle as pkl
import mne
from app import app
# from layouts import leadfield, pos
# from layouts import model_flex, model_gaussian, model_lean, model_lowsnr, model_paper
from components.functions import simulate_source, predict_source, make_fig_objects, load_model, inverse_solution, brain_plotly


print('Loading Some Variables')
# Load Global variables
pth_modeling = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'assets/modeling'))
# Load Triangles through some inverse operator created previously
with open(pth_modeling + '/tris.pkl', 'rb') as f:
    tris = pkl.load(f)
# Load Forward Model for other inverse solutions:
fwd = mne.read_forward_solution(pth_modeling + '/fsaverage-fwd.fif')
# Load generic Epochs structure for other inverse solutions:
# epochs = mne.read_epochs(pth_modeling + '\\epochs-epo.fif')
evokeds = mne.Evoked(pth_modeling + '/evoked-ave.fif')
## Leadfield
with open(pth_modeling +'/leadfield.pkl', 'rb') as f:
    leadfield = pkl.load(f)[0]
## Positions
with open(pth_modeling +'/pos.pkl', 'rb') as f:
    pos = pkl.load(f)[0]

print('Preloading Models')
model_paper = load_model(pth_modeling + '/model_paper/')
model_flex = load_model(pth_modeling + '/model_flex/')
model_lowsnr = load_model(pth_modeling + '/model_lowsnr/')
model_gaussian = load_model(pth_modeling + '/model_gaussian/')
model_lean = load_model(pth_modeling + '/model_lean/')

# Get Brain Plot structure
data = np.arange(0, 5124)
fig, _ = brain_plotly(data, tris, pos)
fig_brain = go.Figure(data=[go.Mesh3d(x=fig.data[0]['x'], y=fig.data[0]['y'], z=fig.data[0]['z'], i=fig.data[0]['i'], j=fig.data[0]['j'], k=fig.data[0]['k'], intensity=data, colorscale='Portland')])#, colorscale='Rainbow', 
fig_brain['layout']['height'] = 700  # 650
fig_brain['layout']['width'] = 700


######################## START ConvDip Callbacks ########################

# Callback for the Simulate-button
@app.callback(
        [Output('loading-output-simulation', 'children'),
        Output('sim_scalp_plot', 'figure'),
        Output('sim_source_plot', 'figure'),
        Output('current_y', 'children'),
        Output('current_snr', 'children')], 
        [Input('sim_button', 'n_clicks')],
        [State('noise_level_input', 'value'), 
        State('number_of_sources_input', 'value'),
        State('size_of_source_input', 'value'),
        State('source_shape', 'value'),
        State('sim_source_plot', 'figure')
        ])

def simulate_sample(*params):
    print('simulating')
    settings = [i for i in params]
    if np.any(settings==None) or settings[0]==0:
        print(f'FIRST CALL RETURNS NOTHING')
        return

    snr = settings[1]
    n_sources = settings[2]
    size = settings[3]
    source_shape = settings[4]

    start = time.time()
    y, x_img, db_choice = simulate_source(snr, n_sources, size, 1, leadfield, pos, source_shape=source_shape)
    end = time.time()
    print(f'Simulation: {end-start}')
    print(f'Simulation: {end-start}')
    print(f'Simulation: {end-start}')
    print(f'Simulation: {end-start}')

    # fig_y, fig_x = make_fig_objects(y, x_img, tris, pos)

    # Some Normalization
    x_img /= np.max(np.abs(x_img))
    y /= np.max(np.abs(y))
    # Plotting
    fig_x = px.imshow(x_img)
    fig_y = fig_brain
    fig_y.data[0].update(intensity=y)

    spinner_output = 'Simulation is Ready', html.Br(), f'Calculation took {(end-start)*1000:.0f} milliseconds'
    return spinner_output, fig_x, fig_y, y, db_choice

# Callback for the Inverse Solution Dropdown

@app.callback(
        Output('inv_description', 'children'),
        [Input('model_selection', 'value')])

def retrieve_description(value):
    if value == 'flex':
        desc = ['ConvDip trained on a large range of SNRs.', html.Br(), html.Br(), 'Trained on Flat sources', html.Br(), html.Br(), 'SNR: 0 to 8 dB.', html.Br(), 'Source diameters: 25-35', html.Br(), 'Number of Sources: 1 to 5']
    elif value == 'lean':
        desc = ['ConvDip trained on gaussian sources with only 1 convolution layer and 1 fully connected layer of 512 neurons.', html.Br(), html.Br(), 'SNR: 3 to 8 dB.', html.Br(), 'Source FWHM: 25-35', html.Br(), 'Number of Sources: 1 to 5']
    elif value == 'gaussian':
        desc = ['ConvDip trained on gaussian sources.', html.Br(), html.Br(), 'SNR: 3 to 8 dB.', html.Br(), 'Source FWHM: 25-35', html.Br(), 'Number of Sources: 1 to 5']
    elif value == 'paper':
        desc = 'ConvDip as described in the paper.', html.Br(), html.Br(), 'Trained on Flat sources', html.Br(), html.Br(), 'SNR: 6 to 9 dB.', html.Br(), 'Source diameters: 25-35', html.Br(), 'Number of Sources: 1 to 5'
    elif value == 'lowsnr':
        desc = 'ConvDip as trained on low SNR data.', html.Br(), html.Br(), 'Trained on Flat sources', html.Br(), html.Br(), 'SNR: 3 to 6 dB.', html.Br(), 'Source diameters: 25-35', html.Br(), 'Number of Sources: 1 to 5'
    elif value == 'eLORETA':
        desc = 'Exact low-resolution tomography, a commonly used inverse solution.', html.Br(), 'Developed by Pascual-Marqui (2007) and implemented in MNE python.', html.Br(), 'Tip:', html.Br(), 'eLORETA assumes large sources with gaussian distribution. Try to simulate a sample with gaussian shape and 50 mm FWHM and reconstruct it with eLORETA.'
    elif value == 'lcmv':
        desc = 'Linear constraint minimum variance (LCMV) beamformer, a commonly used inverse solution.', html.Br(), 'Developed by Van Veen et al. (1997) and implemented in MNE python.'
    elif value == 'MNE':
        desc = 'Minimum norm estimates (MNE), a commonly used inverse solution.', html.Br(), 'Developed by Hämäläinen & Ilmoniemi (1994) and implemented in MNE python.', html.Br(), 'Tip:', html.Br(), 'MNE produces large sources with gaussian distribution. Try to simulate a sample with gaussian shape and 50 mm FWHM and reconstruct it with MNE.'
    elif value == 'dSPM':
        desc = 'Dynamic statistical parametric mapping (dSPM).', html.Br(), 'Developed by Dale et al. (2000) and implemented in MNE python.', html.Br(), 'Tip:', html.Br(), 'dSPM produces large sources with gaussian distribution. Try to simulate a sample with gaussian shape and 50 mm FWHM and reconstruct it with dSPM.'
    
    return desc


# Callback for the Predict button
@app.callback(
        [Output('loading-output-prediction', 'children'),
        Output('pred_scalp_plot', 'figure'),
        Output('pred_source_plot', 'figure')], 
        [Input('predict_button', 'n_clicks')],
        [State('sim_scalp_plot', 'figure'),
        State('model_selection', 'value'),
        State('current_y', 'children'),
        State('current_snr', 'children')]
        ) 

def predict_sample(*params):
    inputs = [i for i in params]
    # Check if sample was simulated already
    if inputs[1]['data'] == [] or inputs[0]==0:
        print('No sample simulated or at least its not plotted')
        spinner_output = 'No simulation available.'
        return spinner_output, None, None
    
    # Check if hidden html.Div has source value
    try:
        y = np.asarray(inputs[3])
        db = np.asarray(inputs[4])[0]
        print(f'y.shape={y.shape}\ndb.shape={db.shape}')
        print(f'db={db}')
    except:
        spinner_output = 'No simulation available.'
        return spinner_output, None, None

    data = inputs[1]['data'][0]['z']
    data = np.asarray(data)
    start = time.time()
    if inputs[2] == 'paper':
        model = model_paper
        y, x_img = predict_source(data, leadfield, model)
    elif inputs[2] == 'gaussian':
        model = model_gaussian
        y, x_img = predict_source(data, leadfield, model)
    elif inputs[2] == 'lean':
        model = model_lean
        y, x_img = predict_source(data, leadfield, model)
    elif inputs[2] == 'lowsnr':
        model = model_lowsnr
        y, x_img = predict_source(data, leadfield, model)
    elif inputs[2] == 'flex':
        model = model_flex
        y, x_img = predict_source(data, leadfield, model)
    elif inputs[2] == 'eLORETA' or inputs[2] == 'lcmv' or inputs[2] == 'MNE' or inputs[2] == 'dSPM' or inputs[2] == 'mxne':
        x = np.sum(y * leadfield, axis=1)
        y, x_img = inverse_solution(x, db, evokeds, fwd, leadfield, inputs[2])
    end = time.time()

    
    # fig_y, fig_x = make_fig_objects(y, x_img, tris, pos)
    # Some Normalization
    x_img /= np.max(np.abs(x_img))
    y /= np.max(np.abs(y))
    # Plotting
    fig_x = px.imshow(x_img)
    fig_y = fig_brain
    fig_y.data[0].update(intensity=y)

    spinner_output = f'Prediction is ready!', html.Br(), f'Calculation took {(end-start)*1000:.0f} milliseconds'
    return spinner_output, fig_x, fig_y