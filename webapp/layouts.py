import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px

import dash_core_components as dcc

import numpy as np
from datetime import datetime as dt
from datetime import date, timedelta
import pandas as pd
from app import app
import mne
import os
import pickle as pkl
import time

from components import Header
from components.functions import simulate_source, predict_source, make_fig_objects, load_model, inverse_solution, brain_plotly


######################## START ConvDip Layout ########################
layout_convdip_page =  html.Div([
    html.Div([
        # CC Header
        Header(),
        # Header Bar
        html.Div([
          html.Center(html.H1(["ConvDip: A convolutional neural network for better M/EEG Source Imaging"], style={'marginTop': 15})),  #, className="gs-header gs-text-header padded",style={'marginTop': 15}));
          html.Br(),
          html.Center(html.H4(["Hecker, Lukas; Rupprecht, Rebekka; Tebartz van Elst, Ludger; Kornmeier, Juergen; 04/2020"], style={'marginTop': 15})),
          ]),
        # Hidden divs: 
        # stores the simulated source vector y
        html.Div(id='current_y', style={'display': 'none'}),
        # stores the selected SNR in dB
        html.Div(id='current_snr', style={'display': 'none'}),

        # Abstract
        dbc.Card(
            dbc.CardBody([
                html.H6('ConvDip is a convolutional neural network that finds solutions for the inverse problem of the EEG. The inverse problem describes the search for the neural generators of the EEG signal. Since the inverse problem is highly underdetermined, there is more than one plausible solution. Several inverse solutions were developed in the past years, each of which with their own advantages and disadvantages.', className="card-text"),
                html.H6(['Here you find a playground to test and compare the algorithm with some conventional methods below']),
                html.H6(['See the paper for more detailed descriptions of the method: ', html.A("Link to BioRxiv", href='https://www.biorxiv.org/content/10.1101/2020.04.09.033506v1', target="_blank")]),
                html.Br(),
                html.H4('Get Started:', className="card-text"),
                html.H5('1.) Simulate a sample by clicking the button <Simulate Sample>'),
                html.H5('2.) Check the resulting brain plot in the middle and the resulting EEG measurement on the right.'),
                html.H5('3.) Predict the source by clicking <Predict Source> and compare the resulting brain plot with the one you simulated before. How similar are they?'),
                html.H5('4.) You can play around with various combinations of simulation parameters and the provided inverse solutions. Have fun !')
                
            ]
            ), style={'margin': '20px'}
        )
        ,
        dbc.Row([
            # Simulation Panel
            dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H4(["Advanced Options"], style={'marginTop':15}),
                        html.Br(),
                        html.H5('Signal to Noise Ratio (in dB):'),
                        dcc.Markdown('''###### *e.g. single value: 6 or range of values: 6, 9*'''),
                        dbc.Input(id='noise_level_input', value='6, 9'),
                        html.Br(),
                        ]),
                    html.Div([
                        html.Br(),
                        html.H5('Number of sources:'),
                        dcc.Markdown('''###### *e.g. single value: 3 or range of values: 1, 5*'''),
                        dbc.Input(id='number_of_sources_input', value='1, 5'),
                        html.Br(),
                    ]),
                    html.Div([
                        html.Br(),
                        html.H5('Diameter of sources (in mm):'),
                        dcc.Markdown('''###### *e.g. single value: 35 or range of values: 25, 35*'''),
                        dbc.Input(id='size_of_source_input', value='25, 35'),
                        html.Br(),
                        ]),
                    html.Div([
                        html.Br(),
                        html.H5('Shape of sources:'),
                        dcc.Markdown('''###### *If gaussian is selected the diameter of source becomes the full width at half maximum (FWHM)*'''),
                        dbc.RadioItems(
                            id='source_shape',
                            options=[
                                {'label': 'Gaussian', 'value': 'gaussian'},
                                {'label': 'Flat', 'value': 'flat'}
                            ],
                            value='gaussian'
                        ),
                        html.Br()
                        ]),
                    html.Div([
                        html.Br(),
                        dbc.Button('Simulate Sample', id='sim_button', color="primary"),
                        dbc.Spinner(html.Div(id="loading-output-simulation")),
                        ]),
                    # html.Div([
                    #     html.Div(id='output_container_button',
                    #     children='Enter the values and press button')
                    #     ]),
                    ])
                ], style={'margin': '20px'}), # end of settings card
            width=3),

            # Simulation: Scalp Map
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                    html.H4("Simulated brain-electric activity", className="card-title"),
                    html.Br(),
                    dcc.Graph(
                        id='sim_source_plot',
                        config={
                            'displayModeBar': False
                            },
                        figure={
                            'data': [],
                            },
                        # style={'width': '700px'}
                        ),
                    html.Br(),
                    html.P("This plot depicts the cortical surface of the brain with its gyri and sulci. Colored in orange you see the simulated electric activity.", className="card-title"),

                        ]),
                    ], style={'margin': '20px'}),
                    width=5),
                
            # Simulation Canvas 2
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Resulting Scalp map of the brain-electric activity", className="card-title"),
                        html.Br(),
                        dcc.Graph(
                            id="sim_scalp_plot",
                            figure={
                                "data": [],
                                },
                            ),
                        html.Br(),
                        html.P("This is a low-resolution representation of the EEG (aerial view). This map is generated by projecting the simulated brain activity to the electrodes of a simulated set of EEG electrodes. This is possible through a forward model which describes the conductive and geometric properties of tissues of the head (e.g. brain, dura, skull). The task of an inverse solution is to infer the activity shown on the left (brain plot) from the scalp map shown.", className="card-title"),
                        ]),
                    ], style={'margin': '20px'}),
                width=4)
                
            ]),  # end of simulation row
        html.Br(),
        html.Br(),
        # Prediction Group
        dbc.Row([
            dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    dbc.Select(
                        id='model_selection',
                        options=[
                            {'label': 'ConvDip for gaussian sources', 'value': 'gaussian'},
                            {'label': 'ConvDip for gaussian sources: Small Model', 'value': 'lean'},
                            {'label': 'ConvDip Flexible', 'value': 'flex'},
                            {'label': 'ConvDip Paper', 'value': 'paper'},
                            {'label': 'ConvDip for low SNR', 'value': 'lowsnr'},
                            {'label': 'eLORETA', 'value': 'eLORETA'},
                            {'label': 'LCMV Beamforming', 'value': 'lcmv'},
                            {'label': 'Minimum Norm Estimate', 'value': 'MNE'},
                            {'label': 'dSPM', 'value': 'dSPM'},
                        ],
                        value='gaussian'
                        ),
                    html.Br(),
                    html.Br(),
                    html.Div(id='inv_description'),
                    html.Br(),
                    dbc.Button('Predict Source', id='predict_button', color="primary"),
                    dbc.Spinner(html.Div(id="loading-output-prediction")),
                    ])
                ], style={'margin': '20px'}),
                width=3),
        # Prediction Figures
        dbc.Col(
        dbc.Card([
            dbc.CardBody([
                html.H4("Predicted brain activity (the inverse solution)", className="card-title"),
                html.Br(),
                dcc.Graph(
                    id='pred_source_plot',
                    config={
                        'displayModeBar': False
                        },
                    figure={
                        'data': [],
                        },
                    # style={'width': '700px'}
                    ),
                html.Br(),
                html.P("This plot depicts the inverse solution. In the best case this brain activity depicts exactly the simulated activity of the figure above. However, the inverse problem does not have a unique solution, wherefore we can only expect rough approximations.", className="card-title"),
                ]),
            ], style={'margin': '20px'}),
            width=5),


        # Simulation Canvas 2
        dbc.Col(
        dbc.Card([
            dbc.CardBody([
                html.H4("Resulting Scalp Map of the prediction.", className="card-title"),
                html.Br(),
                dcc.Graph(
                    id="pred_scalp_plot",
                    figure={
                        "data": [],
                        },
                    ),
                html.Br(),
                html.P("This is a low-resolution representation of the EEG (aerial view). This map is generated by projecting the predicted brain activity (figure to the left) to the electrodes of a simulated set of EEG electrodes. This is possible through a forward that describes the conductive and geometric properties of tissues of the head(e.g. brain, dura, skull).", className="card-title"),

                ]),
            ], style={'margin': '20px'}),
        width=4)
        ]), # End of Prediction Group
        # Architecture
        html.Div([
            html.Center(html.H1(["ConvDip architecture"], style={'marginTop': 15})),  #, className="gs-header gs-text-header padded",style={'marginTop': 15}));
            html.Br(),
            html.Center(html.Img(src='/assets/architecture.png', height=500)),
            ]),
        html.Div([], style={'margin': '20px'}),
        ], className="subpage"), 
    ], className="page")


main_page = html.Div([
    html.Div([
        Header()
        ])

    ])
page_not_found = html.Div([
    html.Div([
        Header(),
        html.H1('404 Page not found :(')
    ])
])