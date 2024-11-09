import dash
import pandas as pd
import numpy as np
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State,ALL
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import plotly.graph_objs as go
from dash import html
import dash_bootstrap_components as dbc
from dash import dcc
import base64
import plotly.express as px
import io
from datetime import datetime
from sklearn.decomposition import PCA

# Initialize the Dash app with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = 'Advanced AI Project'
server = app.server
# App layout
app.layout = html.Div(
    [
        # Navbar Section
        dbc.Row(dbc.Col(dbc.Spinner(html.Div(id='forecast-output-spinner'),fullscreen=True,fullscreen_style={'backgroundColor': 'rgba(0, 0, 0, 0.7)'}))),
        html.Nav(
            className="navbar navbar-expand-lg navbar-dark bg-dark w-100",
            children=[
                # Left section with logo and brand name
                html.A(
                    className="navbar-brand d-flex align-items-center",
                    href="index.py",
                    children=[
                        html.Img(
                            src="assets/doms_logo_wb.png",
                            alt="Logo",
                            height="100px",
                            style={"marginRight": "10px"}
                        ),
                    ]
                ),
                
                # Toggle button for collapsible navbar (for small screens)
                html.Button(
                    className="navbar-toggler",
                    type="button",
                    **{"data-toggle": "collapse", "data-target": "#navbarNav", "aria-controls": "navbarNav", "aria-expanded": "false", "aria-label": "Toggle navigation"},
                    children=[
                        html.Span(className="navbar-toggler-icon")
                    ]
                ),
                
                # Collapsible content
                html.Div(
                    className="collapse navbar-collapse justify-content-end",
                    id="navbarNav",
                    children=[
                        # Right section with user profile and greeting
                        html.Ul(
                            className="navbar-nav",
                            children=[
                                html.Li(
                                    className="nav-item",
                                    children=html.A(
                                        className="nav-link d-flex align-items-center",
                                        href="#",
                                        children=[
                                            html.Div(
                                                html.Img(
                                                    src="assets/img/profile.jpg",
                                                    alt="Profile",
                                                    className="rounded-circle",
                                                    height="30px"
                                                ),
                                                style={"marginRight": "8px"}
                                            ),
                                            html.Span(className="profile-username",children=[
                                                html.Span("Hi", className="op-7"),
                                                html.Span(" Team 5", className="fw-bold")
                                            ])
                                        ]
                                    )
                                )
                            ]
                        )
                    ]
                )
            ]
        ),
        html.Div(className="container",children=[
                html.Div(className="page-inner",children=[
                    html.Div(className="d-flex align-items-left align-items-md-center flex-column flex-md-row pt-2 pb-4",children=[
                        html.Div(children=[
                            html.H3("Dashboard for Consumer Price Index", className="fw-bold mb-3"),
                            html.Div(children = [
                                html.H6("Upload your file here", className="op-7 mb-2"),
                                html.A("Link to user guide",href='https://onedrive.live.com/edit?id=2FEB8F5622E5570E!4028&resid=2FEB8F5622E5570E!4028&ithint=file%2cdocx&authkey=!AqzxOtGUwJckLbc&wdo=2&cid=2feb8f5622e5570e')

                            ]),
                        ]),
                        html.Div(className="ms-md-auto py-2 py-md-0",children=[
                            dcc.Upload(
                                id="formFileLg",
                                children=html.Button("Select File", className="btn btn-dark btn-lg"),
                                multiple=False  # Set to True if you want to allow multiple file uploads
                            ),
                        ]),
                        dcc.Store(id='stored-data'),
                    ]),
                    # Box to display uploaded file info
                    html.Div(id="file-info-box",className="border rounded p-3 mt-3",style={"display": "none"},children=[
                        html.H5("Uploaded File"),
                        html.Hr(),
                        html.Div(id="file-info")  # Display file info here
                    ])
                ]),
        ]),
        html.Div(className="container", children=[
            html.Div(className="row", children=[
                html.Div(className="col-md-12", children=[
                    html.Div(className="card card-round", children=[
                        html.Div(className="card-header", children=[
                            html.Div(className="card-head-row", children=[
                                html.Div(className="card-title", children="Exploratory data analysis"),
                            ])
                        ]),
                        html.Div(className="card-body", children=[
                            html.Div(className="row", children=[
                                html.Div(className="col-md-6", children=[
                                    html.Div(className="card card-round", children=[
                                        html.Div(className="card-header", children=[
                                            html.Div(className="card-head-row", children=[
                                                html.Div(className="card-text font-weight-bold", children="Select State"),
                                            ])
                                        ]),
                                        html.Div(className="card-body", children=[
                                            # Add content here, e.g., a dropdown for selecting a state
                                            dcc.Dropdown(
                                                id='dropdown-for-states',
                                                options=[],
                                                clearable = False,
                                                # value=products[0] if products else None
                                            ),
                                            html.Div(id='display-selected-value-states')
                                        ])
                                    ])
                                ]),
                                html.Div(className="col-md-6", children=[
                                    html.Div(className="card card-round", children=[
                                        html.Div(className="card-header", children=[
                                            html.Div(className="card-head-row", children=[
                                                html.Div(className="card-text font-weight-bold", children="Select Region"),
                                            ])
                                        ]),
                                        html.Div(className="card-body", children=[
                                            # Add content here, e.g., a dropdown for selecting a region
                                            dcc.Dropdown(
                                                id='dropdown-for-regions',
                                                options=[],
                                                clearable = False,
                                                # value=products[0] if products else None
                                            ),
                                            html.Div(id='display-selected-value-regions')
                                        ])
                                    ])
                                ]),
                            ])
                        ]),
                        html.Div(className="card-body", children=[
                            html.Div(className="row", children=[
                                html.Div(className="col-md-12", children=[
                                    html.Div(className="card card-round", children=[
                                        html.Div(className="card-header", children=[
                                            html.Div(className="card-head-row", children=[
                                                html.Div(className="card-text font-weight-bold", children="Select The Year (For calculation of the elbow for clustering)"),
                                            ])
                                        ]),
                                        html.Div(className="card-body", children=[
                                            # Add content here, e.g., a dropdown for selecting a region
                                            dcc.Dropdown(
                                                id='dropdown-for-years',
                                                options=[],
                                                clearable = False,
                                                # value=products[0] if products else None
                                            ),
                                            html.Div(id='display-selected-value-of-years')
                                        ])
                                    ])
                                ]),
                            ])
                        ]),
                        html.Div(
                            html.Button("Please Show", type='submit', className="btn btn-dark col-md-11 col-sm-11 col-xs-11 mb-4", id='submit-btn', n_clicks=0),
                            className="d-flex justify-content-center"
                        ),
                    ]),
        html.Div(id="exploratory-shower",children=[
                    html.Div(className="col-md-12", children=[
                        html.Div(className="card card-round", children=[
                            html.Div(className="card-header", children=[
                                html.Div(className="card-head-row", children=[
                                    html.Div(className="card-title", children="Exploratory Statistics"),
                                ]),
                                html.Div(className="card-body", children=[
                                    html.Div(className="row", children=[
                                        html.Div(className="col-md-6", children=[
                                            html.Div(className="card card-round", children=[
                                                html.Div(className="card-header", children=[
                                                    html.Div(className="card-head-row", children=[
                                                        html.Div(id='display-eda-graph',className="container")
                                                    ])
                                                ])
                                            ])
                                        ]),
                                        html.Div(className="col-md-6", children=[
                                            html.Div(className="card card-round", children=[
                                                html.Div(className="card-header", children=[
                                                    html.Div(className="card-head-row", children=[
                                                        html.Div(id='display-eda-numerics',children=[
                                                            "Range",
                                                            "Highest Value",
                                                            "Lowest Value",
                                                            "CAGR (Compunded Annual Growth Rate)",
                                                            "25th Percentile",
                                                            "75th Percentile"
                                                            "Mean Value"
                                                            "Median Value"
                                                        ])
                                                    ])
                                                ])
                                            ])
                                        ])
                                    ]),
                                ]),
                                html.Div(className="container",id="clustering-card",style={"display":"none"},children=[
                                    html.Div(className="card-title", children="Clustering Statistics"),
                                    html.Div(className="card-body", children=[
                                        html.Div(className="row", children=[
                                            html.Div(className="col-md-12", children=[
                                                html.Div(className="card card-round", children=[
                                                    html.Div(className="card-header", children=[
                                                        html.Div(className="card-head-row", children=[
                                                            html.Div(id='display-elbow-graph',className="container")
                                                        ])
                                                    ])
                                                ])
                                            ]),
                                        ]),
                                    ])
                                ])
                            ]),
                        ])
                    ])
                ]), 
            ]),
        ]),
        html.Div(className="col-md-12", children=[
        html.Div(className="card card-round", children=[
            html.Div(className="card-header", children=[
                html.Div(className="card-head-row", children=[
                    html.Div(className="card-title", children="AI Selector"),
                ]),
            ]),
            html.Div(className="card-body", children=[
                dcc.Dropdown(
                    id='dropdown',
                    options=[
                        {'label': '---Select The Option---', 'value': ''},
                        {'label': 'Forecast of the CPI', 'value': 'forecast_cpi'},
                        {'label': 'Classification of the data into clusters', 'value': 'classify_clusters'}     
                    ],
                    clearable=False,
                    value=''
                ),
                html.Div(id='display-selected-value-of-ai')
            ])
        ])
    ]),

    # This Div will show/hide based on the selected dropdown value
    html.Div(id='forecast-options', className="col-md-12", style={'display': 'none'}, children=[
        html.Div(className="card card-round", children=[
            html.Div(className="card-header", children=[
                html.Div(className="card-head-row", children=[
                    html.Div(className="card-title", children="Forecasting the Future CPI's"),
                ]),
            ]),
            html.Div(className="card-body", children=[
                html.Div(className="row", children=[
                    # State selection dropdown
                    html.Div(className="col-md-6", children=[
                        html.Div(className="card card-round", children=[
                            html.Div(className="card-header", children=[
                                html.Div(className="card-head-row", children=[
                                    html.Div(className="card-text font-weight-bold", children="Select State"),
                                ])
                            ]),
                            html.Div(className="card-body", children=[
                                dcc.Dropdown(
                                    id='state-dropdown-for-forecast',
                                    options=[],
                                    clearable=False
                                ),
                                html.Div(id='state-output-for-forecast')
                            ])
                        ])
                    ]),
                    # Region selection dropdown
                    html.Div(className="col-md-6", children=[
                        html.Div(className="card card-round", children=[
                            html.Div(className="card-header", children=[
                                html.Div(className="card-head-row", children=[
                                    html.Div(className="card-text font-weight-bold", children="Select Region"),
                                ])
                            ]),
                            html.Div(className="card-body", children=[
                                dcc.Dropdown(
                                    id='region-dropdown-for-forecast',
                                    options=[],
                                    clearable=False
                                ),
                                html.Div(id='region-output-for-forecast')
                            ])
                        ])
                    ]),
                ]),
                # Additional settings for forecast
                html.Div(className="row", children=[
                    # Number of Estimators input
                    html.Div(className="col-md-6", children=[
                        html.Div(className="card card-round", children=[
                            html.Div(className="card-header", children=[
                                html.Div(className="card-head-row", children=[
                                    html.Div(className="card-text font-weight-bold", children="Number of Estimators: (Recommended 50-1000)"),
                                ])
                            ]),
                            html.Div(className="card-body", children=[
                                dcc.Input(id="n_estimators", type="number", value=1000, min=1, step=1, className="form-control"),
                                html.Div(id='n_estimators_for_forecast')
                            ])
                        ])
                    ]),
                    # Minimum Samples Split input
                    html.Div(className="col-md-6", children=[
                        html.Div(className="card card-round", children=[
                            html.Div(className="card-header", children=[
                                html.Div(className="card-head-row", children=[
                                    html.Div(className="card-text font-weight-bold", children="Minimum Samples Split: (Recommended 2-5)"),
                                ])
                            ]),
                            html.Div(className="card-body", children=[
                                dcc.Input(id="min_samples_split", type="number", value=2, min=2, step=1, className="form-control"),
                                html.Div(id='min_samples_split_for_forecast')
                            ])
                        ])
                    ]),
                ]),
                html.Div(className="row", children=[
                    html.Div(className="col-md-6", children=[
                        html.Div(className="card card-round", children=[
                            html.Div(className="card-header", children=[
                                html.Div(className="card-head-row", children=[
                                    html.Div(className="card-text font-weight-bold", children="Citerion"),
                                ])
                            ]),
                            html.Div(className="card-body", children=[
                                # Add content here, e.g., a dropdown for selecting a state
                                dcc.Dropdown(
                                    id='criterion',
                                    options=[
                                        {"label": "squared_error", "value": "squared_error"},
                                        {"label": "absolute_error", "value": "absolute_error"},
                                        {"label": "poisson", "value": "poisson"}
                                    ],
                                    value="squared_error",
                                    clearable=False,
                                ),
                                html.Div(id='criterion-for-forecast')
                            ])
                        ])
                    ]),   
                    html.Div(className="col-md-6", children=[
                        html.Div(className="card card-round", children=[
                            html.Div(className="card-header", children=[
                                html.Div(className="card-head-row", children=[
                                    html.Div(className="card-text font-weight-bold", children="Time instances you want to select for forecast"),
                                ])
                            ]),
                            html.Div(className="card-body", children=[
                                # Add content here, e.g., a dropdown for selecting a region
                                dcc.Input(id="months", type="number", value=12, min=1, step=1,className="form-control"),
                                html.Div(id='months-for-forecast')
                            ])
                        ])
                    ]),                         
                ])
            ]),                    
            html.Div(
                html.Button("Please Show", type='submit', className="btn btn-dark col-md-11 col-sm-11 col-xs-11 mb-4", id='forecast-submit', n_clicks=0),
                className="d-flex justify-content-center"
            ),   
            html.Div(id="forecast-shower",style={"display": "none"}, children=[  
                html.Div(className="card-body", children=[
                    html.Div(className="row", children=[
                        # State selection dropdown and graph container
                        html.Div(className="col-md-12", children=[
                            html.Div(className="card card-round", children=[
                                html.Div(className="card-header", children=[
                                    html.Div(className="card-head-row", children=[
                                        html.Div(className="card-title", children="Forecasted Values using Random Forest Technique"),
                                    ])
                                ]),
                                html.Div(id="forecasted-graph"),
                                html.Div(className="card-head-row", children=[
                                    html.Div(id="mape-value"),
                                ])
                            ])
                        ])
                    ])
                ])
            ])           
        ])
    ]),
    html.Div(id='clustering', className="col-md-12",style={'display': 'none'}, children=[
        html.Div(className="card card-round", children=[
            html.Div(className="card-header", children=[
                html.Div(className="card-head-row", children=[
                    html.Div(className="card-title", children="Classifying and Clustering the states based upon CPI's"),
                ]),
            ]),
            html.Div(className="row container", children=[
                # State selection dropdown
                html.Div(className="col-md-6", children=[
                    html.Div(className="card card-round", children=[
                        html.Div(className="card-header", children=[
                            html.Div(className="card-head-row", children=[
                                html.Div(className="card-text font-weight-bold", children="Select Region"),
                            ])
                        ]),
                        html.Div(className="card-body", children=[
                            dcc.Dropdown(
                                id='region-of-clustering',
                                options=[],
                                clearable=False
                            ),
                            html.Div(id='region-output-of-clustering')
                        ])
                    ])
                ]),
                # Region selection dropdown
                html.Div(className="col-md-6", children=[
                    html.Div(className="card card-round", children=[
                        html.Div(className="card-header", children=[
                            html.Div(className="card-head-row", children=[
                                html.Div(className="card-text font-weight-bold", children="Select Year"),
                            ])
                        ]),
                        html.Div(className="card-body", children=[
                            dcc.Dropdown(
                                id='clustering-dropdown-for-years',
                                options=[],
                                clearable = False,
                                # value=products[0] if products else None
                            ),
                            html.Div(id='clustering-display-selected-value-of-years')
                        ])
                    ])
                ]),
                html.Div(className="row container", children=[
                    # State selection dropdown
                    html.Div(className="col-md-12", children=[
                        html.Div(className="card card-round", children=[
                            html.Div(className="card-header", children=[
                                html.Div(className="card-head-row", children=[
                                    html.Div(className="card-text font-weight-bold", children="Select the number of clusters you want to make"),
                                ])
                            ]),
                            html.Div(className="card-body", children=[
                                dcc.Input(id="k_means_input", type="number", value=1, min=1, max=20,step=1, className="form-control"),
                                html.Div(id='k_means_output')
                            ])
                        ])
                    ])
                ]),
                html.Div(
                    html.Button("Please Show us the Classification Results", type='submit', className="btn btn-dark col-md-11 col-sm-11 col-xs-11 mb-4", id='classification-submit', n_clicks=0),
                    className="d-flex justify-content-center"
                ),
                html.Div(id="classification-shower",style={"display": "none"}, children=[  
                html.Div(className="card-body", children=[
                    html.Div(className="row", children=[
                        # State selection dropdown and graph container
                        html.Div(className="col-md-12", children=[
                            html.Div(className="card card-round", children=[
                                html.Div(className="card-header", children=[
                                    html.Div(className="card-head-row", children=[
                                        html.Div(className="card-title", children="Classified Clusters using K-Means Clustering"),
                                    ])
                                ]),
                                html.Div(id="classification-graph"),
                            ])
                        ])
                    ])
                ])
            ]) 
                ]),
            ]),
            
        ])
    ])
    ],
style={"width": "100%"}  # Makes sure the navbar is 100% of the screen width
)
@app.callback(
    [
        Output("file-info-box", "style"), 
        Output("file-info", "children"),
        Output("dropdown-for-states", "options"),  
        Output("dropdown-for-states", "value"),
        Output("state-dropdown-for-forecast", "options"),  
        Output("state-dropdown-for-forecast", "value"),
        Output("dropdown-for-regions", "options"),  
        Output("dropdown-for-regions", "value"),
        Output("dropdown-for-years", "options"),  
        Output("dropdown-for-years", "value"),
        Output("region-dropdown-for-forecast", "options"),  
        Output("region-dropdown-for-forecast", "value"),
        Output('region-of-clustering', "options"),  
        Output('region-of-clustering', "value"),
        Output('clustering-dropdown-for-years', "options"),  
        Output('clustering-dropdown-for-years', "value"),
        Output('forecast-output-spinner', 'children'),
    ],
    [Input("formFileLg", "contents")],
    [State("formFileLg", "filename")]
)
def update_file_info(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        try:
            # Read the uploaded CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            
            # Extract state and region columns
            state_options = [{"label": col, "value": col} for col in df.columns[3:]]
            region_options = [{"label": region, "value": region} for region in df['Sector'].unique()]
            unique_years = [{"label": str(year), "value": year} for year in sorted(df['Year'].unique())]
            # Extract unique years and sort them
            unique_years_sorted = [{"label": "--Select the Year--", "value": ""}] + unique_years
            
            # Set default values to the first option in each list
            default_state_value = state_options[0]['value'] if state_options else None
            default_region_value = region_options[0]['value'] if region_options else None
            default_year_value = unique_years_sorted[0]['value'] if unique_years_sorted else None
            default_years_value = unique_years[0]['value'] if unique_years else None
            
            # Display file name and contents in a text area
            file_content_display = html.Div([
                html.H5(f'Loaded file: {filename}'),
                dcc.Textarea(
                    value=df.to_string(),
                    readOnly=True,
                    style={'width': '100%', 'height': '300px'}
                )
            ])
            
            # Return all values to the callback output
            return {
                "display": "block"
            }, file_content_display, state_options, default_state_value, state_options, default_state_value, region_options, default_region_value, unique_years_sorted, default_year_value, region_options, default_region_value, region_options, default_region_value, unique_years, default_years_value,""

        except Exception as e:
            print(f'Error processing file: {str(e)}')
            error_message = html.Div([
                html.H5(f'Error loading file: {filename}'),
                html.P(f'There was an error processing this file: {str(e)}')
            ])
            return {"display": "block"}, error_message, [], None, [], None, [], None, [], None, [], None, [], None, [],None,""

    # Initial display with no file uploaded
    return {"display": "none"}, None, [], None, [], None, [], None, [], None, [], None, [], None, [],None,""

# Callback to display selected values
@app.callback(
    [
        Output("display-selected-value-states", "children"),
        Output("display-selected-value-regions", "children"),
        Output("display-selected-value-of-years", "children"),
        Output("region-output-of-clustering","children"),
        Output("clustering-display-selected-value-of-years","children"),
        Output("k_means_output","children"),
    ],
    [
        Input("dropdown-for-states", "value"), 
        Input("dropdown-for-regions", "value"), 
        Input("dropdown-for-years", "value"), 
        Input("region-of-clustering","value"),
        Input("clustering-dropdown-for-years","value"),
        Input("k_means_input","value"),
    ]
)
def display_selected_values(selected_state, selected_region, selected_year, clustered_region, clustered_year, clustered_input):
    state_display = f"Selected State: {selected_state}" if selected_state else "Selected State: None"
    region_display = f"Selected Region: {selected_region}" if selected_region else "Selected Region: None"
    year_display = f"Selected Year: {selected_year}" if selected_year else "Selected Year: None"
    cluster_region = f"Selected Region: {clustered_region}" if clustered_region else "Selected Region: None"
    cluster_year = f"Selected Year: {clustered_year}" if clustered_year else "Selected Year: None"
    cluster_input = f"Selected Clusters: {clustered_input}" if clustered_input else "Selected Clusters: None"
    
    return html.Div([html.P(state_display)]), html.Div([html.P(region_display)]), html.Div([html.P(year_display)]), html.Div([html.P(cluster_region)]), html.Div([html.P(cluster_year)]), html.Div([html.P(cluster_input)])
@app.callback(
    [
        Output("state-output-for-forecast", "children"),
        Output("region-output-for-forecast", "children"),
        Output("n_estimators_for_forecast", "children"),
        Output("min_samples_split_for_forecast", "children"),
        Output("criterion-for-forecast", "children"),
        Output("months-for-forecast", "children"),
    ],
    [Input("state-dropdown-for-forecast", "value"), Input("region-dropdown-for-forecast", "value"),Input("n_estimators","value"),Input("min_samples_split","value"),Input("criterion","value"),Input("months","value")]
)
def display_selected_values(selected_state, selected_region, n_estimators, min_samples_split,criterion,months):
    state_display = f"Selected State: {selected_state}" if selected_state else "Selected State: None"
    region_display = f"Selected Region: {selected_region}" if selected_region else "Selected Region: None"
    n_estimators = f"Selected N Estimators: {n_estimators}" if n_estimators else "Selected N Estimators: None"
    min_samples_split = f"Selected Minimum Sample Split: {min_samples_split}" if min_samples_split else "Selected Minimum Sample Split: None"
    criterion = f"Selected Criterion: {criterion}" if criterion else "Selected Criterion: None"
    months = f"Selected Timeline: {months} months" if months else "Selected Timeline: None"
    
    return html.Div([html.P(state_display)]), html.Div([html.P(region_display)]), html.Div([html.P(n_estimators)]), html.Div([html.P(min_samples_split)]), html.Div([html.P(criterion)]), html.Div([html.P(months)])
# Callback to generate scatter plot

@app.callback(
    [
        Output("display-eda-graph", "children"),
        Output("display-eda-numerics", "children"),
        Output("display-elbow-graph", "children"),
        Output("clustering-card", "style"),
        Output("exploratory-shower", "style")  # New output to control visibility of the div
    ],
    [
        Input("submit-btn", "n_clicks")
    ],
    [
        State("dropdown-for-states", "value"), 
        State("dropdown-for-regions", "value"), 
        State("formFileLg", "contents"),
        State("formFileLg", "filename"),
        State("dropdown-for-years", "value")
    ]
)
def display_graph_and_stats(n_clicks, selected_state, selected_region, contents,file_ka_name, selected_years):
    # Set default style to hide "exploratory-shower" and "clustering-card-elbow" divs
    exploratory_shower_style = {"display": "none"}
    clustering_card_elbow_style = {"display": "none"}
    error_message = None

    try:
        if n_clicks > 0 and contents and selected_state and selected_region:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            old_data = df.copy()
            # Clean up "Month" column by stripping any extra spaces
            df["Month"] = df["Month"].str.strip()
            
            
            # Convert "Year" and "Month" columns into a continuous "Date" column
            try:
                df["Date"] = pd.to_datetime(df["Year"].astype(str) + " " + df["Month"], format="%Y %B")
            except ValueError as e:
                error_message = f"Date conversion error: {e}"
                return None, [html.P(error_message)], None, clustering_card_elbow_style, exploratory_shower_style
            
            # Filter data based on selected region
            filtered_df = df[df['Sector'] == selected_region]
            
            # Create scatter plot with "Date" on x-axis
            fig = px.scatter(
                filtered_df,
                x="Date",
                y=selected_state,
                title=f"Scatter Plot for {selected_region} - {selected_state}",
                labels={"Date": "Date", selected_state: "Value"},
                trendline="ols"
            )
            
            # Calculate statistics
            highest_value = filtered_df[selected_state].max()
            lowest_value = filtered_df[selected_state].min()
            value_range = highest_value - lowest_value
            percentile_25th = filtered_df[selected_state].quantile(0.25)
            percentile_75th = filtered_df[selected_state].quantile(0.75)
            mean_value = filtered_df[selected_state].mean()
            median_value = filtered_df[selected_state].median()
            
            # Calculate CAGR
            start_value = filtered_df[selected_state].iloc[0]
            end_value = filtered_df[selected_state].iloc[-1]
            num_years = (filtered_df["Date"].iloc[-1] - filtered_df["Date"].iloc[0]).days / 365.25
            cagr = ((end_value / start_value) ** (1 / num_years) - 1) * 100 if num_years > 0 else None

            # Display statistics
            stats = [
                html.P(f"Range: {value_range}"),
                html.P(f"Highest Value: {highest_value}"),
                html.P(f"Lowest Value: {lowest_value}"),
                html.P(f"CAGR: {cagr:.2f}%"),
                html.P(f"25th Percentile: {percentile_25th}"),
                html.P(f"75th Percentile: {percentile_75th}"),
                html.P(f"Mean Value: {mean_value}"),
                html.P(f"Median Value: {median_value}")
            ]
            
            exploratory_shower_style = {"display": "block"}
            
            if not selected_years:
                return dcc.Graph(figure=fig), stats, None, clustering_card_elbow_style, exploratory_shower_style
            else:
                try:
                    # Filter data for selected region and year
                    
                    data = old_data[(old_data['Sector'] == selected_region) & (old_data['Year'] == selected_years)]
                    data = data.drop(['Sector', 'Year', 'Month'], axis=1).dropna(axis=1, how='all')
                    data = data.transpose()
                    data.columns = [f'Month {i+1}' for i in range(data.shape[1])]  # Rename columns as months
                    data.index.name = 'State'
                    data.reset_index(inplace=True)



                    imputer = SimpleImputer(strategy='mean') 

                    numeric_data = data.iloc[:, 1:]
                    numeric_data_imputed = imputer.fit_transform(numeric_data)
                    # Select only numeric columns for scaling
                    
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(numeric_data_imputed)
                    
                    # Perform K-means clustering and create the Elbow plot
                    inertia = []
                    K = range(1, 10)  # Test up to 10 clusters
                    for k in K:
                        kmeans = KMeans(n_clusters=k, random_state=0)
                        kmeans.fit(scaled_data)
                        inertia.append(kmeans.inertia_)

                    # Plot the Elbow Method using Plotly
                    fig_elbow = go.Figure()
                    fig_elbow.add_trace(go.Scatter(x=list(K), y=inertia, mode='lines+markers', marker=dict(color='blue')))
                    fig_elbow.update_layout(
                        title="Elbow Method for Optimal K",
                        xaxis_title="Number of Clusters",
                        yaxis_title="Inertia"
                    )

                    clustering_card_elbow_style = {"display": "block"}
                    return dcc.Graph(figure=fig), stats, dcc.Graph(figure=fig_elbow), clustering_card_elbow_style, exploratory_shower_style

                except Exception as e:
                    error_message = f"An unexpected error occurred: {e}"
                    return None, [html.P(error_message)], None, clustering_card_elbow_style, exploratory_shower_style

    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        return None, [html.P(error_message)], None, clustering_card_elbow_style, exploratory_shower_style

    return None, None, None, clustering_card_elbow_style, exploratory_shower_style
@app.callback(
    Output('forecast-options', 'style'),
    Output('clustering', 'style'),
    Input('dropdown', 'value')
)
def toggle_forecast_options(selected_value):
    if selected_value == 'forecast_cpi':
        return {'display': 'block'},{'display': 'none'}
    elif selected_value == 'classify_clusters':
        return {'display': 'none'},{'display': 'block'}
    return {'display': 'none'},{'display': 'none'}
@app.callback(
    [
        Output('forecast-shower', 'style'),
        Output('forecasted-graph', 'children'),
        Output('mape-value', 'children'),
    ],
    [
        Input('forecast-submit', 'n_clicks')    
    ],
    [
        State("formFileLg", "contents"),
        State("formFileLg", "filename"),
        State("state-dropdown-for-forecast", "value"), 
        State("region-dropdown-for-forecast", "value"),
        State("n_estimators","value"),
        State("min_samples_split","value"),
        State("criterion","value"),
        State("months","value")
    ]
)
def display_forecast_graph(n_clicks, contents, filename, state, region, n_estimators, min_samples_split, criterion, months):
    # Check if button hasn't been clicked
    if n_clicks is None or n_clicks == 0:
        return {'display': 'none'}, None, None
    
    # Check if file contents are provided
    if contents is None:
        s = html.Div(children=[html.H1('No data uploaded', className="text-warning")], className="container")
        return {}, s, None
    
    try:
        # Ensure filename is provided and decode the uploaded CSV content
        if filename is not None:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return {}, html.Div(children=[html.H1('No file uploaded', className="text-warning")], className="container"), None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return {}, html.Div(children=[html.H1('Error reading file', className="text-danger")], className="container"), None

    # Check if dataframe was loaded successfully
    if df is None:
        return {}, None, None
    
    try:
        # Filter data based on the selected 'Sector' and 'State'
        filtered_df = df[df['Sector'] == region]
        filtered_df = filtered_df[['Sector', 'Year', 'Month', state]]
        filtered_df.reset_index(drop=True, inplace=True)
        filtered_df = filtered_df.dropna()
        filtered_df['Date'] = pd.to_datetime(filtered_df['Year'].astype(str) + '-' + filtered_df['Month'] + '-01')
        filtered_df = filtered_df.sort_values('Date').set_index('Date')
        filtered_df = filtered_df[[state]].dropna()

        # Check if the filtered data has enough rows for forecasting
        if filtered_df.shape[0] < 30:
            s = html.Div(children=[html.H1('Insufficient data for the selected state', className="text-danger")], className="container")
            return {}, s, None

        # Create lag features in the filtered DataFrame
        filtered_df['Lag_1'] = filtered_df[state].shift(1)
        filtered_df['Lag_2'] = filtered_df[state].shift(2)
        filtered_df['Lag_3'] = filtered_df[state].shift(3)
        filtered_df.dropna(inplace=True)  # Drop rows with NaN values due to lagging

        # Prepare data for training
        X = filtered_df[['Lag_1', 'Lag_2', 'Lag_3']]
        y = filtered_df[state]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        # Train the Random Forest model
        model_rf = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            criterion=criterion,
            random_state=42
        )
        model_rf.fit(X_train, y_train)
        predicted = model_rf.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, predicted)

        # Forecast future values
        # Forecast future values
        input_X = X_train.iloc[-1].values.reshape(1, -1)
        future_forecast = []
        for _ in range(months):
            next_pred = model_rf.predict(pd.DataFrame(input_X, columns=X_train.columns))[0]
            future_forecast.append(next_pred)
            input_X = np.append(input_X[:, 1:], next_pred).reshape(1, -1)

        
        forecast_dates = pd.date_range(start=filtered_df.index[-1] + pd.DateOffset(months=1), periods=months, freq='ME')
        
        # Adjust lengths of actual data and forecasted data
        # Ensure the 'actual_forecast' index aligns with both past data and future forecasted dates
        actual_forecast = pd.Series(np.concatenate([y_train.values, future_forecast]), 
                                    index=filtered_df.index[-len(y_train):].append(forecast_dates))

        # Construct the combined DataFrame with aligned lengths
        combined_df = pd.DataFrame({
            'Time': actual_forecast.index,
            'Actual Data': np.concatenate([y_train.values, [np.nan] * months]),
            'Predicted Data': actual_forecast.values
        })

    except Exception as e:
        print(f"Error during data processing and forecasting: {e}")
        return {}, html.Div(children=[html.H1(f'Error during processing: {e}', className="text-danger")], className="container"), None

    # Create the forecast plot
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=combined_df['Time'], y=combined_df['Actual Data'], mode='markers', name='Actual'))
        fig.add_trace(go.Scatter(x=combined_df['Time'], y=combined_df['Predicted Data'], mode='lines', name='Forecast'))
        
        # Update the layout to add axis labels and title
        fig.update_layout(
            title=f"Forecasting for {state}",
            xaxis_title="Time",  # Label for x-axis
            yaxis_title="CPI of People"  # Label for y-axis
        )
    except Exception as e:
        print(f"Error creating the forecast plot: {e}")
        return {}, html.Div(children=[html.H1(f'Error in plot generation: {e}', className="text-danger")], className="container"), None

    return {'display': 'block'}, [dcc.Graph(figure=fig)], html.Div(className="container",children=[html.H3(style={"color": "#FF4500"}, className="mt-4",children=[f"The MAPE value for the forecast plot is {mape:.2%} (MAPE between 0-20% is considered to be good)'"])])

@app.callback(
    [
        Output('classification-shower', 'style'),
        Output('classification-graph', 'children'),
    ],
    [
        Input('classification-submit', 'n_clicks')  
    ],
    [
        State("formFileLg", "contents"),
        State("formFileLg", "filename"),
        State('region-of-clustering', "value"),
        State('clustering-dropdown-for-years', "value"),
        State("k_means_input", "value"),
    ]
)
def clustering(n_clicks, contents, filename, selected_region, selected_years, k_input):
    # Check if the button has been clicked and data is provided
    if n_clicks is None or n_clicks == 0:
        return {'display': 'none'}, None
    if contents is None:
        warning_message = html.Div(children=[html.H1('No data uploaded', className="text-warning")], className="container")
        return {}, warning_message

    try:
        # Ensure filename is provided and decode the uploaded CSV content
        if filename is not None:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            old_data = df.copy()
        else:
            warning_message = html.Div(children=[html.H1('No file uploaded', className="text-warning")], className="container")
            return {}, warning_message
    except Exception as e:
        print(f"Error decoding or reading CSV: {e}")
        error_message = html.Div(children=[html.H1('Error reading file', className="text-danger")], className="container")
        return {}, error_message

    # Check if data is valid
    if df is None:
        return {}, None

    try:
        # Filter the data based on selected region and year
        data = old_data[(old_data['Sector'] == selected_region) & (old_data['Year'] == selected_years)]
        data = data.drop(['Sector', 'Year', 'Month'], axis=1).dropna(axis=1, how='all')
        data = data.transpose()
        data.columns = [f'Month {i+1}' for i in range(data.shape[1])]  # Rename columns as months
        data.index.name = 'State'
        data.reset_index(inplace=True)
    except KeyError as e:
        print(f"Error processing data columns: {e}")
        error_message = html.Div(children=[html.H1(f"Missing column: {e}", className="text-danger")], className="container")
        return {}, error_message
    except Exception as e:
        print(f"Unexpected error during data processing: {e}")
        return {}, html.Div(children=[html.H1('Error processing data', className="text-danger")], className="container")

    try:
        # Impute missing values and scale the data
        imputer = SimpleImputer(strategy='mean')
        numeric_data = data.iloc[:, 1:]
        numeric_data_imputed = imputer.fit_transform(numeric_data)
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data_imputed)
    except Exception as e:
        print(f"Error during data imputation or scaling: {e}")
        return {}, html.Div(children=[html.H1('Error during data processing', className="text-danger")], className="container")

    try:
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=k_input, random_state=0)
        data['Cluster'] = kmeans.fit_predict(scaled_data)

        # Assign class labels based on clusters
        class_labels = {i: f"Class {i+1}" for i in range(k_input)}
        data['Class Label'] = data['Cluster'].map(class_labels)
    except ValueError as e:
        print(f"Error in KMeans clustering with k={k_input}: {e}")
        return {}, html.Div(children=[html.H1('Invalid cluster number', className="text-danger")], className="container")
    except Exception as e:
        print(f"Unexpected error during clustering: {e}")
        return {}, html.Div(children=[html.H1('Error during clustering', className="text-danger")], className="container")

    try:
        # Reduce dimensions with PCA for visualization
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        centroids_pca = pca.transform(kmeans.cluster_centers_)

        # Create plotly figure for clusters
        fig_clusters = go.Figure()
        cluster_colors = px.colors.qualitative.Plotly

        for cluster in range(k_input):
            cluster_data = pca_data[data['Cluster'] == cluster]
            cluster_state_names = data['State'][data['Cluster'] == cluster]
            centroid = centroids_pca[cluster]

            # Plot cluster points
            fig_clusters.add_trace(go.Scatter(
                x=cluster_data[:, 0], y=cluster_data[:, 1],
                mode='markers',
                marker=dict(size=8, color=cluster_colors[cluster]),
                name=f'{class_labels[cluster]}',
                text=cluster_state_names,
                hovertemplate='State: %{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
            ))

            # Plot centroids
            fig_clusters.add_trace(go.Scatter(
                x=[centroid[0]], y=[centroid[1]],
                mode='markers',
                marker=dict(size=15, color=cluster_colors[cluster], symbol='x'),
                name=f'{class_labels[cluster]} Centroid',
                hoverinfo='skip'
            ))

        fig_clusters.update_layout(
            title=f"KMeans Clustering of States by CPI ({selected_region}, {selected_years})",
            xaxis_title="Principal Component 1 (PCA)",
            yaxis_title="Principal Component 2 (PCA)",
            showlegend=True
        )

        return {"display": "block"}, dcc.Graph(figure=fig_clusters)

    except Exception as e:
        print(f"Error during PCA or plotting: {e}")
        return {}, html.Div(children=[html.H1('Error generating plot', className="text-danger")], className="container")
        
if __name__ == "__main__":
    app.run_server(debug=False, port = 8010)
