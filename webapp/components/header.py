import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

def Header():
    return html.Div([
        dbc.Row([get_menu(), get_header(), get_logo()], justify="between")
        # get_logo(),
        # get_header(),
        # html.Br([]),
        # get_menu()
    ], style={'margin': '20px'})

def get_logo():
    logo = dbc.Col(

        html.Div([

        html.Div([
            html.Img(src='../assets/favicon.ico', height='128', width='128')
        ], className="ten columns padded"),

        # html.Div([
        #     dcc.Link('Full View   ', href='/cc-travel-report/full-view')
        # ], className="two columns page-view no-print")

    ], className="row gs-header"),
    width=1
    )
    return logo


def get_header():
    header = dbc.Col(
        html.Div([

            html.Div([
                html.H2('Play with MEEG')
            ], className="twelve columns padded")

        ], className="row gs-header gs-text-header"),
        width=3
    )
    return header


# def get_menu():
#     menu = html.Div([

#         dcc.Link('Main', href='/', className="tab first"),
#         dcc.Link('ConvDip', href='/convdip/', className="tab"),

#     ], className="row ")
#     return menu

def get_menu():
    nav = dbc.Col(
        dbc.Nav([
        dbc.NavItem(dbc.NavLink("Main", href="/")),
        dbc.NavItem(dbc.NavLink("ConvDip", href="/convdip/")),
        dbc.NavItem(dbc.NavLink("Another link", href="/some_site/")),
        # dbc.DropdownMenu(
        #     [dbc.DropdownMenuItem("Item 1"), dbc.DropdownMenuItem("Item 2")],
        #     label="Dropdown",
        #     nav=True,
        # ),
        ]
        ),
        width=3,
    )
    return nav