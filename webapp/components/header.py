import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

def Header():


    navbar = dbc.Navbar(
        [
            get_logo(),
            get_menu(),
            dbc.NavbarToggler(id="navbar-toggler"),
            # dbc.Collapse(search_bar, id="navbar-collapse", navbar=True),
        ],
        color="dark",
        dark=True,
        # fixed='top',
        sticky='top',
    )

    return navbar

def get_logo():
    logo = html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src='../assets/favicon.ico', height="30px")),
                    dbc.Col(dbc.NavbarBrand("LH Neuroscience", className="ml-2")),
                ],
                align="center",
                no_gutters=True,
            ),
            href="/",
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
    nav = dbc.Nav([
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.NavItem(dbc.NavLink("ConvDip", href="/convdip/")),
        dbc.NavItem(dbc.NavLink("Publications", href="/publications/")),
        # dbc.DropdownMenu(
        #     [dbc.DropdownMenuItem("Item 1"), dbc.DropdownMenuItem("Item 2")],
        #     label="Dropdown",
        #     nav=True,
        # ),
        ])

    return nav