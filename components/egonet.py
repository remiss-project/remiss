import hashlib
import logging

import dash_bootstrap_components as dbc
from dash import dcc, Input, Output, State
from dash.exceptions import PreventUpdate

from components.components import RemissComponent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class EgonetComponent(RemissComponent):
    def __init__(self, plot_factory, state, name=None, debug=False, anonymous=False):
        super().__init__(name=name)
        self.anonymous = anonymous
        self.debug = debug
        self.dates = None
        self.plot_factory = plot_factory
        self.state = state
        self.available_datasets = plot_factory.available_datasets
        self.graph_egonet = dcc.Graph(figure={}, id=f'fig-{self.name}',
                                      # config={'displayModeBar': False},
                                      responsive=True,
                                      style={'height': '100%', 'width': '100%'},
                                      )
        self.depth_slider = dcc.Slider(min=1, max=5, step=1, value=1, id=f'slider-{self.name}')

    def layout(self, params=None):
        return dbc.Card([
            dbc.CardHeader('Filtered Network', 
                           id=f'title-{self.name}', 
                           style={'fontSize': '18px', 'fontWeight': 'bold'}),
            dbc.CardBody([
                dcc.Loading(id=f'loading-{self.name}',
                            type='default',
                            children=self.graph_egonet,
                            style={'height': '100%'}
                            ),
            ], style={'height': '100%'}),
            dbc.Collapse([
                dbc.CardFooter([
                    dbc.Row([
                        dbc.Col([
                            self.depth_slider
                        ], width=6),
                    ]),
                ])
            ], id=f'collapse-depth-slider-{self.name}', is_open=False),
        ], style={'height': '100%'})

    def update(self, dataset, user, start_date, end_date, hashtags, depth):
        logger.debug(f'Updating egonet with dataset {dataset}, user {user}, '
                     f'start date {start_date}, end date {end_date}, hashtag {hashtags}, depth {depth}')

        # Show egonet for the selected user
        try:
            if user is None:
                raise ValueError('User is None')
            fig = self.plot_factory.plot_egonet(dataset, user, depth, start_date, end_date, hashtags)
            try:
                username = self.plot_factory.get_username(dataset, user)
            except RuntimeError:
                username = user
            if self.anonymous:
                username = hashlib.md5(username.encode()).hexdigest()[:8]
            title = f'Egonet for {username}'
            show_depth_slider = True
            logger.debug(f'Plotting egonet for user {username} with id {user}')
        except (RuntimeError, ValueError, KeyError) as e:
            # If the user is not available, then show the backbone

            fig = self.plot_factory.plot_hidden_network(dataset, start_date=start_date, end_date=end_date,
                                                        hashtags=hashtags)
            show_depth_slider = False
            title = 'Filtered Hidden Network'
            logger.debug(f'User {user} not available, plotting backbone')

        # remove margin
        # fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        return fig, title, show_depth_slider, not show_depth_slider

    def update_user_storage(self, click_data, dataset):
        if click_data is not None:
            try:
                text = click_data['points'][0]['text']
                user = text.split('<br>')[0].split(':')[1].strip()
                author_id = self.plot_factory.get_user_id(dataset, user)
                logger.debug(f'Updating user state with {user} with author id {author_id}')
                return author_id
            except KeyError:
                logger.debug(f'Error updating user without metadata from click: {click_data}')
                raise PreventUpdate()
            except Exception as e:
                logger.error(f'Error updating user storage from click: {e}')
                raise PreventUpdate()
        raise PreventUpdate()

    def callbacks(self, app):
        app.callback(
            Output(f'fig-{self.name}', 'figure'),
            Output(f'title-{self.name}', 'children'),
            Output(f'collapse-depth-slider-{self.name}', 'is_open'),
            Output(self.depth_slider, 'disabled'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_user, 'data'),
             Input(self.state.current_start_date, 'data'),
             Input(self.state.current_end_date, 'data'),
             Input(self.state.current_hashtags, 'data'),
             Input(self.depth_slider, 'value'),
             ]
        )(self.update)

        app.callback(
            Output(self.state.current_user, 'data', allow_duplicate=True),
            [Input(self.graph_egonet, 'clickData'),
             State(self.state.current_dataset, 'data')],
        )(self.update_user_storage)
