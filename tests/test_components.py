import unittest
from datetime import datetime
from unittest import TestCase
from unittest.mock import Mock

from contextvars import copy_context
from dash._callback_context import context_value
from dash._utils import AttributeDict

import dash
from dash import Dash, dcc
from dash.dcc import DatePickerRange, Graph, Dropdown, Slider
from dash_holoniq_wordcloud import DashWordcloud

from components import DashComponent, TweetUserTimeSeriesComponent, EgonetComponent, RemissDashboard


class TestDashComponent(TestCase):
    def test_name(self):
        class TestComponent(DashComponent):
            def __init__(self, name=None):
                super().__init__(name=name)

        component = TestComponent()
        self.assertEqual(len(component.name), 10)


class TweetUserTimeSeriesComponentTest(TestCase):
    def setUp(self):
        self.plot_factory = Mock()
        self.plot_factory.available_datasets = ['dataset1', 'dataset2', 'dataset3']
        self.plot_factory.get_date_range.return_value = (datetime(2023, 1, 1),
                                                         datetime(2023, 12, 31))
        self.plot_factory.plot_tweet_series.return_value = 'plot_tweet_series'
        self.plot_factory.plot_user_series.return_value = 'plot_user_series'
        self.plot_factory.get_hashtag_freqs.return_value = [('hashtag1', 10), ('hashtag2', 5),
                                                            ('hashtag3', 3), ('hashtag4', 2),
                                                            ('hashtag5', 1), ('hashtag6', 1), ]
        self.component = TweetUserTimeSeriesComponent(self.plot_factory,
                                                      dataset_dropdown=Mock(id='dataset-dropdown'),
                                                      date_picker=Mock(id='date-picker'),
                                                      wordcloud=Mock(id='wordcloud'),
                                                      top_table=Mock(id='top-table'))

    def test_layout(self):
        layout = self.component.layout()

        # check that among the final components of the layout we have:
        # - a DatePickerRange
        # - a Wordcloud
        # - a TweetUserTimeSeriesComponent
        # find components recursively
        def find_components(component, found_components):
            if hasattr(component, 'children'):
                for child in component.children:
                    find_components(child, found_components)
            if isinstance(component, Graph):
                found_components.append(component)

        found_components = []
        find_components(layout, found_components)
        found_components = [type(component) for component in found_components]
        self.assertIn(Graph, found_components)

    def test_layout_ids(self):
        layout = self.component.layout()

        # check that among the ids are correctly patched
        # find components recursively
        def find_components(component, found_components):
            if hasattr(component, 'children'):
                for child in component.children:
                    find_components(child, found_components)
            if isinstance(component, DatePickerRange):
                found_components.append(component)
            if isinstance(component, DashWordcloud):
                found_components.append(component)
            if isinstance(component, Graph):
                found_components.append(component)

        found_components = []
        find_components(layout, found_components)
        component_ids = ['-'.join(component.id.split('-')[:-1]) for component in found_components]
        self.assertIn('fig-tweet', component_ids)
        self.assertIn('fig-users', component_ids)
        found_main_ids = ['-'.join(component.id.split('-')[-1:]) for component in found_components]
        self.assertIn(self.component.name, found_main_ids)
        self.assertEqual(len(set(found_main_ids)), 1)

    def test_update_date_range_callback(self):
        app = Dash()
        self.component.callbacks(app)

        # Simulate the update function for the date range
        date_range_key = (f'..date-picker-{self.component.name}.min_date_allowed...'
                          f'date-picker-{self.component.name}.max_date_allowed...'
                          f'date-picker-{self.component.name}.start_date...'
                          f'date-picker-{self.component.name}.end_date..')
        callback = app.callback_map[date_range_key]
        self.assertEqual(callback['inputs'], [{'id': f'dataset-dropdown-{self.component.name}', 'property': 'value'}])
        expected_outputs = [f'date-picker-{self.component.name}.' + field for field in
                            ['min_date_allowed', 'max_date_allowed', 'start_date', 'end_date']]
        actual_outputs = [output.component_id + '.' + output.component_property for output in callback['output']]
        self.assertEqual(actual_outputs, expected_outputs)
        actual = self.component.update_date_picker('dataset2')

        self.assertEqual(self.plot_factory.get_date_range.call_args[0][0], 'dataset2')
        expected = (datetime(2023, 1, 1), datetime(2023, 12, 31), datetime(2023, 1, 1), datetime(2023, 12, 31))
        self.assertEqual(actual, expected)

    def test_update_wordcloud_callback(self):
        app = Dash()
        self.component.callbacks(app)

        # Simulate the update function for the wordcloud
        wordcloud_key = f'wordcloud-{self.component.name}.list'
        callback = app.callback_map[wordcloud_key]
        self.assertEqual(callback['inputs'], [{'id': f'dataset-dropdown-{self.component.name}', 'property': 'value'}])
        self.assertEqual(callback['output'].component_id, f'wordcloud-{self.component.name}')
        self.assertEqual(callback['output'].component_property, 'list')
        actual = self.component.update_wordcloud('dataset2')
        self.assertEqual(self.plot_factory.get_hashtag_freqs.call_args[0][0], 'dataset2')
        self.assertEqual(actual, self.plot_factory.get_hashtag_freqs.return_value)

    def test_update_plots_callback(self):
        app = Dash()
        self.component.callbacks(app)

        # Simulate the update function for the plots
        plots_key = f'..fig-tweet-{self.component.name}.figure...fig-users-{self.component.name}.figure..'
        callback = app.callback_map[plots_key]
        self.assertEqual(callback['inputs'], [{'id': f'dataset-dropdown-{self.component.name}', 'property': 'value'},
                                              {'id': f'date-picker-{self.component.name}', 'property': 'start_date'},
                                              {'id': f'date-picker-{self.component.name}', 'property': 'end_date'},
                                              {'id': f'wordcloud-{self.component.name}', 'property': 'click'}])
        expected_outputs = [f'fig-tweet-{self.component.name}.figure',
                            f'fig-users-{self.component.name}.figure']
        actual_outputs = [output.component_id + '.' + output.component_property for output in callback['output']]
        self.assertEqual(actual_outputs, expected_outputs)
        actual = self.component.update_plots('dataset2', datetime(2023, 1, 1),
                                             datetime(2023, 12, 31), ['hashtag1', 10])
        expected = (self.plot_factory.plot_tweet_series.return_value,
                    self.plot_factory.plot_user_series.return_value)
        self.assertEqual(actual, expected)

    # def test_wordcloud_lots_of_hashtags(self):
    #     hashtags = [('26M', 420005), ('Barcelona', 299912), ('HazQuePase', 170053), ('28A', 142114),
    #                 ('LaEspañaQueQuieres', 61550), ('VotaPSOE', 55658), ('barcelona', 53418), ('PSOE', 26353),
    #                 ('Liverpool', 25380), ('España', 20002), ('26m', 18019), ('PSOEPonienteSur', 17799),
    #                 ('Madrid', 16674), ('municipalscat', 16594), ('EstamosMuyCerca', 16271), ('ChampionsLeague', 16101),
    #                 ('EspañaVaciada', 14613), ('VOTAPSOE', 14336), ('VaDeLlibertat', 12154),
    #                 ('SiempreHaciaDelante', 12016), ('Messi', 11980), ('110CompromisosPSOE', 10390), ('Vox', 10347),
    #                 ('CórdobaEsp', 10214), ('Elecciones26M', 10108), ('CórdobaEnMarcha', 9957),
    #                 ('IsabelAmbrosio', 9810), ('UCL', 9169), ('CórdobaESP', 8053), ('PP', 7992),
    #                 ('ParaLaGranMayoria', 7620), ('Europa', 7487), ('28Abril', 7246), ('empleo', 6841),
    #                 ('ELDEBATEenRTVE', 6801), ('RT', 6727), ('BARCELONA', 6467), ('Sevilla', 6201), ('Catalunya', 6018),
    #                 ('28AVotaPSOE', 5915), ('LIVBAR', 5764), ('Municipals2019BCN', 5759), ('desaparecido', 5556),
    #                 ('Sabadell', 5549), ('Spain', 5502), ('spain', 5448), ('FesQuePassi', 5385), ('votaPSOE', 5252),
    #                 ('trabajo', 5211), ('HazquePase', 5055), ('EleccionesMunicipales', 5037), ('Valencia', 4945),
    #                 ('debates', 4898), ('empleobarcelona', 4651), ('PedroPresidente', 4630), ('BCN', 4518),
    #                 ('Badalona', 4469), ('LaLiga', 4350), ('hazquepase', 4291), ('LaEuropaQueQuieres', 4284),
    #                 ('ManchesterUnited', 4118), ('LFC', 4015), ('liverpool', 4004), ('Zaragoza', 3922), ('messi', 3849),
    #                 ('travel', 3807), ('1Oct', 3726), ('Barca', 3671), ('ElDebateDecisivo', 3635), ('Ajax', 3617),
    #                 ('URGENTE', 3514), ('EspañaViva', 3464), ('VadeLlibertat', 3453), ('Girona', 3411), ('bcn', 3377),
    #                 ('EleccionesMunicipales2019', 3360), ('Santander', 3278), ('Canarias', 3149), ('JuntsxCat', 3135),
    #                 ('elecciones', 3116), ('Hazquepase', 3096), ('21D', 3086), ('championsleague', 3057),
    #                 ('UnidasPodemos', 3050), ('LHospitalet', 3038), ('madrid', 3010), ('Terrassa', 2949),
    #                 ('Tarragona', 2901), ('Anzoátegui', 2890), ('football', 2836), ('Carmena', 2832),
    #                 ('MásMadrid', 2808), ('28a', 2803), ('Lleida', 2785), ('VamosCiudadanos', 2779), ('concert', 2777),
    #                 ('CentradosEnTuFuturo', 2765), ('VOX', 2750), ('catalunya', 2682), ('Cantabria', 2637)]
    #
    #     min_value = min([x[1] for x in hashtags])
    #     print(min_value)
    #     wordcloud = DashWordcloud(width=1200, height=400,
    #                               shape='circle',
    #                               hover=True,
    #                               id=f'wordcloud',
    #                               weightFactor=10 / min_value,
    #                               list=hashtags)
    #     app = Dash()
    #     app.layout = wordcloud
    #     wordcloud.list = hashtags
    #     app.run_server(debug=True, use_reloader=False)

    def test_wordclould_max_hashtags(self):
        self.component.max_wordcloud_words = 2
        layout = self.component.layout()
        wordcloud = layout.children[0].children[0].children[0]
        self.assertEqual(len(wordcloud.list), 2)


class EgonetComponentTest(TestCase):
    def setUp(self):
        self.plot_factory = Mock()
        self.plot_factory.available_datasets = ['dataset1', 'dataset2', 'dataset3']
        self.plot_factory.get_date_range.return_value = (datetime(2023, 1, 1),
                                                         datetime(2023, 12, 31))
        self.plot_factory.get_users.return_value = ['user1', 'user2', 'user3']
        self.component = EgonetComponent(self.plot_factory,
                                         dataset_dropdown=Mock(id='dataset-dropdown'),
                                         top_table=Mock(id='top-table'))

    def test_layout(self):
        layout = self.component.layout()

        # check that among the final components of the layout we have:
        # - a Dropdown
        # - a Slider
        # - a graph
        # find components recursively
        def find_components(component, found_components):
            if hasattr(component, 'children'):
                for child in component.children:
                    find_components(child, found_components)
            if isinstance(component, Dropdown):
                found_components.append(component)
            if isinstance(component, Slider):
                found_components.append(component)
            if isinstance(component, Graph):
                found_components.append(component)

        found_components = []
        find_components(layout, found_components)
        found_components = [type(component) for component in found_components]
        self.assertIn(Dropdown, found_components)
        self.assertIn(Slider, found_components)
        self.assertIn(Graph, found_components)

    def test_layout_ids(self):
        layout = self.component.layout()

        # check that among the ids are correctly patched
        # find components recursively
        def find_components(component, found_components):
            if hasattr(component, 'children'):
                for child in component.children:
                    find_components(child, found_components)
            if isinstance(component, Dropdown):
                found_components.append(component)
            if isinstance(component, Slider):
                found_components.append(component)
            if isinstance(component, Graph):
                found_components.append(component)

        found_components = []
        find_components(layout, found_components)
        component_ids = ['-'.join(component.id.split('-')[:-1]) for component in found_components]
        self.assertIn('user-dropdown', component_ids)
        self.assertIn('slider', component_ids)
        self.assertIn('fig', component_ids)
        found_main_ids = ['-'.join(component.id.split('-')[-1:]) for component in found_components]
        self.assertIn(self.component.name, found_main_ids)
        self.assertEqual(len(set(found_main_ids)), 1)

    def test_update_egonet_callback(self):
        app = Dash()
        self.component.callbacks(app)

        # Simulate the update function for the plots
        plots_key = f'fig-{self.component.name}.figure'
        callback = app.callback_map[plots_key]
        self.assertEqual(callback['inputs'],
                         [{'id': 'dataset-dropdown', 'property': 'value'},
                          {'id': f'user-dropdown-{self.component.name}', 'property': 'value'},
                          {'id': 'top-table', 'property': 'active_cell'},
                          {'id': f'slider-{self.component.name}', 'property': 'value'}])
        self.assertEqual(callback['output'].component_id, f'fig-{self.component.name}')
        self.assertEqual(callback['output'].component_property, 'figure')

        def run_callback():
            context_value.set(AttributeDict(**{"triggered_id": "top-table"}))
            actual = self.component.update_user_dropdown('dataset2', 'user2', None, 3)
            return actual

        ctx = copy_context()
        actual = ctx.run(run_callback)

        self.assertEqual(self.plot_factory.plot_egonet.call_args[0][0], 'dataset2')
        self.assertEqual(self.plot_factory.plot_egonet.call_args[0][1], 'user2')
        self.assertEqual(self.plot_factory.plot_egonet.call_args[0][2], 3)
        self.assertEqual(actual, self.plot_factory.plot_egonet.return_value)


class RemissDashboardTest(TestCase):
    def setUp(self):
        self.tweet_user_plot_factory = Mock()
        self.tweet_user_plot_factory.available_datasets = ['dataset1', 'dataset2', 'dataset3']
        self.tweet_user_plot_factory.get_date_range.return_value = (datetime(2023, 1, 1),
                                                                    datetime(2023, 12, 31))
        self.tweet_user_plot_factory.get_users.return_value = ['user1', 'user2', 'user3']
        self.tweet_user_plot_factory.get_hashtag_freqs.return_value = [('hashtag1', 10), ('hashtag2', 5),
                                                                       ('hashtag3', 3), ('hashtag4', 2),
                                                                       ('hashtag5', 1), ('hashtag6', 1), ]
        self.egonet_plot_factory = Mock()
        self.egonet_plot_factory.available_datasets = ['dataset1', 'dataset2', 'dataset3']
        self.egonet_plot_factory.get_date_range.return_value = (datetime(2023, 1, 1),
                                                                datetime(2023, 12, 31))
        self.egonet_plot_factory.get_users.return_value = ['user1', 'user2', 'user3']
        self.top_table_factory = Mock()
        self.top_table_factory.available_datasets = ['dataset1', 'dataset2', 'dataset3']
        self.top_table_factory.get_date_range.return_value = (datetime(2023, 1, 1),
                                                              datetime(2023, 12, 31))
        self.top_table_factory.get_users.return_value = ['user1', 'user2', 'user3']
        self.top_table_factory.retweeted_table_columns = ['id', 'text', 'user']
        self.top_table_factory.user_table_columns = ['id', 'name', 'screen_name']
        self.top_table_factory.top_table_columns = ['User', 'Text', 'Retweets', 'Is usual suspect', 'Party']

        self.component = RemissDashboard(self.tweet_user_plot_factory, self.top_table_factory, self.egonet_plot_factory)

    def test_layout(self):
        layout = self.component.layout()

        # check that among the final components of the layout we have:
        # - a Dropdown
        # - a TweetUserTimeSeriesComponent
        # - a EgonetComponent
        # find components recursively
        def find_components(component, found_components):
            if hasattr(component, 'children'):
                for child in component.children:
                    find_components(child, found_components)
            if isinstance(component, Dropdown):
                found_components.append(component)
            if isinstance(component, Slider):
                found_components.append(component)
            if isinstance(component, Graph):
                found_components.append(component)
            if isinstance(component, DatePickerRange):
                found_components.append(component)
            if isinstance(component, DashWordcloud):
                found_components.append(component)

        found_components = []
        find_components(layout, found_components)
        found_components = [type(component) for component in found_components]
        self.assertIn(Dropdown, found_components)
        self.assertIn(Graph, found_components)
        self.assertIn(DashWordcloud, found_components)
        self.assertIn(DatePickerRange, found_components)
        self.assertIn(Slider, found_components)

    def test_layout_ids(self):
        layout = self.component.layout()

        # check that among the ids are correctly patched
        # find components recursively
        def find_components(component, found_components):
            if hasattr(component, 'children'):
                for child in component.children:
                    find_components(child, found_components)
            if isinstance(component, Dropdown):
                found_components.append(component)
            if isinstance(component, Slider):
                found_components.append(component)
            if isinstance(component, Graph):
                found_components.append(component)
            if isinstance(component, DatePickerRange):
                found_components.append(component)
            if isinstance(component, DashWordcloud):
                found_components.append(component)

        found_components = []
        find_components(layout, found_components)
        component_ids = ['-'.join(component.id.split('-')[:-1]) for component in found_components]
        self.assertIn(f'dataset-dropdown', component_ids)
        self.assertIn('date-picker', component_ids)
        self.assertIn('wordcloud', component_ids)
        self.assertIn('fig-tweet', component_ids)
        self.assertIn('fig-users', component_ids)
        self.assertIn('user-dropdown', component_ids)
        self.assertIn('slider', component_ids)
        self.assertIn('fig', component_ids)

    def test_dataset_storage(self):
        def run_callback():
            context_value.set(AttributeDict(**{"triggered_id": "dataset-dropdown"}))
            actual = self.component.update_dataset_storage('dataset2')
            return actual

        ctx = copy_context()
        actual = ctx.run(run_callback)
        self.assertEqual(self.component.current_dataset['dataset'], 'dataset2')
    def test_on_dataset_change_top_table(self):
        pass

    def test_on_dataset_change_egonet_plot(self):
        pass


if __name__ == '__main__':
    unittest.main()
