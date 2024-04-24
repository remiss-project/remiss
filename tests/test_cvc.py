from figures.cvc import CVCPlotFactory
from figures.utils_remiss import get_all_values_users, convert_dict_to_dataframe


def test_init():
    cvc_plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
    fake_spreaders_feats, fact_checkers_feats, random_feats = get_all_values_users('ca')

    # Convert feature values to dataframes
    fake_news_spreaders = convert_dict_to_dataframe(fake_spreaders_feats)
    fact_checkers = convert_dict_to_dataframe(fact_checkers_feats)
    control_cases = convert_dict_to_dataframe(random_feats)
    assert cvc_plot_factory.host == "localhost"
    assert cvc_plot_factory.port == 27017
    assert cvc_plot_factory.database == "CVCUI2"
    assert cvc_plot_factory._available_datasets is None
    assert cvc_plot_factory.fake_news_spreaders.equals(fake_news_spreaders)
    assert cvc_plot_factory.fact_checkers.equals(fact_checkers)
    assert cvc_plot_factory.control_cases.equals(control_cases)


def test_load_data_for_user():
    cvc_plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
    user_data = cvc_plot_factory.load_data_for_user('CVCFeatures2', '100485425')
    assert user_data['twitter_id'] == '100485425'
    assert user_data['description'] == 'basado y sin complejos. Publi al DM CONTRATACIÓN: mas.sabor.estudios@gmail.com'


def test_plot_user_info():
    cvc_plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
    plot_user_info = cvc_plot_factory.plot_user_info('CVCFeatures2', '100485425')
    plot_user_info.show()
    assert len(plot_user_info.layout.annotations) == 7


def test_plot_vertical_barplot_topics():
    cvc_plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
    plot_vertical_barplot_topics = cvc_plot_factory.plot_vertical_barplot_topics('CVCFeatures2', '100485425')
    plot_vertical_barplot_topics.show()
    assert len(plot_vertical_barplot_topics.data) == 4


def test_plot_radarplot_emotions():
    cvc_plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
    plot_radarplot_emotions = cvc_plot_factory.plot_radarplot_emotions('CVCFeatures2', '100485425')
    plot_radarplot_emotions.show()
    assert len(plot_radarplot_emotions.data) == 4


def test_plot_vertical_accumulated_barplot_by_genre():
    cvc_plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
    plot_vertical_accumulated_barplot_by_genre = cvc_plot_factory.plot_vertical_accumulated_barplot_by_genre()
    plot_vertical_accumulated_barplot_by_genre.show()
    assert len(plot_vertical_accumulated_barplot_by_genre.data) == 3


def test_plot_vertical_accumulated_barplot_by_age():
    cvc_plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
    plot_vertical_accumulated_barplot_by_age = cvc_plot_factory.plot_vertical_accumulated_barplot_by_age()
    plot_vertical_accumulated_barplot_by_age.show()
    assert len(plot_vertical_accumulated_barplot_by_age.data) == 4


def test_plot_vertical_barplot_polarity():
    cvc_plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
    plot_vertical_barplot_polarity = cvc_plot_factory.plot_vertical_barplot_polarity('CVCFeatures2', '100485425')
    plot_vertical_barplot_polarity.show()
    assert len(plot_vertical_barplot_polarity.data) == 4


def test_plot_horizontal_bars_plot_interactions():
    cvc_plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
    horizontal_bars_plot_1_py_interactions_1, horizontal_bars_plot_2_py_interactions_2 = cvc_plot_factory.plot_horizontal_bars_plot_interactions(
        'CVCFeatures2', '100485425')
    horizontal_bars_plot_1_py_interactions_1.show()
    horizontal_bars_plot_2_py_interactions_2.show()
    assert len(horizontal_bars_plot_1_py_interactions_1.data) == 4
    assert len(horizontal_bars_plot_2_py_interactions_2.data) == 4


def test_plot_donut_plot_behaviour():
    cvc_plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
    donut_plot_py_behavior_1, donut_plot_py_behavior_2 = cvc_plot_factory.plot_donut_plot_behaviour('CVCFeatures2',
                                                                                                    '100485425')
    donut_plot_py_behavior_1.show()
    donut_plot_py_behavior_2.show()
    assert len(donut_plot_py_behavior_1.data) == 4
    assert len(donut_plot_py_behavior_2.data) == 4
