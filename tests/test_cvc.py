import pandas as pd

from figures.cvc import CVCPlotFactory
from figures.utils_remiss import get_all_values_users, convert_dict_to_dataframe, load_medians_file_all_features


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


def test_median_files():
    cvc_plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
    actual = cvc_plot_factory.load_medians_file_all_features("../cvc_data/results_7_fake_spreaders_cat").sort_index()
    expected = load_medians_file_all_features("../cvc_data/results_7_fake_spreaders_cat")['AllValues']
    expected = pd.Series(expected, name='median').sort_index()
    assert actual == expected


def test_load_data_for_user():
    cvc_plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
    user_data = cvc_plot_factory.load_data_for_user('CVCFeatures2', '100485425')
    assert user_data['twitter_id'] == '100485425'
    assert user_data['name'] == 'Santaflow'


def test_plot_user_info():
    assert False


def test__get_plot_data():
    assert False


def test_plot_vertical_barplot_topics():
    assert False


def test_plot_radarplot_emotions():
    assert False


def test_plot_vertical_accumulated_barplot_by_genre():
    assert False


def test_plot_vertical_accumulated_barplot_by_age():
    assert False


def test_plot_vertical_barplot_polarity():
    assert False


def test_plot_horizontal_bars_plot_interactions():
    assert False


def test_plot_donut_plot_behaviour():
    assert False
