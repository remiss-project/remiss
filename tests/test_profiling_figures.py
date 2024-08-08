from matplotlib import pyplot as plt
from pymongo import MongoClient
from pymongoarrow.monkey import patch_all
from tqdm import tqdm

from figures.profiling import ProfilingPlotFactory
from figures.utils_remiss import get_all_values_users, convert_dict_to_dataframe

patch_all()


def test_init():
    cvc_plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
    fake_spreaders_feats, fact_checkers_feats, random_feats = get_all_values_users('ca')

    # Convert feature values to dataframes
    fake_news_spreaders = convert_dict_to_dataframe(fake_spreaders_feats)
    fact_checkers = convert_dict_to_dataframe(fact_checkers_feats)
    control_cases = convert_dict_to_dataframe(random_feats)
    assert cvc_plot_factory.host == "localhost"
    assert cvc_plot_factory.port == 27017
    assert cvc_plot_factory._available_datasets is None
    assert cvc_plot_factory.fake_news_spreaders.equals(fake_news_spreaders)
    assert cvc_plot_factory.fact_checkers.equals(fact_checkers)
    assert cvc_plot_factory.control_cases.equals(control_cases)


def test_histogram_donut1():
    client = MongoClient('localhost', 27017)
    database = client.get_database('test_dataset_2')
    collection = database.get_collection('profiling')
    feature_names = ['week_days_count_ratio_behav', 'weekend_days_count_ratio_behav', 'tweets_sleep_time_ratio_behav',
                     'tweets_awake_time_ratio_behav']
    for feature in feature_names:
        pipeline = [
            {'$group': {'_id': f'${feature}', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}},
        ]
        df = collection.aggregate_pandas_all(pipeline)
        df.plot(kind='bar', x='_id', y='count', title=feature, color='blue')
        plt.savefig(f'{feature}.png')
    plt.show()


def test_histogram_radaplot():
    client = MongoClient('localhost', 27017)
    database = client.get_database('test_dataset_2')
    collection = database.get_collection('profiling')
    feature_names = ['joy_emolex', 'trust_emolex', 'fear_emolex', 'surprise_emolex', 'sadness_emolex', 'disgust_emolex',
                     'anger_emolex', 'anticipation_emolex']
    for feature in feature_names:
        pipeline = [
            {'$project': {feature: 1}},
        ]
        df = collection.aggregate_pandas_all(pipeline)
        print(df.describe())
    plt.show()


def test_load_data_for_user():
    cvc_plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
    user_data = cvc_plot_factory.load_data_for_user('test_dataset_2', '1442274168')
    assert user_data['twitter_id'] == '1442274168'
    assert user_data[
               'description'] == 'Amo mi país, si quieres romperlo eres mi adversario. 🇪🇸💚👍🏼. #simesiguestesigo'


def test_plot_user_info():
    cvc_plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
    plot_user_info = cvc_plot_factory.plot_user_info('test_dataset_2', '1442274168')
    plot_user_info.show()
    assert len(plot_user_info.layout.annotations) == 7


def test_plot_vertical_barplot_topics():
    cvc_plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
    plot_vertical_barplot_topics = cvc_plot_factory.plot_vertical_barplot_topics('test_dataset_2', '1442274168')
    plot_vertical_barplot_topics.show()
    assert len(plot_vertical_barplot_topics.data) == 4


def test_plot_radarplot_emotions():
    cvc_plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
    plot_radarplot_emotions = cvc_plot_factory.plot_radarplot_emotions('test_dataset_2', '1442274168')
    plot_radarplot_emotions.show()
    assert len(plot_radarplot_emotions.data) == 4


def test_plot_all_users():
    cvc_plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
    client = MongoClient('localhost', 27017)
    database = client.get_database('test_dataset_2')
    collection = database.get_collection('profiling')
    user_ids = collection.distinct('twitter_id')
    for user_id in tqdm(user_ids):
        cvc_plot_factory.plot_radarplot_emotions('test_dataset_2', user_id).show()


def test_plot_vertical_accumulated_barplot_by_genre():
    cvc_plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
    plot_vertical_accumulated_barplot_by_genre = cvc_plot_factory.plot_vertical_accumulated_barplot_by_genre()
    plot_vertical_accumulated_barplot_by_genre.show()
    assert len(plot_vertical_accumulated_barplot_by_genre.data) == 3


def test_plot_vertical_accumulated_barplot_by_age():
    cvc_plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
    plot_vertical_accumulated_barplot_by_age = cvc_plot_factory.plot_vertical_accumulated_barplot_by_age()
    plot_vertical_accumulated_barplot_by_age.show()
    assert len(plot_vertical_accumulated_barplot_by_age.data) == 4


def test_plot_vertical_barplot_polarity():
    cvc_plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
    plot_vertical_barplot_polarity = cvc_plot_factory.plot_vertical_barplot_polarity('test_dataset_2', '1442274168')
    plot_vertical_barplot_polarity.show()
    assert len(plot_vertical_barplot_polarity.data) == 4


def test_plot_horizontal_bars_plot_interactions():
    cvc_plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
    horizontal_bars_plot_1_py_interactions_1, horizontal_bars_plot_2_py_interactions_2 = cvc_plot_factory.plot_horizontal_bars_plot_interactions(
        'test_dataset_2', '1442274168')
    horizontal_bars_plot_1_py_interactions_1.show()
    horizontal_bars_plot_2_py_interactions_2.show()
    assert len(horizontal_bars_plot_1_py_interactions_1.data) == 4
    assert len(horizontal_bars_plot_2_py_interactions_2.data) == 4


def test_plot_donut_plot_behaviour():
    cvc_plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
    donut_plot_py_behavior_1, donut_plot_py_behavior_2 = cvc_plot_factory.plot_donut_plot_behaviour('test_dataset_2',
                                                                                                    '1442274168')
    donut_plot_py_behavior_1.show()
    donut_plot_py_behavior_2.show()
    assert len(donut_plot_py_behavior_1.data) == 4
    assert len(donut_plot_py_behavior_2.data) == 4
