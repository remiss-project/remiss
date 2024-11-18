import logging

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymongo import MongoClient

from figures.figures import MongoPlotFactory
from figures.utils_plot import draw_vertical_barplot, draw_radarplot, \
    draw_vertical_acumulated_barplot_plotly, draw_horizontal_barplot  # , draw_donutplot

logger = logging.getLogger(__name__)

class ProfilingPlotFactory(MongoPlotFactory):
    def __init__(self, host="localhost", port=27017, available_datasets=None, lang='en',
                 data_dir='./profiling_data'):
        super().__init__(host, port, available_datasets)
        self.lang = lang
        self.data_dir = Path(data_dir)

        # Read feature selection files
        feature_files = {
            'ca': 'features_seleccion_catalan.csv',
            'es': 'features_seleccion_castellano.csv',
            'en': 'features_seleccion_ingles.csv'
        }
        self.features = pd.read_csv(self.data_dir / feature_files[lang], sep=';')

        # Set line spacer vector based on language
        self.line_spacer_vector = {
            'ca': [0, 1, 1, 11, 16, 11, 6, 0, 3, 0, 9],
            'es': [1, 1, 1, 11, 9, 0, 14, 0, 2, 9, 0],
            'en': [1, 1, 1, 11, 15, 0, 14, 0, 2, 9, 0]
        }[lang]

        fake_spreaders_feats, fact_checkers_feats, random_feats = self.get_all_values_users(lang)

        # Convert feature values to dataframes
        self.fake_news_spreaders = convert_dict_to_dataframe(fake_spreaders_feats)
        self.fact_checkers = convert_dict_to_dataframe(fact_checkers_feats)
        self.control_cases = convert_dict_to_dataframe(random_feats)

    def is_user_profiled(self, dataset, user_id):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('profiling')
        user_data = collection.find_one({'twitter_id': user_id})
        client.close()
        return user_data is not None

    def load_data_for_user(self, dataset, user_id):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('profiling')
        user_data = collection.find_one({'twitter_id': user_id})
        client.close()
        return user_data

    def get_all_values_users(self, lang):
        if lang == 'en':
            rel_fake_spreaders = load_medians_file_all_features(self.data_dir / "results_1_fake_spreaders_en")
            rel_fact_checkers = load_medians_file_all_features(self.data_dir / "results_2_fact_checkers_eng")
            rel_random = load_medians_file_all_features(self.data_dir / "results_3_random_en")
        elif lang == 'es':
            rel_fake_spreaders = load_medians_file_all_features(self.data_dir / "results_4_fake_spreaders_esp")
            rel_fact_checkers = load_medians_file_all_features(self.data_dir / "results_5_fact_checkers_esp")
            rel_random = load_medians_file_all_features(self.data_dir / "results_6_random_es")
        else:  # asumiendo catalan sin que existan otros lenguajes
            rel_fake_spreaders = load_medians_file_all_features(self.data_dir / "results_7_fake_spreaders_cat")
            rel_fact_checkers = load_medians_file_all_features(self.data_dir / "results_5_fact_checkers_esp")
            rel_random = load_medians_file_all_features(self.data_dir / "results_6_random_es")
        return rel_fake_spreaders, rel_fact_checkers, rel_random

    def plot_user_info(self, dataset, user_id):
        user_data = self.load_data_for_user(dataset, user_id)
        anonymized_description = user_data["user_summary"]
        anonymized_user_name = user_data['anonymizedName']

        verified = "Sí" if user_data['is_verified'] else "No"
        n_followers = str(user_data['followers_count'])
        n_followed = str(user_data['friends_count'])
        estimated_age = user_data['age_demo']
        estimated_gender = user_data['gender_demo']

        if estimated_gender == "male":
            estimated_gender = "Home"
        elif estimated_gender == "female":
            estimated_gender = "Dona"
        elif estimated_gender == "organization":
            estimated_gender = "Organització"

        return create_user_info_plot(anonymized_user_name, anonymized_description,
                                     n_followers, n_followed,
                                     estimated_age, estimated_gender, verified)

    def _get_plot_data(self, dataset, user_id, category):
        user_data = self.load_data_for_user(dataset, user_id)
        category_features = self.features.loc[
            self.features["category"] == category]
        feature_names = category_features["feature_name"].tolist()
        display_names = category_features["display_name"].tolist()

        user_data = [user_data[x] for x in feature_names]
        fake_news_spreaders = self.fake_news_spreaders[feature_names].values.astype(float).tolist()[0]
        fact_checkers = self.fact_checkers[feature_names].values.astype(float).tolist()[0]
        control_cases = self.control_cases[feature_names].values.astype(float).tolist()[0]
        return user_data, fake_news_spreaders, fact_checkers, control_cases, display_names

    def plot_vertical_barplot_topics(self, dataset, user_id):
        user_data, fake_news_spreaders, fact_checkers, control_cases, display_names = self._get_plot_data(dataset,
                                                                                                          user_id,
                                                                                                          'temas')

        bar_names = display_names
        values_to_show = [fake_news_spreaders, fact_checkers, control_cases, user_data]
        colors_to_show_values = ['red', 'green', 'blue', 'orange']
        labels = ['Difusors de Notícies Falses', "Verificadors", 'Usuaris de Control', 'Usuari']
        title = ("Top 20 temes d'interès amb més diferències significatives (p<0.05) entre difusors de "
                 "notícies falses i usuaris de control")
        return draw_vertical_barplot(bar_names, values_to_show, colors_to_show_values,
                                     labels, title)

    def plot_radarplot_emotions(self, dataset, user_id):
        user_data, fake_news_spreaders, fact_checkers, control_cases, display_names = self._get_plot_data(dataset,
                                                                                                          user_id,
                                                                                                          'emociones')

        labels = ['Narrative Shapers', 'Fact Checkers', 'Control Users', 'User']

        title = "Emocions"
        values = [fake_news_spreaders, fact_checkers, control_cases, user_data]
        colors = ['#1269A6', '#DD319D', '#6CAB12', '#319DE9']

        translations = {
            'Alegría*': 'Joy*',
            'Confiança*': 'Trust*',
            'Por*': 'Fear*',
            'Sorpresa*': 'Surprise*',
            'Tristesa': 'Sadness',
            'Fàstic': 'Disgust',
            'Ira*': 'Anger*',
            'Anticipació*': 'Anticipation*'
        }

        display_names = [translations.get(x, x) for x in display_names]
        return draw_radarplot(display_names, values, labels, title, colors)

    def plot_vertical_accumulated_barplot_by_genre(self):
        user_types = ['Narrative Shapers', 'Fact Checkers', 'Random Control Users']
        title = "User Group by Gender"

        if self.lang == "ca":
            females = (32.14, 17.98, 30.72)
            males = (67.86, 47.19, 66.09)
            genderless_organizations = (0, 34.83, 3.19)
        elif self.lang == "es":
            females = (19.03, 17.98, 30.72)
            males = (67.21, 47.19, 66.09)
            genderless_organizations = (13.77, 34.83, 3.19)
        elif self.lang == "en":
            females = (34.57, 27.12, 36.60)
            males = (59.48, 37.29, 61.08)
            genderless_organizations = (5.95, 35.59, 2.32)

        values = [females, males, genderless_organizations]
        labels = ["Female", "Male", "Genderless Organizations"]
        colors = ['lightskyblue', 'orange', 'grey']

        return draw_vertical_acumulated_barplot_plotly(user_types, values, colors,
                                                       labels, title)

    def plot_vertical_accumulated_barplot_by_age(self):
        user_types = ['Narrative Shapers', 'Fact Checkers', 'Random Control Users']
        title = "User Groups by Age"
        if self.lang == "ca":
            lower_19 = [14.29, 17.24, 36.83]
            lower_29 = (14.29, 18.97, 24.85)
            lower_39 = (21.43, 12.07, 12.28)
            higher_39 = (50.00, 51.72, 26.05)
        elif self.lang == "es":
            lower_19 = [13.62, 17.24, 36.83]
            lower_29 = (14.55, 18.97, 24.85)
            lower_39 = (21.13, 12.07, 12.28)
            higher_39 = (50.70, 51.72, 26.05)
        elif self.lang == "en":
            lower_19 = [5.53, 3.95, 33.51]
            lower_29 = (9.88, 21.05, 33.77)
            lower_39 = (13.83, 26.32, 12.66)
            higher_39 = (70.75, 48.68, 20.05)

        values = [lower_19, lower_29, lower_39, higher_39]
        labels = ["<19", "19-29", "30-39", ">39"]
        colors = ['lightskyblue', 'orange', 'grey', 'yellow']

        return draw_vertical_acumulated_barplot_plotly(user_types, values, colors,
                                                       labels, title)

    def plot_vertical_barplot_polarity(self, dataset, user_id):
        user_data = self.load_data_for_user(dataset, user_id)
        category_features = self.features.loc[
            self.features["category"] == 'sentimientos_hate_speech']
        features_names = category_features["feature_name"].tolist()
        bar_names = category_features["display_name"].tolist()
        current_user = [user_data[x] for x in features_names]

        fake_news_spreaders = self.fake_news_spreaders[features_names].values.astype(float).tolist()[0]
        fact_checkers = self.fact_checkers[features_names].values.astype(float).tolist()[0]
        control_cases = self.control_cases[features_names].values.astype(float).tolist()[0]
        # label1 = 'Difusors de Notícies Falses'
        # label2 = 'Verificadors'
        # label3 = 'Usuaris de Control'
        # label4 = 'Usuari'
        label1 = 'Narrative Shapers'
        label2 = 'Fact Checkers'
        label3 = 'Control Users'
        label4 = 'User'

        values_to_show = [fake_news_spreaders, fact_checkers, control_cases, current_user]
        # colors_to_show_values = ['red', 'green', 'blue', 'orange']
        colors_to_show_values = ['#1269A6', '#DD319D', '#6CAB12', '#319DE9']

        labels = [label1, label2, label3, label4]

        translations = {'positiva*': 'Positive*', 'negativa*': 'Negative*', 'neutral*': 'Neutral*',
                        "discurs d'odi": "Hate Speech"}
        bar_names = [translations.get(x, x) for x in bar_names]
                
        return draw_vertical_barplot(bar_names, values_to_show, colors_to_show_values,
                                     labels)

    def plot_horizontal_bars_plot_interactions(self, dataset, user_id):
        user_data = self.load_data_for_user(dataset, user_id)
        category_features = self.features.loc[
            self.features["category"] == 'interacciones']
        features_names = category_features["feature_name"].tolist()
        bar_names = category_features["display_name"].tolist()
        current_user = [user_data[x] for x in features_names]

        fake_news_spreaders = self.fake_news_spreaders[features_names].values.astype(float).tolist()[0]
        fact_checkers = self.fact_checkers[features_names].values.astype(float).tolist()[0]
        control_cases = self.control_cases[features_names].values.astype(float).tolist()[0]
        labels = ['Difusors de Notícies Falses', 'Verificadors', 'Usuaris de Control', 'Usuari']
        title = "Interaccions Socials"
        index_list_for_plot_1 = [0, 1, 3]
        index_list_for_plot_2 = [2, 4, 5, 6]

        values_to_show_1 = [list(np.take(fake_news_spreaders, index_list_for_plot_1)),
                            list(np.take(fact_checkers, index_list_for_plot_1)),
                            list(np.take(control_cases, index_list_for_plot_1)),
                            list(np.take(current_user, index_list_for_plot_1))]
        values_to_show_2 = [list(np.take(fake_news_spreaders, index_list_for_plot_2)),
                            list(np.take(fact_checkers, index_list_for_plot_2)),
                            list(np.take(control_cases, index_list_for_plot_2)),
                            list(np.take(current_user, index_list_for_plot_2))]

        bar_names_1 = list(np.take(bar_names, index_list_for_plot_1))
        bar_names_2 = list(np.take(bar_names, index_list_for_plot_2))

        colors_to_show_values = ['red', 'green', 'blue', 'orange']

        horizontal_bars_plot_1_py_interactions_1 = draw_horizontal_barplot(bar_names_1, values_to_show_1,
                                                                           colors_to_show_values, labels,
                                                                           title)

        horizontal_bars_plot_2_py_interactions_2 = draw_horizontal_barplot(bar_names_2, values_to_show_2,
                                                                           colors_to_show_values, labels,
                                                                           title)
        return horizontal_bars_plot_1_py_interactions_1, horizontal_bars_plot_2_py_interactions_2

    def plot_donut_plot_behaviour(self, dataset, user_id):
        user_data = self.load_data_for_user(dataset, user_id)
        if user_data is None:
            return {}, {}
        category_features = self.features.loc[
            self.features["category"] == 'comportamiento']
        features_names = category_features["feature_name"].tolist()
        display_names = category_features["display_name"].tolist()
        current_user = [user_data[x] for x in features_names]

        translations = {
            'proporció de tweets publicats en días laborables (Dill-Div)': 'Proportion of tweets posted on weekdays (Monday to Friday)',
            'proporció de tweets publicats en cap de setmana (Diss i Diu)': 'Proportion of tweets posted on weekends (Saturday and Sunday)',
            'proporció de tweets publicats a la nit': 'Proportion of tweets posted at night',
            'proporció de tweets publicats durant el día': 'Proportion of tweets posted during the day'
        }
        display_names = [translations.get(x, x) for x in display_names]

        fake_news_spreaders = self.fake_news_spreaders[features_names].values.astype(float).tolist()[0]
        fact_checkers = self.fact_checkers[features_names].values.astype(float).tolist()[0]
        control_cases = self.control_cases[features_names].values.astype(float).tolist()[0]

        # titles = ['Difusors de Notícies Falses', 'Verificadors', 'Usuaris de Control', 'Usuari']
        titles = ['Narrative Shapers', 'Fact Checkers', 'Control Users', 'User']

        # Data for the first donut plot
        values_to_show_1 = [[fake_news_spreaders[0], fake_news_spreaders[1]],
                            [fact_checkers[0], fact_checkers[1]],
                            [control_cases[0], control_cases[1]],
                            [current_user[0], current_user[1]]]
        labels_1 = [display_names[0], display_names[1]]

        donut_plot_py_behavior_1 = draw_donutplot(values_to_show_1, labels_1, titles)

        # Data for the second donut plot
        labels_2 = [display_names[2], display_names[3]]
        values_to_show_2 = [[fake_news_spreaders[2], fake_news_spreaders[3]],
                            [fact_checkers[2], fact_checkers[3]],
                            [control_cases[2], control_cases[3]],
                            [current_user[2], current_user[3]]]

        donut_plot_py_behavior_2 = draw_donutplot(values_to_show_2, labels_2, titles)

        return donut_plot_py_behavior_1, donut_plot_py_behavior_2

    def get_median_values_users(self, lang):
        if lang == 'en':
            rel_feats = load_relevant_features(
                self.data_dir / "results_selected_features_names1_fake_spreaders_en3_random_en")
            rel_fake_spreaders = load_medians_file(self.data_dir / "results_1_fake_spreaders_en", rel_feats)
            rel_fact_checkers = load_medians_file(self.data_dir / "results_2_fact_checkers_eng", rel_feats)
            rel_random = load_medians_file(self.data_dir / "results_3_random_en", rel_feats)
        elif lang == 'es':
            rel_feats = load_relevant_features(
                self.data_dir / "results_selected_features_names4_fake_spreaders_esp6_random_es")
            rel_fake_spreaders = load_medians_file(self.data_dir / "results_4_fake_spreaders_esp", rel_feats)
            rel_fact_checkers = load_medians_file(self.data_dir / "results_5_fact_checkers_esp", rel_feats)
            rel_random = load_medians_file(self.data_dir / "results_6_random_es", rel_feats)
        else:  # asumiendo catalan sin que existan otros lenguajes
            rel_feats = load_relevant_features(
                self.data_dir / "results_selected_features_names6_random_es7_fake_spreaders_cat")
            rel_fake_spreaders = load_medians_file(self.data_dir / "results_7_fake_spreaders_cat", rel_feats)
            rel_fact_checkers = load_medians_file(self.data_dir / "results_5_fact_checkers_esp", rel_feats)
            rel_random = load_medians_file(self.data_dir / "results_6_random_es", rel_feats)
        return rel_feats, rel_fake_spreaders, rel_fact_checkers, rel_random


def load_relevant_features(relevant_features_path):
    relevant_features = {}
    with open(relevant_features_path, 'r', encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            feature = line.replace("\n", "")
            if ":" in feature:
                relevant_features[feature] = []
                current_feat = feature
            else:
                relevant_features[current_feat].append(feature)
    return (relevant_features)


def load_medians_file(filepath, relevant_feats):
    relevant_vals = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            line = line.split("\t")
            feature = line[0]
            median = line[1].replace("\n", "")
            for feature_type in relevant_feats:
                if feature in relevant_feats[feature_type]:
                    if feature_type not in relevant_vals:
                        relevant_vals[feature_type] = {}
                    relevant_vals[feature_type][feature] = median
    return relevant_vals


def convert_dict_to_dataframe(relevantFeaturesDict):
    # df = pd.DataFrame();
    listOfKeys = [];
    listOfValues = []
    for dictTittle, currentDict in relevantFeaturesDict.items():
        # print("\-------------------------DIC TITTLE:", dictTittle)
        for key in currentDict.keys():
            # print("	",key + ':', currentDict[key])
            listOfKeys.append(key);
            listOfValues.append(currentDict[key])
    # df[str(key)]=currentDict[key];
    #	print ("DataFrame is ", pd)
    # print ("keys", listOfKeys)
    #	print ("values", listOfValues)
    #	print("lengths", len(listOfKeys),len(listOfValues));
    tmpDict = dict(zip(listOfKeys, listOfValues))
    df = pd.DataFrame(tmpDict, index=[0]);
    # print("    /-/*-/*-/*-/-*/*-/-*/-*/-*/-*/-DATAFRAME IS",df.to_string());
    # print ("Size is ", df.info())
    return df


def load_medians_file_all_features(filepath):
    relevant_vals = {}
    relevant_vals['AllValues'] = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            line = line.split("\t")
            feature = line[0]
            median = line[1].replace("\n", "")
            relevant_vals['AllValues'][feature] = median
    return relevant_vals


def create_user_info_plot(user_name, anonymized_description, n_followers, n_followed, estimated_age,
                          estimated_gender_str, verified_str, width=700, height=700):
    # Create a blank figure
    fig = go.Figure()

    # Add annotations for user information
    user_info_annotations = [
        ("Anàlisis de l'usuari:" + user_name, 4.0),
        (format_description(anonymized_description), 3.5),
        ("Nombre of Followers: " + n_followers, 2.0),
        ("Nombre of Followed: " + n_followed, 1.75),
        ("Edat estimada: " + estimated_age, 1.5),
        ("Génere estimat: " + estimated_gender_str, 1.25),
        ("Compte verificat: " + verified_str, 1.0)
    ]

    for text, y in user_info_annotations:
        fig.add_annotation(
            text=text,
            x=0.05,
            y=y,
            xanchor='left',
            yanchor='top',
            font=dict(size=18, color='grey'),
            showarrow=False
        )

    # Set the layout parameters
    fig.update_layout(
        width=width,
        height=height,
        plot_bgcolor='white',
        yaxis={'visible': False, 'showticklabels': False},
        xaxis={'visible': False, 'showticklabels': False}
    )

    return fig


def format_description(description):
    # Break the description into multiple lines
    description_lines = [description[i:i + 60] for i in range(0, len(description), 60)]
    return '<br>'.join(description_lines)


def draw_donutplot(values, labels, titles):
    
    logger.debug(f'Values: {values}')
    logger.debug(f'Labels: {labels}')
    logger.debug(f'Titles: {titles}')

    colors = ['#1269A6', '#DD319D', '#6CAB12', '#319DE9']
    
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'pie'}, {'type': 'pie'}], [{'type': 'pie'}, {'type': 'pie'}]],
        subplot_titles=titles
    )

    for i in range(len(values)):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        color = colors[i]
        
        fig.add_trace(go.Pie(
            labels=labels,
            values=values[i],
            hole=0.4,
            marker=dict(colors=[color, color + "88"]),
            textinfo='percent',
            name=titles[i],
            textposition='inside',
            direction='clockwise',
            showlegend=True,
            hoverinfo='label+percent',
        ), row, col)

    fig.update_layout(
        showlegend=False,
    )

    return fig
