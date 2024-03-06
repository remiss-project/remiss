from pathlib import Path

import numpy as np
import pandas as pd
from pymongo import MongoClient

from figures.figures import MongoPlotFactory
from figures.utils_plot import create_user_info_image, draw_vertical_barplot, draw_radarplot, \
    draw_vertical_acumulated_barplot_plotly, draw_horizontal_barplot, draw_donutplot
from figures.utils_remiss import convert_dict_to_dataframe, get_all_values_users, get_median_values_users


class CVCPlotFactory(MongoPlotFactory):
    def __init__(self, host="localhost", port=27017, database="CVCUI2", available_datasets=None, lang='ca',
                 data_dir='./data'):
        super().__init__(host, port, database, available_datasets)
        self.lang = lang
        self.data_dir = Path(data_dir)

        # Read feature selection files
        feature_selection_cat = pd.read_csv(self.data_dir / 'features_seleccion_catalan.csv', sep=';')
        feature_selection_esp = pd.read_csv(self.data_dir / 'features_seleccion_castellano.csv', sep=';')
        feature_selection_eng = pd.read_csv(self.data_dir / 'features_seleccion_ingles.csv', sep=';')

        # Set dataframe based on language
        if self.lang == 'ca':
            self.feature_selection = feature_selection_cat
            self.line_spacer_vector = [0, 1, 1, 11, 16, 11, 6, 0, 3, 0, 9]
        elif self.lang == 'es':
            self.feature_selection = feature_selection_esp
            self.line_spacer_vector = [1, 1, 1, 11, 9, 0, 14, 0, 2, 9, 0]
        elif self.lang == 'en':
            self.feature_selection = feature_selection_eng
            self.line_spacer_vector = [1, 1, 1, 11, 15, 0, 14, 0, 2, 9, 0]
        # Get feature values for different user groups
        (self.rel_feats, self.rel_fake_spreaders,
         self.rel_fact_checkers, self.rel_random) = get_median_values_users(self.lang)
        fake_spreaders_feats, fact_checkers_feats, random_feats = get_all_values_users(self.lang)

        # Convert feature values to dataframes
        self.spreaders = convert_dict_to_dataframe(fake_spreaders_feats)
        self.checkers = convert_dict_to_dataframe(fact_checkers_feats)
        self.control_cases = convert_dict_to_dataframe(random_feats)

    def load_data_for_user(self, dataset, user_id):
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.database)
        collection = database.get_collection(dataset)
        user_data = collection.find_one({'author.id': user_id})
        client.close()
        return user_data

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

        return create_user_info_image(anonymized_user_name, anonymized_description,
                                      n_followers, n_followed,
                                      estimated_age, estimated_gender, verified)

    def plot_vertical_barplot_topics(self, dataset, user_id):
        user_data = self.load_data_for_user(dataset, user_id)
        category_features = self.feature_selection.loc[
            self.feature_selection["category"] == 'temas']
        features_names = category_features["feature_name"].tolist()
        topics_of_interest = category_features["display_name"].tolist()

        current_user = [user_data[x] for x in features_names]
        fake_news_spreaders = self.spreaders[features_names].values.astype(float).tolist()[0]
        fact_checkers = self.checkers[features_names].values.astype(float).tolist()[0]
        control_cases = self.control_cases[features_names].values.astype(float).tolist()[0]

        label1 = 'Difusors de Notícies Falses'
        label2 = 'Usuaris de Control'
        label3 = 'Usuari'
        label4 = "Verificadors"

        bar_names = topics_of_interest
        values_to_show = [fake_news_spreaders, fact_checkers, control_cases, current_user]
        colors_to_show_values = ['red', 'green', 'blue', 'orange']
        labels = [label1, label4, label2, label3]
        title = "Top 20 temes d'interès amb més diferències significatives (p<0.05) entre difusors de notícies falses i usuaris de control"
        return draw_vertical_barplot(bar_names, values_to_show, colors_to_show_values,
                                     labels, title)

    def plot_radarplot_emotions(self, dataset, user_id):
        user_data = self.load_data_for_user(dataset, user_id)
        category_features = self.feature_selection.loc[
            self.feature_selection["category"] == 'emociones']
        features_names = category_features["feature_name"].tolist()
        categories = category_features["display_name"].tolist()

        current_user = [user_data[x] for x in features_names]
        fake_news_spreaders = self.spreaders[features_names].values.astype(float).tolist()[0]
        fact_checkers = self.checkers[features_names].values.astype(float).tolist()[0]
        control_cases = self.control_cases[features_names].values.astype(float).tolist()[0]

        labels = ('Fake News Spreaders', 'Fact Checkers', 'Random Control Users', 'User')
        title = "Emotions"
        values = [fake_news_spreaders, fact_checkers, control_cases, current_user]
        colors = ['red', 'green', 'lightskyblue', 'orange']

        return draw_radarplot(categories, values, labels, title, colors)

    def plot_vertical_accumulated_barplot_by_genre(self, dataset, user_id, genre):
        user_types = ['Fake News Spreaders', 'Fact Checkers', 'Random Control Users']
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

    def plot_vertical_accumulated_barplot_by_age(self, dataset, user_id, age):
        user_types = ['Fake News Spreaders', 'Fact Checkers', 'Random Control Users']
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
        category_features = self.feature_selection.loc[
            self.feature_selection["category"] == 'sentimientos_hate_speech']
        features_names = category_features["feature_name"].tolist()
        bar_names = category_features["display_name"].tolist()
        current_user = [user_data[x] for x in features_names]

        fake_news_spreaders = self.spreaders[features_names].values.astype(float).tolist()[0]
        fact_checkers = self.checkers[features_names].values.astype(float).tolist()[0]
        control_cases = self.control_cases[features_names].values.astype(float).tolist()[0]
        label1 = 'Difusors de Notícies Falses'
        label2 = 'Verificadors'
        label3 = 'Usuaris de Control'
        label4 = 'Usuari'

        values_to_show = [fake_news_spreaders, fact_checkers, control_cases, current_user]
        colors_to_show_values = ['red', 'green', 'blue', 'orange']
        labels = [label1, label2, label3, label4]
        title = "Polaritat dels Tweets"

        return draw_vertical_barplot(bar_names, values_to_show, colors_to_show_values,
                                     labels, title)

    def plot_horizontal_bars_plot_interactions(self, dataset, user_id):
        user_data = self.load_data_for_user(dataset, user_id)
        category_features = self.feature_selection.loc[
            self.feature_selection["category"] == 'interacciones']
        features_names = category_features["feature_name"].tolist()
        bar_names = category_features["display_name"].tolist()
        current_user = [user_data[x] for x in features_names]

        fake_news_spreaders = self.spreaders[features_names].values.astype(float).tolist()[0]
        fact_checkers = self.checkers[features_names].values.astype(float).tolist()[0]
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
        category_features = self.feature_selection.loc[
            self.feature_selection["category"] == 'comportamiento']
        features_names = category_features["feature_name"].tolist()
        display_names = category_features["display_name"].tolist()
        current_user = [user_data[x] for x in features_names]

        fake_news_spreaders = self.spreaders[features_names].values.astype(float).tolist()[0]
        fact_checkers = self.checkers[features_names].values.astype(float).tolist()[0]
        control_cases = self.control_cases[features_names].values.astype(float).tolist()[0]

        titles = ['Difusors de Notícies Falses', 'Verificadors', 'Usuaris de Control', 'Usuari']

        # Data for the first donut plot
        values_to_show_1 = [[fake_news_spreaders[0], fake_news_spreaders[1]],
                            [fact_checkers[0], fact_checkers[1]],
                            [control_cases[0], control_cases[1]],
                            [current_user[0], current_user[1]]]
        labels_1 = [display_names[0], display_names[1]]

        donut_plot_py_behavior_1 = draw_donutplot(values_to_show_1, labels_1, titles,
                                                  "Gràfic de Comportament 1")

        # Data for the second donut plot
        labels_2 = [display_names[2], display_names[3]]
        values_to_show_2 = [[fake_news_spreaders[2], fake_news_spreaders[3]],
                            [fact_checkers[2], fact_checkers[3]],
                            [control_cases[2], control_cases[3]],
                            [current_user[2], current_user[3]]]

        donut_plot_py_behavior_2 = draw_donutplot(values_to_show_2, labels_2, titles,
                                                  "Gràfic de Comportament 2")

        return donut_plot_py_behavior_1, donut_plot_py_behavior_2
