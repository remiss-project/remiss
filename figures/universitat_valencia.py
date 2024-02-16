from figures.figures import RemoteAPIFactory


class EmotionPerHourPlotFactory(RemoteAPIFactory):
    def __init__(self, api_url='http://agentsim.uv.es:5000/api', chart_id='graph1'):
        super().__init__(api_url, chart_id)

    def plot_emotion_per_hour(self, dataset, start_time, end_time):
        return self.plotly_html_to_figure(self.fetch_plot_html(dataset, start_time, end_time))


class AverageEmotionBarPlotFactory(RemoteAPIFactory):
    def __init__(self, api_url='http://agentsim.uv.es:5000/api', chart_id='graph2'):
        super().__init__(api_url, chart_id)

    def plot_average_emotion(self, dataset, start_time, end_time):
        return self.plotly_html_to_figure(self.fetch_plot_html(dataset, start_time, end_time))


class TopProfilesPlotFactory(RemoteAPIFactory):
    def __init__(self, api_url='http://agentsim.uv.es:5000/api', chart_id='graph3'):
        super().__init__(api_url, chart_id)

    def plot_top_profiles(self, dataset, start_time, end_time):
        return self.plotly_html_to_figure(self.fetch_plot_html(dataset, start_time, end_time))


class TopHashtagsPlotFactory(RemoteAPIFactory):
    def __init__(self, api_url='http://agentsim.uv.es:5000/api', chart_id='graph4'):
        super().__init__(api_url, chart_id)

    def plot_top_hashtags(self, dataset, start_time, end_time):
        return self.plotly_html_to_figure(self.fetch_plot_html(dataset, start_time, end_time))


class TopicRankingPlotFactory(RemoteAPIFactory):
    def __init__(self, api_url='http://agentsim.uv.es:5000/api', chart_id='graph5'):
        super().__init__(api_url, chart_id)

    def plot_topic_ranking(self, dataset, start_time, end_time):
        return self.plotly_html_to_figure(self.fetch_plot_html(dataset, start_time, end_time))


# network topics by probability of talk about fake news
class NetworkTopicsPlotFactory(RemoteAPIFactory):
    def __init__(self, api_url='http://agentsim.uv.es:5000/api', chart_id='graph6'):
        super().__init__(api_url, chart_id)

    def plot_network_topics(self, dataset, start_time, end_time):
        return self.plotly_html_to_figure(self.fetch_plot_html(dataset, start_time, end_time))
