import os

from dash_bootstrap_components.themes import FLATLY
from dash_oop_components import DashApp

from components import TweetUserTimeSeries
from figures import CountAreaPlots

REMISS_MONGODB_HOST = os.environ.get('REMISS_MONGODB_HOST', 'localhost')
REMISS_MONGODB_PORT = int(os.environ.get('REMISS_MONGODB_PORT', 27017))
REMISS_MONGODB_DATABASE = os.environ.get('REMISS_MONGODB_DATABASE', 'test_remiss')

plot_factory = CountAreaPlots(REMISS_MONGODB_HOST, REMISS_MONGODB_PORT, REMISS_MONGODB_DATABASE)
dashboard = TweetUserTimeSeries(plot_factory)

if __name__ == '__main__':
    dashboard.to_yaml("dashboard_component.yaml")
    app = DashApp(dashboard, querystrings=True, bootstrap=FLATLY)
    app.run(debug=True)
