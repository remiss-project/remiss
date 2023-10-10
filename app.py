import os

import dash
import dash_bootstrap_components as dbc

from components import RemissDashboard
from figures import TweetUserPlotFactory, EgonetPlotFactory

REMISS_MONGODB_HOST = os.environ.get('REMISS_MONGODB_HOST', 'localhost')
REMISS_MONGODB_PORT = int(os.environ.get('REMISS_MONGODB_PORT', 27017))
REMISS_MONGODB_DATABASE = os.environ.get('REMISS_MONGODB_DATABASE', 'test_remiss')

tweet_user_plot_factory = TweetUserPlotFactory(host=REMISS_MONGODB_HOST, port=REMISS_MONGODB_PORT,
                                                  database=REMISS_MONGODB_DATABASE)
egonet_plot_factory = EgonetPlotFactory(host=REMISS_MONGODB_HOST, port=REMISS_MONGODB_PORT,
                                        database=REMISS_MONGODB_DATABASE)
dashboard = RemissDashboard(tweet_user_plot_factory, egonet_plot_factory)

app = dash.Dash(external_stylesheets=[dbc.themes.JOURNAL])
app.layout = dashboard.layout()
# Run the app
if __name__ == '__main__':
    app.run(debug=True)
