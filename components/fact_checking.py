from components.components import RemissComponent


class FactCheckingComponent(RemissComponent):
    def __init__(self, plot_factory, state,
                 name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.state = state

    def layout(self, params=None):
        pass

    def update(self, dataset, hashtags, start_date, end_date):
        pass

    def callbacks(self, app):
        pass
