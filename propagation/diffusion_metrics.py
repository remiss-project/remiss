class DiffusionMetrics:

    def plot_propagation_tree(self, dataset, tweet_id):
        graph = self.get_propagation_tree(dataset, tweet_id)
        fig = self.get_propagation_figure(graph)
        return fig

    def get_size_over_time(self, dataset, tweet_id):
        graph = self.get_propagation_tree(dataset, tweet_id)
        # get the difference between the first tweet and the rest in minutes
        size = pd.Series(graph.vs['created_at'], index=graph.vs['label'])
        size = size - graph.vs.find(tweet_id=graph['conversation_id'])['created_at']
        size = size.dt.total_seconds() / 60
        return size

    def plot_size_over_time(self, dataset, tweet_id):
        size = self.get_size_over_time(dataset, tweet_id)
        # temporal cumulative histogram over time. the x axis is in minutes
        fig = px.histogram(size, x=size, nbins=100, cumulative=True)
        # set the x axis to be in minutes
        fig.update_xaxes(title_text='Minutes')
        fig.update_yaxes(title_text='Cumulative number of tweets')

        return fig

    def get_depth_over_time(self, dataset, tweet_id):
        graph = self.get_propagation_tree(dataset, tweet_id)
        shortest_paths = self.get_shortest_paths_to_conversation_id(graph)
        created_at = pd.Series(graph.vs['created_at'])
        order = created_at.argsort()
        shortest_paths = shortest_paths.iloc[order]
        created_at = created_at.iloc[order]
        depths = {}
        for i, time in enumerate(created_at):
            depths[time] = shortest_paths.iloc[:i + 1].max()

        depths = pd.Series(depths, name='Depth')
        return depths

    def plot_depth_over_time(self, dataset, tweet_id):
        depths = self.get_depth_over_time(dataset, tweet_id)
        fig = px.line(depths, x=depths.index, y=depths.values)
        fig.update_xaxes(title_text='Time')
        fig.update_yaxes(title_text='Depth')

        return fig

    def get_max_breadth_over_time(self, dataset, tweet_id):
        graph = self.get_propagation_tree(dataset, tweet_id)
        shortest_paths = self.get_shortest_paths_to_conversation_id(graph)
        created_at = pd.Series(graph.vs['created_at'])
        order = created_at.argsort()
        shortest_paths = shortest_paths.iloc[order]
        created_at = created_at.iloc[order]
        max_breadth = {}
        for i, time in enumerate(created_at):
            max_breadth[time] = shortest_paths.iloc[:i + 1].value_counts().max()

        max_breadth = pd.Series(max_breadth, name='Max Breadth')
        return max_breadth

    def plot_max_breadth_over_time(self, dataset, tweet_id):
        max_breadth = self.get_max_breadth_over_time(dataset, tweet_id)

        fig = px.line(max_breadth, x=max_breadth.index, y=max_breadth.values)
        fig.update_xaxes(title_text='Time')
        fig.update_yaxes(title_text='Max Breadth')

        return fig

    def get_structural_virality(self, dataset, tweet_id):
        graph = self.get_propagation_tree(dataset, tweet_id)
        # Specifically, we define structural
        # virality as the average distance between all pairs
        # of nodes in a diffusion tree
        shortests_paths = pd.DataFrame(graph.as_undirected().shortest_paths_dijkstra())
        shortests_paths = shortests_paths.replace(float('inf'), pd.NA)
        structured_virality = shortests_paths.mean().mean()
        return structured_virality

    def get_structural_virality_and_timespan(self, dataset, tweet_id):
        graph = self.get_propagation_tree(dataset, tweet_id)
        # Specifically, we define structural
        # virality as the average distance between all pairs
        # of nodes in a diffusion tree
        shortests_paths = pd.DataFrame(graph.as_undirected().shortest_paths_dijkstra())
        shortests_paths = shortests_paths.replace(float('inf'), pd.NA)
        created_at = pd.Series(graph.vs['created_at'])
        structured_virality = shortests_paths.mean().mean()
        timespan = created_at.max() - created_at.min()
        return structured_virality, timespan

    def get_structural_viralities(self, dataset):
        conversation_ids = self.get_conversation_ids(dataset)
        structured_viralities = []
        timespans = []
        for conversation_id in conversation_ids['conversation_id']:
            structured_virality, timespan = self.get_structural_virality_and_timespan(dataset, conversation_id)
            structured_viralities.append(structured_virality)
            timespans.append(timespan)
        # Cast to pandas
        structured_viralities = pd.DataFrame({'conversation_id': conversation_ids['conversation_id'],
                                              'structured_virality': structured_viralities,
                                              'timespan': timespans})
        structured_viralities = structured_viralities.set_index('conversation_id')

        return structured_viralities

    def get_structural_virality_over_time(self, dataset, tweet_id):
        graph = self.get_propagation_tree(dataset, tweet_id)
        # Specifically, we define structural
        # virality as the average distance between all pairs
        # of nodes in a diffusion tree
        shortests_paths = pd.DataFrame(graph.as_undirected().shortest_paths_dijkstra())
        shortests_paths = shortests_paths.replace(float('inf'), pd.NA)
        created_at = pd.Series(graph.vs['created_at'])
        order = created_at.argsort()
        shortests_paths = shortests_paths.iloc[order, order]
        created_at = created_at.iloc[order]
        structured_virality = {}
        for i, time in enumerate(created_at):
            current_shortests_paths = shortests_paths.iloc[:i + 1, :i + 1]
            structured_virality[time] = current_shortests_paths.mean().mean()

        structured_virality = pd.Series(structured_virality, name='Structured Virality')
        return structured_virality

    def plot_structural_virality_over_time(self, dataset, tweet_id):
        structured_virality = self.get_structural_virality_over_time(dataset, tweet_id)

        fig = px.line(structured_virality, x=structured_virality.index, y=structured_virality.values)
        fig.update_xaxes(title_text='Time')
        fig.update_yaxes(title_text='Structured Virality')

        return fig
