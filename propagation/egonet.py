


class Egonet:
    def get_egonet(self, dataset, user, depth):
        """
        Returns the egonet of a user of a certain date and depth if present,
        otherwise returns the simplified hidden network
        :param dataset:
        :param user:
        :param depth:
        :return:
        """
        hidden_network = self.get_hidden_network(dataset)
        # check if the user is in the hidden network
        if user:
            try:
                node = hidden_network.vs.find(username=user)
                neighbours = hidden_network.neighborhood(node, order=depth)
                egonet = hidden_network.induced_subgraph(neighbours)
                return egonet
            except (RuntimeError, ValueError) as ex:
                print(f'Computing neighbourhood for user {user} failed with error {ex}')
        if self.simplification:
            return self.get_simplified_hidden_network(dataset)
        else:
            return hidden_network

    def get_hidden_network(self, dataset):
        stem = f'hidden_network'
        if dataset not in self._hidden_networks:
            if self.cache_dir and self.is_cached(dataset, stem):
                network = self.load_from_cache(dataset, stem)
            else:
                network = self._compute_hidden_network(dataset)
                layout = self.compute_layout(network)

                network['layout'] = layout
                if self.cache_dir:
                    self.save_to_cache(dataset, network, stem)
            self._hidden_networks[dataset] = network

        return self._hidden_networks[dataset]

    def get_simplified_hidden_network(self, dataset):
        stem = f'hidden_network-{self.simplification}-{self.threshold}'
        if dataset not in self._simplified_hidden_networks:
            if self.cache_dir and self.is_cached(dataset, stem):
                network = self.load_from_cache(dataset, stem)
            else:
                network = self.get_hidden_network(dataset)
                network = self._simplify_graph(network)
                if self.cache_dir:
                    self.save_to_cache(dataset, network, stem)
            self._simplified_hidden_networks[dataset] = network

        return self._simplified_hidden_networks[dataset]

    def _get_authors(self, dataset, start_date=None, end_date=None):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        nested_pipeline = [
            {'$project': {'author_id': '$author.id',
                          'username': '$author.username',
                          'is_usual_suspect': '$author.remiss_metadata.is_usual_suspect',
                          'party': '$author.remiss_metadata.party',
                          'num_tweets': '$author.public_metrics.tweet_count',
                          'num_followers': '$author.public_metrics.followers_count',
                          'num_following': '$author.public_metrics.following_count',
                          }
             }]
        self._add_date_filters(nested_pipeline, start_date, end_date)

        node_pipeline = [
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.author': {'$exists': True}}},
            {'$project': {'_id': 0, 'author_id': '$referenced_tweets.author.id',
                          'username': '$referenced_tweets.author.username',
                          'is_usual_suspect': '$referenced_tweets.author.remiss_metadata.is_usual_suspect',
                          'party': '$referenced_tweets.author.remiss_metadata.party',
                          'num_tweets': '$referenced_tweets.author.public_metrics.tweet_count',
                          'num_followers': '$referenced_tweets.author.public_metrics.followers_count',
                          'num_following': '$referenced_tweets.author.public_metrics.following_count'
                          }},
            {'$unionWith': {'coll': 'raw', 'pipeline': nested_pipeline}},  # Fetch missing authors
            {'$group': {'_id': '$author_id',
                        'username': {'$first': '$username'},
                        'is_usual_suspect': {'$addToSet': '$is_usual_suspect'},
                        'party': {'$addToSet': '$party'},
                        'num_tweets': {'$last': '$num_tweets'},
                        'num_followers': {'$last': '$num_followers'},
                        'num_following': {'$last': '$num_following'}

                        }},

            {'$project': {'_id': 0,
                          'author_id': '$_id',
                          'username': 1,
                          'is_usual_suspect': {'$anyElementTrue': '$is_usual_suspect'},
                          'party': {'$arrayElemAt': ['$party', 0]},
                          'num_tweets': 1,
                          'num_followers': 1,
                          'num_following': 1},
             }
        ]
        self._add_date_filters(node_pipeline, start_date, end_date)
        print('Computing authors')
        start_time = time.time()
        schema = Schema({'author_id': str, 'username': str, 'is_usual_suspect': bool, 'party': str,
                         'num_tweets': int, 'num_followers': int, 'num_following': int})
        authors = collection.aggregate_pandas_all(node_pipeline, schema=schema)
        print(f'Authors computed in {time.time() - start_time} seconds')
        client.close()
        return authors

    def _get_references(self, dataset, start_date=None, end_date=None):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')

        references_pipeline = [
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.author': {'$exists': True}}},
            {'$project': {'_id': 0, 'source': '$author.id', 'target': '$referenced_tweets.author.id'}},
            {'$group': {'_id': {'source': '$source', 'target': '$target'},
                        'weight': {'$count': {}}}},
            {'$project': {'_id': 0, 'source': '$_id.source', 'target': '$_id.target', 'weight': 1}},
            {'$group': {'_id': '$source',
                        'node_weight': {'$sum': '$weight'},
                        'references': {'$push': {'target': '$target', 'weight': '$weight'}}}},
            {'$unwind': '$references'},
            {'$project': {'_id': 0, 'source': '$_id', 'target': '$references.target',
                          'weight': '$references.weight',
                          'weight_inv': {'$divide': [1, '$references.weight']},
                          'weight_norm': {'$divide': ['$references.weight', '$node_weight']},
                          }},
        ]
        self._add_date_filters(references_pipeline, start_date, end_date)
        print('Computing references')
        start_time = time.time()
        references = collection.aggregate_pandas_all(references_pipeline)
        print(f'References computed in {time.time() - start_time} seconds')
        client.close()
        return references

    def _compute_hidden_network(self, dataset):
        """
        Computes the hidden graph, this is, the graph of users that have interacted with each other.
        :param dataset: collection name within the database where the tweets are stored
        :return: a networkx graph with the users as nodes and the edges representing interactions between users
        """
        authors = self._get_authors(dataset)
        references = self._get_references(dataset)

        if len(authors) == 0:
            # in case of no authors we return an empty graph
            return ig.Graph(directed=True)

        print('Computing graph')
        start_time = time.time()
        # switch id by position (which will be the node id in the graph) and set it as index
        author_to_id = authors['author_id'].reset_index().set_index('author_id')
        # we only have reputation and legitimacy for a subset of the authors, so the others will be set to nan
        available_legitimacy = self.get_legitimacy(dataset)
        available_reputation = self.get_reputation(dataset)
        available_status = self.get_status(dataset)
        reputation = pd.DataFrame(np.nan, index=author_to_id.index, columns=available_reputation.columns)
        reputation.loc[available_reputation.index] = available_reputation
        legitimacy = pd.Series(np.nan, index=author_to_id.index)
        legitimacy[available_legitimacy.index] = available_legitimacy['legitimacy']
        status = pd.DataFrame(np.nan, index=author_to_id.index, columns=available_status.columns)
        status.loc[available_status.index] = available_status

        g = ig.Graph(directed=True)
        g.add_vertices(len(authors))
        g.vs['author_id'] = authors['author_id']
        g.vs['username'] = authors['username']
        g.vs['is_usual_suspect'] = authors['is_usual_suspect']
        g.vs['party'] = authors['party']
        g['reputation'] = reputation
        g['status'] = status
        g.vs['legitimacy'] = legitimacy.to_list()
        available_t_closeness = self.get_t_closeness(g)
        g.vs['t-closeness'] = available_t_closeness.to_list()
        g.vs['num_connections'] = g.degree()
        g.vs['num_tweets'] = authors['num_tweets']
        g.vs['num_followers'] = authors['num_followers']
        g.vs['num_following'] = authors['num_following']

        if len(references) > 0:
            # convert references which are author id based to graph id based
            references['source'] = author_to_id.loc[references['source']].reset_index(drop=True)
            references['target'] = author_to_id.loc[references['target']].reset_index(drop=True)

            g.add_edges(references[['source', 'target']].to_records(index=False).tolist())
            g.es['weight'] = references['weight']
            g.es['weight_inv'] = references['weight_inv']
            g.es['weight_norm'] = references['weight_norm']

        print(g.summary())
        print(f'Graph computed in {time.time() - start_time} seconds')


        return g

    def _simplify_graph(self, network):
        if self.simplification == 'maximum_spanning_tree':
            network = network.spanning_tree(weights=network.es['weight_inv'])
        elif self.simplification == 'k_core':
            network = network.k_core(self.k_cores)
        elif self.simplification == 'backbone':
            network = self.compute_backbone(network, self.threshold, self.delete_vertices)
        else:
            raise ValueError(f'Unknown simplification {self.simplification}')
        return network

    @staticmethod
    def compute_backbone(graph, alpha=0.05, delete_vertices=True):
        # Compute alpha for all edges (1 - weight_norm)^(degree_of_source_node - 1)
        weights = np.array(graph.es['weight_norm'])
        degrees = np.array([graph.degree(e[0]) for e in graph.get_edgelist()])
        alphas = (1 - weights) ** (degrees - 1)
        good = np.nonzero(alphas > alpha)[0]
        backbone = graph.subgraph_edges(graph.es.select(good), delete_vertices=delete_vertices)
        if 'layout' in graph.attributes():
            layout = pd.DataFrame(graph['layout'].coords, columns=['x', 'y', 'z'], index=graph.vs['author_id'])
            backbone['layout'] = Layout(layout.loc[backbone.vs['author_id']].values.tolist(), dim=3)
        return backbone