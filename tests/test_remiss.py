from pathlib import Path
from unittest import TestCase

from remiss import flatten_tweets
import pandas as pd


class TestRemiss(TestCase):
    def test_flatten_tweets(self):
        flatten_tweets('test_resources/test.jsonl')
        df = pd.read_csv('test_resources/test.csv')
        self.assertEqual(df.shape, (934, 84))
        self.assertEqual(df.columns.to_list(),
                         ['id', 'conversation_id', 'referenced_tweets.replied_to.id', 'referenced_tweets.retweeted.id',
                          'referenced_tweets.quoted.id', 'author_id', 'in_reply_to_user_id', 'in_reply_to_username',
                          'retweeted_user_id', 'retweeted_username', 'quoted_user_id', 'quoted_username', 'created_at',
                          'text', 'lang', 'source', 'public_metrics.impression_count', 'public_metrics.reply_count',
                          'public_metrics.retweet_count', 'public_metrics.quote_count', 'public_metrics.like_count',
                          'public_metrics.bookmark_count', 'reply_settings', 'edit_history_tweet_ids',
                          'edit_controls.edits_remaining', 'edit_controls.editable_until',
                          'edit_controls.is_edit_eligible', 'possibly_sensitive', 'withheld.scope',
                          'withheld.copyright', 'withheld.country_codes', 'entities.annotations', 'entities.cashtags',
                          'entities.hashtags', 'entities.mentions', 'entities.urls', 'context_annotations',
                          'attachments.media', 'attachments.media_keys', 'attachments.poll.duration_minutes',
                          'attachments.poll.end_datetime', 'attachments.poll.id', 'attachments.poll.options',
                          'attachments.poll.voting_status', 'attachments.poll_ids', 'author.id', 'author.created_at',
                          'author.username', 'author.name', 'author.description',
                          'author.entities.description.cashtags', 'author.entities.description.hashtags',
                          'author.entities.description.mentions', 'author.entities.description.urls',
                          'author.entities.url.urls', 'author.url', 'author.location', 'author.pinned_tweet_id',
                          'author.profile_image_url', 'author.protected', 'author.public_metrics.followers_count',
                          'author.public_metrics.following_count', 'author.public_metrics.listed_count',
                          'author.public_metrics.tweet_count', 'author.verified', 'author.verified_type',
                          'author.withheld.scope', 'author.withheld.copyright', 'author.withheld.country_codes',
                          'geo.coordinates.coordinates', 'geo.coordinates.type', 'geo.country', 'geo.country_code',
                          'geo.full_name', 'geo.geo.bbox', 'geo.geo.type', 'geo.id', 'geo.name', 'geo.place_id',
                          'geo.place_type', 'matching_rules', '__twarc.retrieved_at', '__twarc.url', '__twarc.version'])

        Path('test_resources/test.csv').unlink()
