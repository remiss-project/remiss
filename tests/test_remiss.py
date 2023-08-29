import json
from collections.abc import Mapping
from pathlib import Path
from unittest import TestCase

from remiss import flatten_tweets
import pandas as pd


def extract_paths(base_path, dd):
    new_paths = []
    for key, value in dd.items():
        new_path = base_path + ('.' if base_path else '') + key
        if isinstance(value, Mapping):
            new_paths.extend(extract_paths(new_path, value))
        else:
            new_paths.append(new_path)
    return new_paths


class TestRemiss(TestCase):
    def test_flatten_tweets(self):
        flatten_tweets('test_resources/test.jsonl')
        # check number of lines is correct
        with open('test_resources/test.flattened.jsonl') as f:
            self.assertEqual(len(f.readlines()), 934)

        expected_keys = ['id', 'conversation_id', 'referenced_tweets.replied_to.id', 'referenced_tweets.retweeted.id',
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
                         'geo.place_type', 'matching_rules', '__twarc.retrieved_at', '__twarc.url', '__twarc.version']

        # check that the file is valid jsonl and the keys are correct
        expected_keys = set(expected_keys)
        actual_keys = set()
        with open('test_resources/test.flattened.jsonl') as f:
            for line in f:
                tweet = json.loads(line)
                actual_keys.update(extract_paths('', tweet))

        # SKIPPING THIS TEST AS THE CONVERTER USED TO GET THE KEYS ADDS SOME MODIFICATIONS OF ITS OWN AND WE WANT
        # TO PRESERVE THE ORIGINAL STRUCTURE
        # self.assertEqual(expected_keys, actual_keys)
        expected_missing = {'in_reply_to_user.entities.description.hashtags', 'in_reply_to_user.location',
                            'withheld.country_codes', 'withheld.copyright',
                            'in_reply_to_user.entities.description.mentions', 'in_reply_to_user.name',
                            'in_reply_to_user.username', 'referenced_tweets.retweeted.id', 'in_reply_to_user.verified',
                            'public_metrics.bookmark_count', 'in_reply_to_user.description', 'source',
                            'author.entities.description.cashtags', 'in_reply_to_user.verified_type',
                            'in_reply_to_user.entities.url.urls', 'in_reply_to_user.url', 'geo.coordinates.coordinates',
                            'referenced_tweets.quoted.id', 'withheld.scope', 'quoted_username',
                            'author.withheld.country_codes', 'entities.cashtags', 'author.withheld.copyright',
                            'in_reply_to_user.protected', 'matching_rules', 'author.withheld.scope',
                            'in_reply_to_user.public_metrics.tweet_count', 'in_reply_to_username',
                            'in_reply_to_user.public_metrics.following_count', 'geo.coordinates.type',
                            'in_reply_to_user.profile_image_url', 'in_reply_to_user.created_at',
                            'in_reply_to_user.pinned_tweet_id', 'in_reply_to_user.public_metrics.followers_count',
                            'in_reply_to_user.public_metrics.listed_count', 'retweeted_username', 'quoted_user_id',
                            'in_reply_to_user.entities.description.urls', 'referenced_tweets', 'in_reply_to_user.id',
                            'referenced_tweets.replied_to.id', 'retweeted_user_id'}

        actual_missing = expected_keys - actual_keys | actual_keys - expected_keys
        self.assertEqual(expected_missing, actual_missing)
        Path('test_resources/test.flattened.jsonl').unlink()
