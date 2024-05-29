from app import prepopulate
from preprocess import preprocess_tweets, generate_test_data, validate_fact_checking_dataset_data
import fire


if __name__ == '__main__':
    fire.Fire({'preprocess_tweets': preprocess_tweets,
               'generate_test_data': generate_test_data,
               'prepopulate': prepopulate,
               'validate_fact_checking_dataset_data': validate_fact_checking_dataset_data})
