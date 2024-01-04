from preprocess import preprocess_tweets, generate_test_data
import fire

if __name__ == '__main__':
    fire.Fire({'preprocess_tweets': preprocess_tweets,
               'generate_test_data': generate_test_data})