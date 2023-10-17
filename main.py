from preprocess import preprocess_tweets
import fire

if __name__ == '__main__':
    fire.Fire({'preprocess_tweets': preprocess_tweets})