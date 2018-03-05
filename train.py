import tensorflow as tf
import numpy as np
import argparse
import model

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--model_dir', default='./train', type=str, help='training data')
parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])
    
    estimator = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=args.model_dir,
        params={
        })

if __name__ == '__main__':
    tf.app.run(main)