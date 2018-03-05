import tensorflow as tf
import numpy as np
import argparse
import model
import data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--model_dir', default='./train', type=str, help='training data')
parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')
parser.add_argument('--train_dir_sketch', required=True, type=str, help='training data')
parser.add_argument('--train_dir_color', required=True, type=str, help='training data')
parser.add_argument('--eval_dir_sketch', required=True, type=str, help='training data')
parser.add_argument('--eval_dir_color', required=True, type=str, help='training data')

def main(argv):
    args = parser.parse_args(argv[1:])
    
    estimator = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=args.model_dir,
        params={
        })

    estimator.train(
        input_fn=lambda:data_loader.input_fn(args.train_dir_sketch, args.train_dir_color, args.batch_size),
        steps=args.train_steps
    )

    estimator.eval(
        input_fn=lambda:data_loader.input_fn(args.eval_dir_sketch, args.eval_dir_color, args.batch_size)
    )

if __name__ == '__main__':
    tf.app.run(main)