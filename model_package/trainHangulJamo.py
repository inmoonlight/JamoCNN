import tensorflow as tf
import numpy as np
import json
import os
import sys
sys.path.append('../../')
import time

import datetime

# My own alarm package
from alarmoon import Alarmoon
from filterVisualization import FilterVisualization
from JamoCNN import JamoCNN
from preprocessing import Hangulization

filter_vis = FilterVisualization()

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
preprocess = Hangulization()
x, y = preprocess.load_data()
# Randomly shuffle data
np.random.seed(9)
shuffle_indices = np.random.permutation(np.arange(len(x)))
x_shuffled = [x[shuffled_idx] for shuffled_idx in shuffle_indices]
y_shuffled = y[shuffle_indices]
# Split train/test set
n_dev_samples = 30000
# TODO: Create a fuckin' correct cross validation procedure
x_train, x_dev = x_shuffled[:-n_dev_samples], x_shuffled[-n_dev_samples:]
y_train, y_dev = y_shuffled[:-n_dev_samples], y_shuffled[-n_dev_samples:]
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = JamoCNN()

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        directory = "/home/jihyung/Dropbox/DataScience/JamoCNN/"
        out_dir = os.path.abspath(os.path.join(directory, "JamoCNN_hangulize", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)
            
#            with tf.variable_scope('conv-maxpool-1', reuse=True) as scope_conv:
#                scope_conv.reuse_variables()
#                weights = tf.get_variable('W')
#                grid = filter_vis.put_kernels_on_grid(weights)
#                grid_val = sess.run(grid, feed_dict)
#                train_summary_W1 = tf.summary.image('conv-1/features', grid_val, max_outputs=1)
#                train_summary_writer.add_summary(train_summary_W1, step)
            
        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            dev_size = len(x_batch)
            max_batch_size = 128
            num_batches = dev_size//max_batch_size
            acc = []
            losses = []
            print("Number of batches in dev set is " + str(num_batches))
            for i in range(num_batches):
                x_batch_dev, y_batch_dev = preprocess.get_batched_one_hot(
                    x_batch, y_batch, i * max_batch_size, (i + 1) * max_batch_size)
                feed_dict = {
                  cnn.input_x: x_batch_dev,
                  cnn.input_y: y_batch_dev,
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                acc.append(accuracy)
                losses.append(loss)
                time_str = datetime.datetime.now().isoformat()
                print("batch " + str(i + 1) + " in dev >>" +
                      " {}: loss {:g}, acc {:g}".format(time_str, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
            print("\nMean accuracy=" + str(sum(acc)/len(acc)))
            print("Mean loss=" + str(sum(losses)/len(losses)))


        # Generate batches in one-hot-encoding format
        batches = preprocess.batch_iter(x_train, y_train, FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

                
# Send finish message (This part should be omitted!)
alarm = Alarmoon()
alarm.send_message('Hangulize + JamoCNN is done!')