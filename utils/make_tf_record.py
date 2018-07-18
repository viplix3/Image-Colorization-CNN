import tensorflow as tf
import os
import numpy as np
import threading
from datetime import datetime
from skimage.color import rgb2lab
import cv2

tf.app.flags.DEFINE_string('output_dir', './../TFRecords',
                           'output directory containing tfrecords')
tf.app.flags.DEFINE_integer('train_threads', 4, 'Number of threads for\
                            processing train TFRecord')
tf.app.flags.DEFINE_integer('val_threads', 1, 'Number of threads for processing\
                            validation TFRecord')
tf.app.flags.DEFINE_integer('train_shards', 8, 'Number of shards for traning\
                            TFRecord')
tf.app.flags.DEFINE_integer('val_shards', 1, 'Number of shards for validation\
                            TFRecord')
tf.app.flags.DEFINE_string('data_dir', './../Dataset/', 'Data Directory')
tf.app.flags.DEFINE_integer('img_size', 512, 'Size of the image for model')

FLAGS = tf.app.flags.FLAGS

if not (os.path.exists(FLAGS.output_dir)):
    os.mkdir(FLAGS.output_dir)


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _process_image(filename):

    filename = os.path.join(FLAGS.data_dir, filename)
    image_data = cv2.imread(filename)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    image_data = cv2.resize(image_data, (FLAGS.img_size, FLAGS.img_size))

    LAB_image = rgb2lab(image_data)
    X = LAB_image[:, :, 0].reshape(FLAGS.img_size, FLAGS.img_size, 1)
    Y = LAB_image[:, :, 1:].reshape(FLAGS.img_size, FLAGS.img_size, 2)

    X = X / 100.0
    Y = Y / 128.0
    return X, Y


def convert_to_example(image_data, output_data):

    image_raw = image_data.tostring()
    output_raw = output_data.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image_data': bytes_feature(image_raw),
        'output_data': bytes_feature(output_raw)
    }))

    return example


def tfrecord_batches(mode, file_names, thread_idx, ranges, num_shards, num_threads, split):

    num_shards_per_thread = int(num_shards/num_threads)
    shard_ranges = np.linspace(ranges[thread_idx][0], ranges[thread_idx][1],
                               num_shards_per_thread+1).astype(np.int)
    num_files_in_thread = ranges[thread_idx][1] - ranges[thread_idx][0]

    counter = 0
    for s in range(num_shards_per_thread):
        shard = thread_idx * num_shards_per_thread + s
        output_filename = '%s-%.5d-%.5d.tfrecord' % (mode, shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_count = 0
        files_in_shards = np.arange(shard_ranges[s], shard_ranges[s+1],
                                    dtype=int)

        for i in files_in_shards:
            if mode == 'val':
                filename = file_names[i % split]
            else:
                filename = file_names[i]
            image_data, output_data = _process_image(filename)

            example = convert_to_example(image_data, output_data)

            writer.write(example.SerializeToString())
            shard_count += 1
            counter += 1

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' % (datetime.now(),
                                                         thread_idx, shard_count, output_file))
        shard_count = 0

    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_idx, counter, num_files_in_thread))


def process_tfrecord(mode, file_names, num_shards, num_threads, split=0):

    spacing = np.linspace(split, len(file_names)+split, num_threads+1).astype(np.int)
    ranges = []
    for i in range(len(spacing)-1):
        ranges.append([spacing[i], spacing[i+1]])

    print('Launching %d threads for spacings: %s' % (num_threads, ranges))

    coord = tf.train.Coordinator()
    threads = []

    for thread_idx in range(len(ranges)):
        args = (mode, file_names, thread_idx, ranges, num_shards, num_threads,
                split)
        t = threading.Thread(target=tfrecord_batches, args=args)
        t.start()
        threads.append(t)

    coord.join(threads)
    print('%s: Finished writing all %d images in dataset'
          % (datetime.now(), len(file_names)))


def make_tfrecord():

    assert not FLAGS.train_shards % FLAGS.train_threads, ('Please make \
    FLAGS.train_shards conssumerate with FLAGS.train_threads')

    assert not FLAGS.val_shards % FLAGS.val_threads, ('Please make \
    FLAGS.val_shards consumerate with FLAGS.val_threads')

    file_names = os.listdir(FLAGS.data_dir)
    split = int(0.1*len(file_names))
    train_filenames = file_names[:-split]
    val_filenames = file_names[-split:]

    print('Preparing training data')
    process_tfrecord('train', train_filenames, FLAGS.train_shards,
                     FLAGS.train_threads)

    print('Prepating validation data')
    process_tfrecord('val', val_filenames, FLAGS.val_shards,
                     FLAGS.val_threads, len(file_names)-split)


if __name__ == '__main__':

    print('Saving results to %s' % FLAGS.output_dir)
    make_tfrecord()
