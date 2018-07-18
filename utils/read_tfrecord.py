import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def list_tfrecord_file(folder, file_list):

    
	tfrecord_list = []
	for i in range(len(file_list)):
		current_file_abs_path = os.path.join(folder, file_list[i])		
		if current_file_abs_path.endswith(".tfrecord"):
			tfrecord_list.append(current_file_abs_path)
			print("Found %s successfully!" % file_list[i])				
		else:
			pass
	return tfrecord_list


	
# Traverse current directory
def tfrecord_auto_traversal(folder, current_folder_filename_list):

	if current_folder_filename_list != None:
		print("%s files were found under %s folder. " % (len(current_folder_filename_list), folder))
		print("Please be noted that only files ending with '*.tfrecord' will be loaded!")
		tfrecord_list = list_tfrecord_file(folder, current_folder_filename_list)
		if len(tfrecord_list) != 0:
			print("Found %d files:\n %s\n\n\n" %(len(tfrecord_list), current_folder_filename_list))
		else:
			print("Cannot find any tfrecord files, please check the path.")
	return tfrecord_list



# def read_tf_records(filename, img_w=512, img_h=512, num_channels=1, batch_size=32):

# 		features = {'image_data': tf.FixedLenFeature([], tf.string),
# 					'output_data': tf.FixedLenFeature([], tf.string)}

# 		min_queue_examples = 100

# 		filename_queue = tf.train.string_input_producer([filename], num_epochs=None)

# 		reader = tf.TFRecordReader()
# 		_, serialized_example = reader.read(filename_queue)

# 		batch = tf.train.shuffle_batch([serialized_example], batch_size=batch_size, capacity=min_queue_examples+10*batch_size, num_threads=4, 
# 				min_after_dequeue=min_queue_examples)
# 		parsed_example = tf.parse_example(batch, features=features)

# 		image_raw = tf.decode_raw(parsed_example['image_data'], tf.float64)
# 		images = tf.cast(tf.reshape(image_raw, [batch_size, img_w, img_h, num_channels]), tf.float64)

# 		output_raw = tf.decode_raw(parsed_example['output_data'], tf.float64)
# 		output = tf.cast(tf.reshape(output_raw, [batch_size, img_w, img_h, num_channels+1]), tf.float64)

# 		return(images, output)



def read_tf_records(filename, img_w=512, img_h=512, num_channels=1, batch_size=32):

		features = {'image_data': tf.FixedLenFeature([], tf.string),
					'output_data': tf.FixedLenFeature([], tf.string)}

		min_queue_examples = 100

		for i in range(len(filename)):

			filename_queue = tf.train.string_input_producer([filename[i]], num_epochs=None)

		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)

		batch = tf.train.shuffle_batch([serialized_example], batch_size=batch_size, capacity=min_queue_examples+10*batch_size, num_threads=4, 
				min_after_dequeue=min_queue_examples)
		parsed_example = tf.parse_example(batch, features=features)

		image_raw = tf.decode_raw(parsed_example['image_data'], tf.float64)
		images = tf.cast(tf.reshape(image_raw, [batch_size, img_w, img_h, num_channels]), tf.float64)

		output_raw = tf.decode_raw(parsed_example['output_data'], tf.float64)
		output = tf.cast(tf.reshape(output_raw, [batch_size, img_w, img_h, num_channels+1]), tf.float64)

		return(images, output)