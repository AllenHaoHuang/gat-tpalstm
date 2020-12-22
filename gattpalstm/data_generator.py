from abc import ABCMeta, abstractmethod
import logging
import os
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from gattpalstm.utils import check_path_exists
import networkx as nx


class DataGenerator(metaclass=ABCMeta):
    def __init__(self, para):
        # TODO CHANGE BACK
        self.DIRECTORY = "./data"
        self.para = para
        self.iterator = None

    def inputs(self, mode, batch_size, num_epochs=None):
        """Reads input data num_epochs times.
        Args:
        mode: String for the corresponding tfrecords ('train', 'validation', 'test')
        batch_size: Number of examples per returned batch.
        num_epochs: Number of times to read the input data, or 0/None to
        train forever.
        Returns:
        A tuple (images, labels), where:
        * images is a float tensor with shape [batch_size, 28, 28]
        in the range [0.0, 1.0].
        * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, mnist.NUM_CLASSES).
        This function creates a one_shot_iterator, meaning that it will only iterate
        over the dataset once. On the other hand there is no special initialization
        required.
        """
        if mode != "train" and mode != "validation" and mode != "test":
            raise ValueError("mode: {} while mode should be "
                             "'train', 'validation', or 'test'".format(mode))

        filename = self.DATA_PATH + "/" + mode + ".tfrecords"
        logging.info("Loading data from {}".format(filename))

        with tf.name_scope("input"):
            # TFRecordDataset opens a binary file and
            # reads one record at a time.
            # `filename` could also be a list of filenames,
            # which will be read in order.
            dataset = tf.data.TFRecordDataset(filename)

            # The map transformation takes a function and
            # applies it to every element
            # of the dataset.
            dataset = dataset.map(self._decode)
            for f in self._get_map_functions():
                dataset = dataset.map(f)

            # The shuffle transformation uses a finite-sized buffer to shuffle
            # elements in memory. The parameter is the number of elements in the
            # buffer. For completely uniform shuffling, set the parameter to be
            # the same as the number of elements in the dataset.
            if self.para.mode == "train":
                dataset = dataset.shuffle(1000 + 3 * batch_size)

            # dataset = dataset.repeat(num_epochs)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(batch_size)

            self.iterator = dataset.make_initializable_iterator()
            return self.iterator.get_next()

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_list_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _float_list_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @abstractmethod
    def _decode(self, serialized_example):
        pass

    @abstractmethod
    def _get_map_functions(self):
        pass


class TimeSeriesDataGenerator(DataGenerator):
    def __init__(self, para):
        DataGenerator.__init__(self, para)
        self.h = para.horizon
        self.DATA_PATH = os.path.join(self.DIRECTORY,
                                      para.data_set)
        self.split = [0, 0.6, 0.8, 1]
        self.split_names = ["train", "validation", "test"]
        # TODO CHANGE BACK AFTER TEST
        self.out_fn = "./data/preprocessed/close.csv"
        self._preprocess(para)
        self._setup_graph()
        del self.raw_dat, self.dat

    def _preprocess(self, para):
        self.forex_names = open(self.out_fn).readline().strip().split(",")[:para.num_var]
        self.raw_dat = np.loadtxt(self.out_fn, delimiter=",", skiprows=1)[:, :para.num_var]

        para.input_size = self.INPUT_SIZE = self.raw_dat.shape[1]
        self.rse = self._compute_rse()

        para.max_len = self.MAX_LEN = self.para.highway
        assert self.para.highway == self.para.attention_len
        para.output_size = self.OUTPUT_SIZE = self.raw_dat.shape[1]
        para.total_len = self.TOTAL_LEN = 1
        self.dat = np.zeros(self.raw_dat.shape)
        self.scale = np.ones(self.INPUT_SIZE)
        for i in range(self.INPUT_SIZE):
            mn = np.min(self.raw_dat[:, i])
            self.scale[i] = np.max(self.raw_dat) - mn
            self.dat[:, i] = (self.raw_dat[:, i] - mn) / self.scale[i]
        logging.info('rse = {}'.format(self.rse))
        for i in range(len(self.split) - 1):
            self._convert_to_tfrecords(self.split[i], self.split[i + 1],
                                       self.split_names[i])

    def _compute_rse(self):
        st = int(self.raw_dat.shape[0] * self.split[2])
        ed = int(self.raw_dat.shape[0] * self.split[3])
        Y = np.zeros((ed - st, self.INPUT_SIZE))
        for target in range(st, ed):
            Y[target - st] = self.raw_dat[target]
        return np.std(Y)

    def _convert_to_tfrecords(self, st, ed, name):
        st = int(self.dat.shape[0] * st)
        ed = int(self.dat.shape[0] * ed)
        out_fn = os.path.join(self.DATA_PATH, name + ".tfrecords")
        #if check_path_exists(out_fn):
        #    return
        with tf.python_io.TFRecordWriter(out_fn) as record_writer:
            for target in tqdm(range(st, ed)):
                end = target - self.h + 1
                beg = end - self.para.max_len

                if beg < 0:
                    continue
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "x":
                                self._float_list_feature(self.dat[beg:end].
                                                         flatten()),
                            "y":
                                self._float_list_feature(self.dat[target]),
                        }))
                record_writer.write(example.SerializeToString())

    def _get_map_functions(self):
        return []

    def _decode(self, serialized_example):
        example = tf.parse_single_example(
            serialized_example,
            features={
                "x":
                    tf.FixedLenFeature([self.MAX_LEN, self.INPUT_SIZE],
                                       tf.float32),
                "y":
                    tf.FixedLenFeature([self.OUTPUT_SIZE], tf.float32),
            },
        )
        rnn_input = tf.to_float(
            tf.reshape(example["x"], (self.MAX_LEN, self.INPUT_SIZE)))
        rnn_input_len = tf.constant(self.MAX_LEN, dtype=tf.int32)
        target_output = tf.expand_dims(tf.to_float(example["y"]), 0)
        target_output = tf.tile(target_output, [self.MAX_LEN, 1])
        return rnn_input, rnn_input_len, target_output

    def _setup_graph(self):
        """
        sets up networkx graph and
        """
        forex_graph = nx.Graph()
        for name in self.forex_names:
            forex_graph.add_node(name)

        for name1 in self.forex_names:
            for name2 in self.forex_names:
                if name1 != name2 and (name1[:3] in name2 or name1[3:] in name2):
                    forex_graph.add_edge(name1, name2)

        forex_adjacency_matrix = nx.adjacency_matrix(forex_graph, nodelist=self.forex_names).todense()
        forex_connections_matrix = np.asarray(forex_adjacency_matrix + np.identity(self.para.num_var), dtype=bool)

        self.graph = forex_graph
        self.graph_masks = []

        for i in range(self.para.num_var):
            self.graph_masks.append(forex_connections_matrix[i][:])

#
# if __name__ == "__main__":
#
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--attention_len', type=int, default=16)
#     parser.add_argument('--batch_size', type=int, default=32)
#     parser.add_argument('--data_set', type=str, default='tf')
#     parser.add_argument('--decay', type=int, default=0)
#     parser.add_argument('--dropout', type=float, default=0.2)
#     parser.add_argument('--file_output', type=int, default=1)
#     parser.add_argument('--highway', type=int, default=16)
#     parser.add_argument('--horizon', type=int, default=3)
#     parser.add_argument('--init_weight', type=float, default=0.1)
#     parser.add_argument('--learning_rate', type=float, default=1e-5)
#     parser.add_argument('--max_gradient_norm', type=float, default=5.0)
#     parser.add_argument('--mode', type=str, default='train')
#     parser.add_argument('--model_dir', type=str, default='./models/model')
#     parser.add_argument('--mts', type=int, default=1)
#     parser.add_argument('--num_epochs', type=int, default=40)
#     parser.add_argument('--num_layers', type=int, default=3)
#     parser.add_argument('--num_units', type=int, default=338)
#
#     para = parser.parse_args()
#     para.num_var = 5
#
#     a = TimeSeriesDataGenerator(para)
#     print(a.graph_masks)