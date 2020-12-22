import os
import tensorflow as tf

from gattpalstm.setup import params_setup, logging_config_setup, config_setup
from gattpalstm.model_utils import create_graph, load_weights, print_num_of_trainable_parameters
from gattpalstm.train import train
from gattpalstm.test import test


def main():
    para = params_setup()
    logging_config_setup(para)

    for num_var in range(2, 20 + 1):

        para.num_var = num_var
        graph, model, data_generator = create_graph(para)

        with tf.Session(config=config_setup(), graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            # load_weights(para, sess, model)
            print_num_of_trainable_parameters()

            try:
                if para.mode == 'train':
                    train(para, sess, model, data_generator)
                elif para.mode == 'test':
                    test(para, sess, model, data_generator)

            except KeyboardInterrupt:
                print('KeyboardInterrupt')
            finally:
                print('Stop')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
