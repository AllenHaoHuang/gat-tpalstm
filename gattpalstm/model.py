import logging
import tensorflow as tf
from tensorflow.layers import dense
from gattpalstm.attention_wrapper import TemporalPatternAttentionCellWrapper
import numpy as np


class GATTPALSTM:
    def __init__(self, para, data_generator):
        self.data_generator = data_generator
        self.para = para
        self.dtype = tf.float32
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.graph_masks = self.data_generator.graph_masks

        self._build_graph()
        if self.para.mode == "train":
            self._build_optimizer()

        self.saver = tf.train.Saver(max_to_keep=self.para.num_epochs)

    def _build_graph(self):
        logging.debug("Building graph")

        # rnn_inputs: [batch_size, max_len, input_size]
        # rnn_inputs_len: [batch_size]
        # target_outputs: [batch_size, max_len, output_size]
        self.rnn_inputs, self.rnn_inputs_len, self.target_outputs = self.data_generator.inputs(
            self.para.mode, self.para.batch_size)

        #TODO recode for efficiency to allow parallelization
        for graph_layer_index in range(self.para.num_graph_layers):
            print(graph_layer_index)

            if graph_layer_index == 0:
                nodes_rnn_inputs = self.rnn_inputs
                rnn_length = self.para.max_len
            else:
                rnn_length = self.para.num_units

            nodes_rnn_outputs = []

            for node_index in range(self.para.num_var):
                # graph mask : [num_var]
                graph_mask = self.graph_masks[node_index]

                # embedding size
                num_node_connections = np.sum(graph_mask)

                # node_rnn_inputs : [batch_size, max_len, num_node_connections]
                node_rnn_inputs = tf.reshape(tf.boolean_mask(nodes_rnn_inputs, graph_mask, axis=2),
                                             (-1, rnn_length, num_node_connections))

                node_rnn_inputs_embed = tf.nn.relu(dense(node_rnn_inputs, self.para.num_units))
                node_rnn_inputs_embed = tf.unstack(node_rnn_inputs_embed, axis=1)

                _, node_rnn_states = tf.nn.static_rnn(
                    cell=self._build_rnn_cell(),
                    inputs=node_rnn_inputs_embed,
                    sequence_length=self.rnn_inputs_len,
                    dtype=self.dtype,
                    scope="graph" + str(graph_layer_index + 1) + "node" + str(node_index + 1) + "lstm"
                )
                node_rnn_states = tf.concat(
                    [node_rnn_states[i][1] for i in range(self.para.num_layers)],
                    1,
                )

                # rnn_output = dense(node_rnn_states, 1)
                nodes_rnn_outputs.append(node_rnn_states)

            # if not last layer reload rnn outputs as input
            if graph_layer_index < self.para.num_graph_layers - 1:
                nodes_rnn_inputs = tf.stack(nodes_rnn_outputs, axis=2)
            else:
                self.final_rnn_states = tf.stack(nodes_rnn_outputs, axis=2)

        all_rnn_outputs = []
        for node_index in range(self.para.num_var):
            # graph mask : [num_var]
            graph_mask = self.graph_masks[node_index]

            # embedding size
            num_node_connections = np.sum(graph_mask)

            node_rnn_outputs = tf.reshape(tf.boolean_mask(self.final_rnn_states, graph_mask, axis=2),
                                         (-1, rnn_length * num_node_connections))
            node_output = dense(node_rnn_outputs, 1)
            all_rnn_outputs.append(node_output)
        self.all_rnn_outputs = tf.concat(all_rnn_outputs, axis=1)


        if self.para.highway > 0:
            reg_outputs = tf.transpose(
                self.rnn_inputs[:, -self.para.highway:, :], [0, 2, 1])
            reg_outputs = dense(reg_outputs, 1)
            self.all_rnn_outputs += tf.squeeze(reg_outputs)

        if self.para.mode == "train" or self.para.mode == "validation":
            self.labels = self.target_outputs[:, self.para.max_len - 1, :]
            self.loss = self._compute_loss(
                outputs=self.all_rnn_outputs, labels=self.labels)
        elif self.para.mode == "test":
            self.labels = self.target_outputs[:, self.para.max_len - 1, :]
            if not self.para.mts:
                self.all_rnn_outputs = tf.sigmoid(self.all_rnn_outputs)

    def _build_optimizer(self):
        logging.debug("Building optimizer")

        trainable_variables = tf.trainable_variables()
        if self.para.decay > 0:
            lr = tf.train.exponential_decay(
                self.para.learning_rate,
                self.global_step,
                self.para.decay,
                0.995,
                staircase=True,
            )
        else:
            lr = self.para.learning_rate
        self.opt = tf.train.AdamOptimizer(lr)
        gradients = tf.gradients(self.loss, trainable_variables)
        clip_gradients, _ = tf.clip_by_global_norm(gradients,
                                                   self.para.max_gradient_norm)
        self.update = self.opt.apply_gradients(
            zip(clip_gradients, trainable_variables),
            global_step=self.global_step,
        )

    def _compute_loss(self, outputs, labels):
        """
        outputs: [batch_size, output_size]
        labels: [batch_size, output_size]
        """
        if self.para.mts:
            loss = tf.reduce_mean(
                tf.losses.absolute_difference(
                    labels=labels, predictions=outputs))
        else:
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=outputs, labels=labels))
        return loss

    def _build_single_cell(self):
        cell = tf.contrib.rnn.LSTMBlockCell(self.para.num_units)
        if self.para.mode == "train":
            cell = tf.contrib.rnn.DropoutWrapper(
                cell=cell,
                input_keep_prob=(1.0 - self.para.dropout),
                output_keep_prob=(1.0 - self.para.dropout),
                state_keep_prob=(1.0 - self.para.dropout),
            )
        cell = TemporalPatternAttentionCellWrapper(
            cell,
            self.para.attention_len,
        )
        return cell

    def _build_rnn_cell(self):
        return tf.contrib.rnn.MultiRNNCell(
            [self._build_single_cell() for _ in range(self.para.num_layers)])
