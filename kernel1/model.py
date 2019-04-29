from dependencies import *
# from loaddata import *


data = dict()

with open(data_save_path, 'rb') as f:
   data = pickle.load(f)
f.close()

X_train = data['X_train']
X_dev = data['X_dev']
X_test = data['X_test']

y_tra = data['y_tra']
y_dev = data['y_dev']
y_test = data['y_test']

embedding_matrix = data['embedding']

pos_index = data['position']


def _dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        # todo: check that this is correct
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class AttentionWeight(Layer):
    """
        This code is a modified version of cbaziotis implementation:  GithubGist cbaziotis/AttentionWithContext.py
        Attention operation, with a context/query vector, for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, steps)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWeight())
        """

    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWeight, self).__init__(**kwargs)

    def build(self, input_shape):
        shape1 = input_shape[0]
        shape2 = input_shape[1]

        self.W = self.add_weight((shape2[-1], shape1[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((shape2[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)


    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, inputs, mask=None):
        x = inputs[0]
        u = inputs[1]

        uit = _dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = K.batch_dot(uit, u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        return a

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Dot` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])

        return shape1[0], shape1[1]

    def get_config(self):
        config = {
            'W_regularizer': regularizers.serialize(self.W_regularizer),
            'b_regularizer': regularizers.serialize(self.b_regularizer),
            'W_constraint': constraints.serialize(self.W_constraint),
            'b_constraint': constraints.serialize(self.b_constraint),
            'bias': self.bias
        }
        base_config = super(AttentionWeight, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class FeatureSelection1D(Layer):
    """
        Normalize feature along a specific axis.
        Supports Masking.

        # Input shape
            A ND tensor with shape: `(samples, timesteps, features)
            A 2D tensor with shape: [samples, num_selected_features]
        # Output shape
            ND tensor with shape: `(samples, num_selected_features, features)`.
        :param kwargs:
        """

    def __init__(self, num_selects, **kwargs):
        self.num_selects = num_selects
        self.supports_masking = True
        super(FeatureSelection1D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(FeatureSelection1D, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # don't pass the mask to the next layers
        return None

    def call(self, inputs, mask=None):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError('FeatureSelection1D layer should be called '
                             'on a list of 2 inputs.')

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a = K.cast(mask, K.floatx()) * inputs[0]
        else:
            a = inputs[0]

        b = inputs[1]

        a = tf.batch_gather(
            a, b
        )

        return a

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `FeatureSelection1D` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])

        if shape2[0] != shape1[0]:
            raise ValueError("batch size must be same")

        if shape2[1] != self.num_selects:
            raise ValueError("must conform to the num_select")

        return (shape1[0], self.num_selects, shape1[2])

    def get_config(self):
        config = {
            'num_selects': self.num_selects
        }
        base_config = super(FeatureSelection1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

histories = list()
file_paths = list()
cos = list()


def build_e2e_birnn_attention_model(
        voca_dim, time_steps, pos_tag_size, pos_tag_dim, extra_feature_dims, output_dim, rnn_dim, model_dim, mlp_dim,
        item_embedding=None, rnn_depth=1, mlp_depth=1,
        drop_out=0.5, rnn_drop_out=0., rnn_state_drop_out=0.,
        trainable_embedding=False, gpu=False, return_customized_layers=False):
    """
    Create A End-to-End Bidirectional RNN Attention Model.

    :param voca_dim: vocabulary dimension size.
    :param time_steps: the length of input
    :param extra_feature_dims: the dimention size of the auxilary feature
    :param output_dim: the output dimension size
    :param model_dim: rrn dimension size
    :param mlp_dim: the dimension size of fully connected layer
    :param item_embedding: integer, numpy 2D array, or None (default=None)
        If item_embedding is a integer, connect a randomly initialized embedding matrix to the input tensor.
        If item_embedding is a matrix, this matrix will be used as the embedding matrix.
        If item_embedding is None, then connect input tensor to RNN layer directly.
    :param rnn_depth: rnn depth
    :param mlp_depth: the depth of fully connected layers
    :param drop_out: dropout rate of fully connected layers
    :param rnn_drop_out: dropout rate of rnn layers
    :param rnn_state_drop_out: dropout rate of rnn state tensor
    :param trainable_embedding: boolean
    :param gpu: boolean, default=False
        If True, CuDNNLSTM is used instead of LSTM for RNN layer.
    :param return_customized_layers: boolean, default=False
        If True, return model and customized object dictionary, otherwise return model only
    :return: keras model
    """

    # sequences inputs
    if item_embedding is not None:
        inputp = models.Input(shape=(time_steps,), dtype='int32', name='inputp')
        inputa = models.Input(shape=(time_steps,), dtype='int32', name='inputa')
        inputb = models.Input(shape=(time_steps,), dtype='int32', name='inputb')
        inputs = [inputp, inputa, inputb]

        if isinstance(item_embedding, np.ndarray):
            assert voca_dim == item_embedding.shape[0]
            embed_dim = item_embedding.shape[1]
            emb_layer = layers.Embedding(
                voca_dim, item_embedding.shape[1], input_length=time_steps,
                weights=[item_embedding, ], trainable=trainable_embedding,
                mask_zero=False, name='embedding_layer0'
            )
        elif utils.is_integer(item_embedding):
            embed_dim = item_embedding
            emb_layer = layers.Embedding(
                voca_dim, item_embedding, input_length=time_steps,
                trainable=trainable_embedding,
                mask_zero=False, name='embedding_layer0'
            )
        else:
            raise ValueError("item_embedding must be either integer or numpy matrix")

        xs = list(map(
            lambda input_: emb_layer(input_),
            inputs
        ))
    else:
        inputp = models.Input(shape=(time_steps, voca_dim), dtype='float32', name='inputp')
        inputa = models.Input(shape=(time_steps, voca_dim), dtype='float32', name='inputa')
        inputb = models.Input(shape=(time_steps, voca_dim), dtype='float32', name='inputb')
        embed_dim = voca_dim
        xs = [inputp, inputa, inputb]

    # pos tag
    inputposp = models.Input(shape=(time_steps,), dtype='int32', name='inputposp')
    inputposa = models.Input(shape=(time_steps,), dtype='int32', name='inputposa')
    inputposb = models.Input(shape=(time_steps,), dtype='int32', name='inputposb')
    inputpos = [inputposp, inputposa, inputposb]
    pos_emb_layer = layers.Embedding(
        pos_tag_size, pos_tag_dim, input_length=time_steps,
        trainable=True, mask_zero=False, name='pos_embedding_layer0'
    )
    xpos = list(map(
        lambda input_: pos_emb_layer(input_),
        inputpos
    ))

    embed_concate_layer = layers.Concatenate(axis=2, name="embed_concate_layer")
    for i in range(len(xs)):
        xs[i] = embed_concate_layer([xs[i], xpos[i]])

    # mention position in the sentence
    inputpi = models.Input(shape=(1,), dtype='int32', name='inputpi')
    inputai = models.Input(shape=(1,), dtype='int32', name='inputai')
    inputbi = models.Input(shape=(1,), dtype='int32', name='inputbi')
    xis = [inputpi, inputai, inputbi]

    # addtional mention-pair features
    inputpa = models.Input(shape=(extra_feature_dims,), dtype='float32', name='inputpa')
    inputpb = models.Input(shape=(extra_feature_dims,), dtype='float32', name='inputpb')
    xextrs = [inputpa, inputpb]

    # rnn
    birnns = list()
    rnn_batchnorms = list()
    rnn_dropouts = list()
    if gpu:
        # rnn encoding
        for i in range(rnn_depth):
            rnn_dropout = layers.SpatialDropout1D(rnn_drop_out)
            birnn = layers.Bidirectional(
                layers.CuDNNGRU(rnn_dim, return_sequences=True),
                name='bi_lstm_layer' + str(i))
            rnn_batchnorm = layers.BatchNormalization(name='rnn_batch_norm_layer' + str(i))

            birnns.append(birnn)
            rnn_dropouts.append(rnn_dropout)
            rnn_batchnorms.append(rnn_batchnorm)

        xs_ = list()
        for x_ in xs:
            for i in range(len(birnns)):
                x_ = rnn_dropouts[i](x_)
                x_ = birnns[i](x_)
                x_ = rnn_batchnorms[i](x_)
            xs_.append(x_)
        xs = xs_
    else:
        # rnn encoding
        for i in range(rnn_depth):
            birnn = layers.Bidirectional(
                layers.GRU(rnn_dim, return_sequences=True, dropout=rnn_drop_out,
                           recurrent_dropout=rnn_state_drop_out),
                name='bi_lstm_layer' + str(i))
            rnn_batchnorm = layers.BatchNormalization(name='rnn_batch_norm_layer' + str(i))

            birnns.append(birnn)
            rnn_batchnorms.append(rnn_batchnorm)

        xs_ = list()
        for x_ in xs:
            for i in range(len(birnns)):
                x_ = birnns[i](x_)
                x_ = rnn_batchnorms[i](x_)
            xs_.append(x_)
        xs = xs_

    # attention aggregated rnn embedding + mention rnn embedding + mention-pair features
    select_layer = FeatureSelection1D(1, name='boundary_selection_layer')
    flatten_layer1 = layers.Flatten('channels_first', name="flatten_layer1")
    permute_layer = layers.Permute((2, 1), name='permuted_attention_x')
    attent_weight = AttentionWeight(name="attention_weight")
    focus_layer = layers.Dot([2, 1], name='focus' + '_layer')
    reshape_layer = layers.Reshape((1, rnn_dim * 2), name="reshape_layer")
    concate_layer = layers.Concatenate(axis=1, name="attention_concate_layer")
    atten_dropout_layer = layers.Dropout(drop_out, name='attention_dropout_layer')
    map_layer1 = layers.Dense(model_dim, activation="relu", name="map_layer1")
    # map_layer2 = layers.TimeDistributed(layers.Dense(model_dim, activation="relu"), name="map_layer2")
    map_layer2 = map_layer1
    flatten_layer = layers.Flatten('channels_first', name="flatten_layer")
    for i in range(len(xs)):
        if i == 0:
            map_layer = map_layer1
        else:
            map_layer = map_layer2

        select_ = select_layer([xs[i], xis[i]])
        flatten_select_ = flatten_layer1(select_)
        att = attent_weight([xs[i], flatten_select_])

        focus = focus_layer([permute_layer(xs[i]), att])
        xs[i] = concate_layer([select_, reshape_layer(focus)])
        xs[i] = flatten_layer(xs[i])
        xs[i] = atten_dropout_layer(xs[i])
        xs[i] = map_layer(xs[i])

    feature_dropout_layer = layers.Dropout(rate=drop_out, name="feature_dropout_layer")
    feature_map_layer = layers.Dense(model_dim, activation="relu", name="feature_map_layer")
    xextrs = [feature_map_layer(feature_dropout_layer(xextr)) for xextr in xextrs]

    x = layers.Concatenate(axis=1, name="concat_feature_layer")(xs + xextrs)
    x = layers.Dropout(drop_out, name='dropout_layer')(x)

    # MLP Layers
    for i in range(mlp_depth - 1):
        x = layers.Dense(mlp_dim, activation='selu', kernel_initializer='lecun_normal', name='selu_layer' + str(i))(x)
        x = layers.AlphaDropout(drop_out, name='alpha_layer' + str(i))(x)

    outputs = layers.Dense(output_dim, activation="softmax", name="softmax_layer0")(x)

    model = models.Model([inputp, inputa, inputb] + inputpos + xis + [inputpa, inputpb], outputs)

    if return_customized_layers:
        return model, {'FeatureSelection1D': FeatureSelection1D, 'AttentionWeight': AttentionWeight}

    return model

# Build Model

voca_dim = embedding_matrix.shape[0]
# voca_dim = 22297

pos_tag_size = len(pos_index) + 1
# pos_tag_size = 18

# time_steps = max_len
time_steps = 50

embed_dim = embedding_matrix.shape[1]
# embed_dim = 300

pos_tag_dim = 5

# extra_feature_dims = num_pos_features
extra_feature_dims = 45

output_dim = 3
rnn_dim = 50
model_dim = 10
mlp_dim = 10
rnn_depth = 1
mlp_depth=1
drop_out=0.2
rnn_drop_out=0.5
gpu = False
return_customized_layers=True

model, co = build_e2e_birnn_attention_model(
        voca_dim, time_steps, pos_tag_size, pos_tag_dim, extra_feature_dims, output_dim, rnn_dim, model_dim, mlp_dim,
        item_embedding=embedding_matrix, rnn_depth=rnn_depth, mlp_depth=mlp_depth,
        drop_out=drop_out, rnn_drop_out=rnn_drop_out, rnn_state_drop_out=rnn_drop_out,
        trainable_embedding=False, gpu=gpu, return_customized_layers=return_customized_layers)
cos.append(co)


# Train model
adam = ko.Nadam(clipnorm=1.0)
model.compile(adam, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])

file_path = "best_e2e_rnn_atten_model.hdf5"
check_point = kc.ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1, save_best_only = True, mode = "min")
early_stop = kc.EarlyStopping(monitor = "val_loss", mode = "min", patience=5)
history = model.fit(X_train, y_tra, batch_size=30, epochs=40, validation_data=(X_dev, y_dev), callbacks = [check_point, early_stop])

file_paths.append(file_path)
histories.append(np.min(np.asarray(history.history['val_loss'])))

del model, history
gc.collect()


# Make prediction

# print("load best model: " + str(file_paths[np.argmin(histories)]))
# model = models.load_model(
#     file_paths[np.argmin(histories)], cos[np.argmin(histories)])

# print(cos[np.argmin(histories)])
model = models.load_model(
    "best_e2e_rnn_atten_model.hdf5",{'FeatureSelection1D': FeatureSelection1D, 'AttentionWeight': AttentionWeight})


# for i in range(len(X_test)):
#     # X_test[i] = np.delete(X_test[i], 1024, 0)
#     X_test[i] = X_test[i][1024:1027]

y_preds_tra = model.predict(X_train, batch_size = 1024, verbose = 1)
y_preds_dev = model.predict(X_dev, batch_size = 1024, verbose = 1)
y_preds_test = model.predict(X_test, batch_size = 1024, verbose = 1)
# y_preds_test = np.zeros((2000,3))
# X_test_tmp = [None] * 11
# for i in range(2000):
#     for j in range(len(X_test)):
#         X_test_tmp[j] = X_test[j][i]
#     y_preds_test[i] = model.predict(X_test_tmp, batch_size=1024, verbose=1)

y_preds_value_tra = np.argmax(y_preds_tra,axis=1)
y_preds_value_dev = np.argmax(y_preds_dev,axis=1)
y_preds_value_test = np.argmax(y_preds_test,axis=1)

predvals_tra = np.array(y_preds_value_tra)
predvals_dev = np.array(y_preds_value_dev)
predvals_test = np.array(y_preds_value_test)

truevals_tra = np.array(y_tra)
truevals_dev = np.array(y_dev)
truevals_test = np.array(y_test)

pred_dic_tra = dict()
pred_dic_dev = dict()
pred_dic_test = dict()

pred_dic_tra['prediction'] = predvals_tra
pred_dic_dev['prediction'] = predvals_dev
pred_dic_test['prediction'] = predvals_test

pred_dic_tra['label'] = truevals_tra
pred_dic_dev['label'] = truevals_dev
pred_dic_test['label'] = truevals_test

with open(result_save_path + '_train.pickle', 'wb') as f:
   pickle.dump(pred_dic_tra, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()

with open(result_save_path + '_dev.pickle', 'wb') as f:
   pickle.dump(pred_dic_dev, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()

with open(result_save_path + '_test.pickle', 'wb') as f:
   pickle.dump(pred_dic_test, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()



