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
    reshape_layer = layers.Reshape((1, rnn_dim*2), name="reshape_layer")
    concate_layer = layers.Concatenate(axis=1, name="attention_concate_layer")
    atten_dropout_layer = layers.Dropout(drop_out, name='attention_dropout_layer')
    map_layer1 = layers.Dense(model_dim, activation="relu", name="map_layer1")
    #map_layer2 = layers.TimeDistributed(layers.Dense(model_dim, activation="relu"), name="map_layer2")
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
    feature_map_layer = layers.Dense(model_dim, activation="relu",name="feature_map_layer")
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



voca_dim = embedding_matrix.shape[0]
pos_tag_size = len(pos_index)
time_steps = max_len

embed_dim = embedding_matrix.shape[1]
pos_tag_dim = 5
extra_feature_dims = num_pos_features
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