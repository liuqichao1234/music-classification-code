def attention_layer(inputs, reduction=16):
    channel_avg = GlobalAveragePooling2D()(inputs)
    channel_avg = Reshape((1, 1, channel_avg.shape[-1]))(channel_avg)

    channel_conv = Conv2D(filters=channel_avg.shape[-1] // reduction,
                          kernel_size=1, padding="same", activation="relu")(channel_avg)
    channel_conv = Conv2D(filters=channel_avg.shape[-1],
                          kernel_size=1, padding="same")(channel_conv)
    base_channel_att = Activation('sigmoid')(channel_conv)

    channel_std = Lambda(lambda x: tf.math.reduce_std(x, axis=[1, 2], keepdims=True))(inputs)


    gate_input = Concatenate(axis=-1)([channel_avg, channel_std])
    gate_weights = Conv2D(filters=gate_input.shape[-1] // reduction,
                          kernel_size=1, padding="same", activation="relu")(gate_input)
    gate_weights = Conv2D(filters=1,
                          kernel_size=1, padding="same", activation="sigmoid")(gate_weights)

    adjusted_att = Lambda(lambda x: 0.5 + x[0] * x[1])([gate_weights, base_channel_att])

    spatial_avg = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(inputs)
    spatial_max = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(inputs)
    spatial_concat = Concatenate(axis=-1)([spatial_avg, spatial_max])
    spatial_att = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(spatial_concat)

    combined_attention = Multiply()([adjusted_att, spatial_att])

    output = Multiply()([inputs, combined_attention])
    output = Add()([inputs, output])

    return output
