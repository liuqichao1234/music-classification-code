import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPool2D, GlobalAveragePooling2D,
                                     Reshape, Activation, Dropout, Flatten, Dense,
                                     Multiply, Concatenate, Lambda, LayerNormalization,
                                     MultiHeadAttention, Add, UpSampling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Resizing


def attention_layer(inputs, reduction=16):
    """attention注意力模块"""
    channel_avg = GlobalAveragePooling2D()(inputs)
    channel_avg = Reshape((1, 1, channel_avg.shape[-1]))(channel_avg)
    channel_conv = Conv2D(filters=channel_avg.shape[-1] // reduction,
                          kernel_size=1, padding="same", activation="relu")(channel_avg)
    channel_conv = Conv2D(filters=channel_avg.shape[-1],
                          kernel_size=1, padding="same")(channel_conv)
    channel_attention = Activation('sigmoid')(channel_conv)

    spatial_avg = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(inputs)
    spatial_max = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(inputs)
    spatial_concat = Concatenate(axis=-1)([spatial_avg, spatial_max])
    spatial_conv = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(spatial_concat)

    combined_attention = Multiply()([channel_attention, spatial_conv])
    output = Multiply()([inputs, combined_attention])
    return output


class MyCNNTransformerModel(Model):
    def __init__(self, input_shape, num_classes):
        super(MyCNNTransformerModel, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def transformer_encoder(self, inputs, num_heads=4, ff_dim=128, dropout=0.1):
        """构建Transformer编码器块"""
        # 自注意力机制
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=inputs.shape[-1] // num_heads)(inputs, inputs)
        attention_output = Dropout(dropout)(attention_output)
        out1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)

        # 前馈网络
        ffn_output = Dense(ff_dim, activation="relu")(out1)
        ffn_output = Dense(inputs.shape[-1])(ffn_output)
        ffn_output = Dropout(dropout)(ffn_output)
        return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    def dynamic_feature_fusion(self, cnn_feat, trans_feat):
        """简化版门控机制(单个权重)"""
        # 确保形状相同
        if cnn_feat.shape[1] != trans_feat.shape[1] or cnn_feat.shape[2] != trans_feat.shape[2]:
            trans_feat = Resizing(height=cnn_feat.shape[1],
                                  width=cnn_feat.shape[2],
                                  interpolation='bilinear')(trans_feat)

        # 计算门控权重
        concat_features = Concatenate(axis=-1)([cnn_feat, trans_feat])
        gap = GlobalAveragePooling2D()(concat_features)

        # 单个标量权重
        gate_weight = Dense(1, activation='sigmoid')(gap)
        gate_weight = Reshape((1, 1, 1))(gate_weight)

        # 融合
        fused_features = Add()([
            Multiply()([cnn_feat, gate_weight]),
            Multiply()([trans_feat, 1 - gate_weight])
        ])

        return fused_features

    def build_model(self):
        # 输入层
        inputs = Input(shape=self.input_shape)

        # 初始卷积层 (两条支路共享)
        x = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)

        # ============== 主CNN支路 ==============
        cnn_branch = Conv2D(filters=32, kernel_size=3, activation='relu')(x)
        cnn_branch = MaxPool2D(pool_size=2, strides=2)(cnn_branch)
        cnn_branch = attention_layer(cnn_branch)

        # 第二组卷积层
        cnn_branch = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(cnn_branch)
        cnn_branch = Conv2D(filters=64, kernel_size=3, activation='relu')(cnn_branch)
        cnn_branch = MaxPool2D(pool_size=2, strides=2)(cnn_branch)
        cnn_branch = attention_layer(cnn_branch)

        # ============== Transformer支路 ==============
        # 空间下采样
        trans_branch = Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(x)
        trans_branch = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(trans_branch)

        # 转换为序列格式 (H*W, C)
        h, w, c = trans_branch.shape[1:]
        trans_branch = Reshape((h * w, c))(trans_branch)

        # Transformer编码器
        trans_branch = self.transformer_encoder(trans_branch)
        trans_branch = self.transformer_encoder(trans_branch)

        # 转换回空间格式
        trans_branch = Reshape((h, w, c))(trans_branch)

        # ============== 特征融合 ==============
        # 空间对齐 (如果需要)
        if cnn_branch.shape[1] != trans_branch.shape[1]:
            trans_branch = trans_branch = Resizing(
                height=cnn_branch.shape[1],
                width=cnn_branch.shape[2],
                interpolation='bilinear'
            )(trans_branch)

        # 动态特征融合
        fused_features = self.dynamic_feature_fusion(cnn_branch, trans_branch)

        # ============== 继续主网络 ==============
        x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(fused_features)
        x = Conv2D(filters=128, kernel_size=3, activation='relu')(x)
        x = MaxPool2D(pool_size=2, strides=2)(x)
        x = attention_layer(x)

        # Dropout 层
        x = Dropout(0.1)(x)

        # 第四组卷积层
        x = Conv2D(filters=256, kernel_size=7, padding='same', activation='relu', dilation_rate=2)(x)
        x = Conv2D(filters=256, kernel_size=7, activation='relu', dilation_rate=2)(x)
        x = attention_layer(x)

        # Dropout 层
        x = Dropout(0.3)(x)

        # Flatten 层
        x = Flatten()(x)

        # Fully connected layers
        x = Dense(units=1200, activation='relu')(x)
        x = Dropout(0.45)(x)

        # 输出层
        output = Dense(units=self.num_classes, activation='softmax')(x)

        # 构建模型
        model = Model(inputs=inputs, outputs=output)
        return model

    def compile(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_data=None):
        return self.model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs,
                              verbose=verbose, callbacks=callbacks, validation_data=validation_data)

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1):
        return self.model.evaluate(x=x, y=y, batch_size=batch_size, verbose=verbose)

    def predict(self, x, batch_size=None, verbose=0):
        return self.model.predict(x, batch_size=batch_size, verbose=verbose)

    def summary(self):
        return self.model.summary()

    def save(self, filepath):
        self.model.save(filepath)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)