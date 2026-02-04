import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, Reshape

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
import os

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_dir = r"MyDrive/musicclassification/gtzan/genres_original"
classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.activations import sigmoid
from tensorflow.image import resize


def load_and_preprocess_data(data_dir, classes, target_shape=(300, 150)):
    data = []
    labels = []
    for i_class, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        print("Processing--", class_name)
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.wav'):
                file_path = os.path.join(class_dir, file_name)
                try:
                    # Try to load the audio file
                    audio_data, sample_rate = librosa.load(file_path, sr=None)

                    # Performing preprocessing
                    chunk_duration = 4
                    overlap_duration = 2
                    chunk_samples = chunk_duration * sample_rate
                    overlap_samples = overlap_duration * sample_rate
                    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

                    # Iterate over each chunk
                    for i in range(num_chunks):
                        start = i * (chunk_samples - overlap_samples)
                        end = start + chunk_samples
                        chunk = audio_data[start:end]
                        mel_spectrogram = np.abs(librosa.stft(y=chunk))

                        # Resize matrix to the target shape
                        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
                        data.append(mel_spectrogram)
                        labels.append(i_class)
                except Exception as e:
                    # Handle the exception and skip the corrupted file
                    print(f"Error processing file {file_path}: {e}")

    return np.array(data), np.array(labels)


data, labels = load_and_preprocess_data(data_dir, classes)

from tensorflow.keras.utils import to_categorical

if labels.ndim == 1:  # 检查标签是否为一维（未进行独热编码）
    labels = to_categorical(labels, num_classes=len(classes))

# 打乱数据
np.random.seed(42)
np.random.shuffle(data)
np.random.seed(42)
np.random.shuffle(labels)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
# 然后从训练集中划分出验证集
# 随机输出一张图片的shape
print(X_train[0].shape)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, Reshape
from tensorflow.keras.activations import sigmoid

import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, Reshape, Activation, multiply
from tensorflow.keras.layers import (Input, Conv2D, MaxPool2D, GlobalAveragePooling2D,
                                     Reshape, Activation, Dropout, Flatten, Dense,
                                     Multiply, Concatenate, Lambda)
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Conv2D, Lambda, Concatenate, Multiply, Add, \
    Activation
import tensorflow as tf


def eca_layer(inputs, reduction=16):
    """
    改进的ECA注意力模块，加入门控机制实现动态通道权重调整
    参数:
        inputs: 输入特征图 (B, H, W, C)
        reduction: 通道缩减比例
    返回:
        经过门控动态注意力加权的特征图
    """
    # 1. 原始ECA通道注意力分支
    channel_avg = GlobalAveragePooling2D()(inputs)
    channel_avg = Reshape((1, 1, channel_avg.shape[-1]))(channel_avg)

    # 使用两层MLP生成基础通道注意力
    channel_conv = Conv2D(filters=channel_avg.shape[-1] // reduction,
                          kernel_size=1, padding="same", activation="relu")(channel_avg)
    channel_conv = Conv2D(filters=channel_avg.shape[-1],
                          kernel_size=1, padding="same")(channel_conv)
    base_channel_att = Activation('sigmoid')(channel_conv)

    # 2. 门控机制分支
    # 使用全局标准差作为额外信息源
    channel_std = Lambda(lambda x: tf.math.reduce_std(x, axis=[1, 2], keepdims=True))(inputs)

    # 门控权重生成
    gate_input = Concatenate(axis=-1)([channel_avg, channel_std])  # 结合平均值和标准差
    gate_weights = Conv2D(filters=gate_input.shape[-1] // reduction,
                          kernel_size=1, padding="same", activation="relu")(gate_input)
    gate_weights = Conv2D(filters=1,
                          kernel_size=1, padding="same", activation="sigmoid")(gate_weights)

    # 3. 动态调整通道注意力
    # 门控控制基础注意力的增强程度 (范围在[0.5, 1.5]之间)
    adjusted_att = Lambda(lambda x: 0.5 + x[0] * x[1])([gate_weights, base_channel_att])

    # 4. 空间注意力分支
    spatial_avg = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(inputs)
    spatial_max = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(inputs)
    spatial_concat = Concatenate(axis=-1)([spatial_avg, spatial_max])
    spatial_att = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(spatial_concat)

    # 5. 结合通道和空间注意力
    combined_attention = Multiply()([adjusted_att, spatial_att])

    # 6. 残差连接增强
    output = Multiply()([inputs, combined_attention])
    output = Add()([inputs, output])  # 残差连接

    return output


class MyCNNModel(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(MyCNNModel, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        # 输入层
        inputs = tf.keras.Input(shape=self.input_shape)

        # 第一组卷积层
        x = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
        x = Conv2D(filters=32, kernel_size=3, activation='relu')(x)
        x = MaxPool2D(pool_size=2, strides=2)(x)
        x = eca_layer(x)

        # 第二组卷积层
        x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(filters=64, kernel_size=3, activation='relu')(x)
        x = MaxPool2D(pool_size=2, strides=2)(x)
        x = eca_layer(x)

        # 第三组卷积层
        x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(filters=128, kernel_size=3, activation='relu')(x)
        x = MaxPool2D(pool_size=2, strides=2)(x)
        x = eca_layer(x)

        # Dropout 层
        x = Dropout(0.1)(x)

        # 第四组卷积层
        x = Conv2D(filters=256, kernel_size=7, padding='same', activation='relu', dilation_rate=2)(x)
        x = Conv2D(filters=256, kernel_size=7, activation='relu', dilation_rate=2)(x)
        x = eca_layer(x)

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
        model = tf.keras.Model(inputs=inputs, outputs=output)

        # 模型汇总
        model.summary()
        return model

    def compile(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def summary(self):
        self.model.summary()

    def save(self, filepath):
        self.model.save(filepath)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)


# 使用该类
input_shape = (300, 150, 1)  # 例如 (150, 150, 1)，请根据实际输入形状修改
num_classes = 10

# 创建模型
my_cnn_model = MyCNNModel(input_shape, num_classes)

# 编译模型
my_cnn_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 打印模型架构
my_cnn_model.summary()

# 定义回调（保存最佳模型）
checkpoint = ModelCheckpoint('working/best_model.keras', save_best_only=True, monitor='val_accuracy',
                             mode='max')
training_history = my_cnn_model.model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100, batch_size=32,
                                          callbacks=[checkpoint])

# 加载最佳模型
my_cnn_model.load_weights('/kaggle/working/best_model.keras')

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# 在测试集上评估模型
loss, accuracy = my_cnn_model.model.evaluate(X_test, Y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# 绘制混淆矩阵
Y_pred = my_cnn_model.model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true_classes = np.argmax(Y_test, axis=1)

# 假设你有一个类标签列表，例如：
classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
cm = confusion_matrix(Y_true_classes, Y_pred_classes)

# 绘制混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 绘制损失函数和准确率曲线
# 从训练过程返回的 history 对象
history = training_history  # 假设你已经有了这个对象，来自 `model.fit`

# 绘制损失函数曲线
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.show()
