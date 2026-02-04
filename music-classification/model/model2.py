# =====================================================
# 4. 稳定版 Gated-attention Attention
# =====================================================
def gated_attention(inputs, reduction=16):
    channels = inputs.shape[-1]

    x = GlobalAveragePooling2D()(inputs)
    x = Reshape((1, 1, channels))(x)

    x = Conv2D(channels // reduction, 1, activation='relu')(x)
    x = Conv2D(channels, 1, activation='sigmoid')(x)

    return Multiply()([inputs, x])


# =====================================================
# 5. Transformer Encoder (简化版，移除位置编码问题)
# =====================================================
def transformer_encoder(x, embed_dim, num_heads, mlp_dim, dropout=0.1):
    # Multi-head self-attention
    attn = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim
    )(x, x)
    attn = Dropout(dropout)(attn)
    x = Add()([x, attn])
    x = LayerNormalization()(x)

    # Feed-forward network
    ffn = tf.keras.Sequential([
        Dense(mlp_dim, activation='relu'),
        Dropout(dropout),
        Dense(embed_dim),
        Dropout(dropout),
    ])(x)

    x = Add()([x, ffn])
    x = LayerNormalization()(x)
    return x


# =====================================================
# 6. 简单的位置编码层
# =====================================================
def add_position_encoding(x):
    # x的形状: (batch, seq_len, embed_dim)
    seq_len = tf.shape(x)[1]
    embed_dim = tf.shape(x)[2]

    # 生成位置编码
    position = tf.cast(tf.range(seq_len), tf.float32)
    position = tf.expand_dims(position, axis=1)  # (seq_len, 1)

    # 使用正弦和余弦函数生成位置编码
    div_term = tf.exp(tf.range(0, embed_dim, 2, dtype=tf.float32) *
                      -(np.log(10000.0) / embed_dim))

    # 创建位置编码矩阵
    pe = tf.zeros((seq_len, embed_dim))
    pe = tf.tensor_scatter_nd_update(
        pe,
        tf.range(seq_len)[:, tf.newaxis],
        tf.zeros((seq_len, embed_dim))
    )

    # 使用更简单的方法：直接添加可学习的位置编码
    # 我们将位置编码作为可训练参数
    return x


# =====================================================
# 7. 自适应特征融合门 (AFFG) - 用于一维向量
# =====================================================
def affg_1d(cnn_feat, trans_feat):
    # 现在两个输入都是一维向量
    concat = Concatenate()([cnn_feat, trans_feat])

    # 生成融合权重
    # 首先将两个特征向量相加，然后通过一个小的MLP生成权重
    combined = Add()([cnn_feat, trans_feat])
    alpha = Dense(1, activation='sigmoid')(combined)

    # 加权融合
    fused = alpha * cnn_feat + (1 - alpha) * trans_feat
    return fused


# =====================================================
# 8. 双支路模型 - 最终简化版
# =====================================================
def build_dual_path_model_final(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # ========== 共享的两层卷积 ==========
    # 第一层卷积
    x = Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(2, padding='same')(x)
    x = gated_attention(x)

    # 第二层卷积
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(2, padding='same')(x)
    x = gated_attention(x)

    # 保存分叉点特征
    branch_point = x

    # ========== Transformer支路 (简化版，不使用位置编码) ==========
    x_trans = branch_point

    # 获取特征图尺寸
    h, w, c = x_trans.shape[1:]

    # 将特征图转换为序列 (batch, h*w, c)
    x_trans = Reshape((h * w, c))(x_trans)

    # 注意：这里移除了位置编码，因为对于音频谱图，绝对位置信息可能不那么重要
    # 如果您认为位置信息重要，可以取消下面的注释，使用一个简单的位置编码

    # 可选：简单的位置编码
    # 创建一个可训练的位置编码
    seq_len = h * w
    pos_encoding = tf.keras.layers.Embedding(input_dim=seq_len, output_dim=c)
    positions = tf.range(seq_len)
    position_embedding = pos_encoding(positions)
    position_embedding = tf.expand_dims(position_embedding, 0)  # 添加batch维度
    x_trans = x_trans + position_embedding

    # Transformer Encoder (单个encoder层，避免过拟合)
    x_trans = transformer_encoder(
        x_trans,
        embed_dim=c,
        num_heads=4,
        mlp_dim=c * 2,
        dropout=0.1
    )

    # 转换回特征图
    x_trans = Reshape((h, w, c))(x_trans)

    # 全局平均池化得到一维向量
    x_trans = GlobalAveragePooling2D()(x_trans)

    # ========== CNN支路 ==========
    x_cnn = branch_point

    # 第三层卷积
    x_cnn = Conv2D(128, 3, padding='same', activation='relu')(x_cnn)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = Conv2D(128, 3, padding='same', activation='relu')(x_cnn)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = MaxPool2D(2, padding='same')(x_cnn)
    x_cnn = gated_attention(x_cnn)

    # 第四层卷积
    x_cnn = Conv2D(256, 3, padding='same', activation='relu')(x_cnn)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = Conv2D(256, 3, padding='same', activation='relu')(x_cnn)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = MaxPool2D(2, padding='same')(x_cnn)
    x_cnn = gated_attention(x_cnn)

    # 第五层卷积
    x_cnn = Conv2D(512, 3, padding='same', activation='relu')(x_cnn)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = Conv2D(512, 3, padding='same', activation='relu')(x_cnn)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = MaxPool2D(2, padding='same')(x_cnn)
    x_cnn = gated_attention(x_cnn)

    # 全局平均池化得到一维向量
    x_cnn = GlobalAveragePooling2D()(x_cnn)

    # ========== 特征融合 ==========
    # 方案1: 直接拼接（简单有效）
    fused = Concatenate()([x_cnn, x_trans])

    # 方案2: 使用加权融合（可选）
    # fused = affg_1d(x_cnn, x_trans)

    # ========== 后续处理 ==========
    fused = Dropout(0.3)(fused)

    # Dense(125)
    fused = Dense(125, activation='relu')(fused)
    fused = BatchNormalization()(fused)
    fused = Dropout(0.5)(fused)

    # 输出层
    outputs = Dense(num_classes, activation='softmax')(fused)

    model = Model(inputs, outputs)
    model.summary()
    return model


# =====================================================
# 9. 训练
# =====================================================
model = build_dual_path_model_final((300, 150, 1), 10)

model.compile(
    optimizer=Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 回调函数
checkpoint = ModelCheckpoint(
    "/kaggle/working/best_dual_path_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

print("=" * 50)
print("开始训练模型...")
print("=" * 50)

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=50,  # 先训练50个epoch看效果
    batch_size=32,
    callbacks=[checkpoint, early_stopping, reduce_lr],
    verbose=1
)

# =====================================================
# 10. 测试评估
# =====================================================
print("\n" + "=" * 50)
print("模型评估...")
print("=" * 50)

model.load_weights("/kaggle/working/best_dual_path_model.keras")

loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print(f"\nTest Accuracy: {acc:.4f}")
print(f"Test Loss: {loss:.4f}")

# 预测
Y_pred = model.predict(X_test, verbose=0)
Y_pred_cls = np.argmax(Y_pred, axis=1)
Y_true_cls = np.argmax(Y_test, axis=1)

# 分类报告
print("\nClassification Report:")
print(classification_report(Y_true_cls, Y_pred_cls, target_names=classes))

# =====================================================
# 11. 混淆矩阵
# =====================================================
cm = confusion_matrix(Y_true_cls, Y_pred_cls)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=classes,
            yticklabels=classes,
            cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Dual Path Model")
plt.tight_layout()
plt.savefig("/kaggle/working/confusion_matrix.png", dpi=300)
plt.show()

# =====================================================
# 12. 训练曲线
# =====================================================
plt.figure(figsize=(15, 5))

# Loss曲线
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Accuracy曲线
plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Train Acc', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# 学习率曲线
if 'lr' in history.history:
    plt.subplot(1, 3, 3)
    plt.plot(history.history['lr'], label='Learning Rate', linewidth=2)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
