#  收集文件路径（不加载数据）
# ==============================
def collect_file_paths(data_dir, classes):
    file_paths = []
    labels = []
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        for f in os.listdir(cls_dir):
            if f.endswith(".wav"):
                file_paths.append(os.path.join(cls_dir, f))
                labels.append(label)
    return file_paths, labels


file_paths, labels = collect_file_paths(data_dir, classes)
labels = to_categorical(labels, num_classes=num_classes)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    file_paths, labels, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42)

# ===============================
#  Generator：实时音频加载 + 预处理
# ===============================
def audio_generator(file_paths, labels):
    for file_path, label in zip(file_paths, labels):
        try:
            audio, sr = librosa.load(file_path, sr=None)

            chunk_duration = 4
            overlap_duration = 2
            chunk_samples = chunk_duration * sr
            overlap_samples = overlap_duration * sr

            num_chunks = int(np.ceil(
                (len(audio) - chunk_samples) /
                (chunk_samples - overlap_samples)
            )) + 1

            for i in range(num_chunks):
                start = i * (chunk_samples - overlap_samples)
                end = start + chunk_samples
                chunk = audio[start:end]

                if len(chunk) < chunk_samples:
                    continue

                spec = np.abs(librosa.stft(chunk))
                spec = np.expand_dims(spec, axis=-1)
                spec = resize(spec, target_shape)

                yield spec.numpy().astype(np.float32), label.astype(np.float32)

        except Exception as e:
            print(f"Skip corrupted file: {file_path}")

# ===============================
# tf.data.Dataset 构建
# ===============================
def build_dataset(file_paths, labels, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_generator(
        lambda: audio_generator(file_paths, labels),
        output_signature=(
            tf.TensorSpec(shape=(300, 150, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(num_classes,), dtype=tf.float32)
        )
    )
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
