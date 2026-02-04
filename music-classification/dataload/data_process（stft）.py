
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
                        mel_spectrogram =np.abs(librosa.stft(y=chunk))
                        # 生成梅尔频谱(核心替换)
                        # mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
                        # # 梅尔频谱转对数刻度(音频预处理必做，提升模型效果)，非常关键
                        # mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
                        # Resize matrix to the target shape
                        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
                        data.append(mel_spectrogram)
                        labels.append(i_class)
                except Exception as e:
                    # Handle the exception and skip the corrupted file
                    print(f"Error processing file {file_path}: {e}")

    return np.array(data), np.array(labels)
data,labels=load_and_preprocess_data(data_dir,classes)







