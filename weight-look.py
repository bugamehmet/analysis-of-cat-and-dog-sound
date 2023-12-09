from keras.models import load_model
import scipy.io.wavfile as sci_wav
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.optimizers import Adam

# Modeli yükleyin (modelin mimarisini ve ağırlıklarını içerir)
model = load_model('model_with_weights.keras')

adam_optimizer = Adam(decay=None)
model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])

# Model özeti
model.summary()

# Modelin ağırlıklarını alın
weights = model.get_weights()

ROOT_DIR = "cats_dogs/"
CSV_PATH = "train_test_split.csv"

def read_wav_files(wav_files):
    if not isinstance(wav_files, list):
        wav_files = [wav_files]
    return [sci_wav.read(ROOT_DIR + f)[1] for f in wav_files]

def load_test_dataset(dataframe):
    df = dataframe
    test_dataset = {}
    for k in ['test_cat', 'test_dog']:
        v = list(df[k].dropna())
        v = read_wav_files(v)
        v = np.concatenate(v).astype('float32')
        test_dataset[k] = v

        print('load {} with {} sec of audio'.format(k, len(v) / 16000))
    return test_dataset




df = pd.read_csv(CSV_PATH)
test_dataset = load_test_dataset(df)


test_data_cat = test_dataset['test_cat'].reshape(-1, 6109640, 1)
test_data_dog = test_dataset['test_dog'].reshape(-1, 4499160, 1)


test_labels_cat = np.zeros(len(test_data_cat))
test_labels_dog = np.ones(len(test_data_dog))

# Modeli test veri seti üzerinde değerlendirin
loss_cat, accuracy_cat = model.evaluate(test_data_cat, test_labels_cat)
loss_dog, accuracy_dog = model.evaluate(test_data_dog, test_labels_dog)

# Toplam test loss ve accuracy'yi hesaplayın
total_loss = (loss_cat * len(test_data_cat) + loss_dog * len(test_data_dog)) / (len(test_data_cat) + len(test_data_dog))
total_accuracy = (accuracy_cat * len(test_data_cat) + accuracy_dog * len(test_data_dog)) / (len(test_data_cat) + len(test_data_dog))

print(f'Total Test Loss: {total_loss}')
print(f'Total Test Accuracy: {total_accuracy}')

# Her bir sınıf için ayrı ayrı sonuçları görüntüleyin
print(f'Test Cat Loss: {loss_cat}, Accuracy: {accuracy_cat}')
print(f'Test Dog Loss: {loss_dog}, Accuracy: {accuracy_dog}')

# Grafiği çizdirin
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

# Loss grafiği
ax1.plot([total_loss], label='Total Test Loss', marker='o')
ax1.plot([loss_cat, loss_dog], label=['Test Cat Loss', 'Test Dog Loss'], marker='o')
ax1.legend()
ax1.set_title('Test Loss')

# Accuracy grafiği
ax2.plot([total_accuracy], label='Total Test Accuracy', marker='o')
ax2.plot([accuracy_cat, accuracy_dog], label=['Test Cat Accuracy', 'Test Dog Accuracy'], marker='o')
ax2.legend()
ax2.set_title('Test Accuracy')

plt.show()
