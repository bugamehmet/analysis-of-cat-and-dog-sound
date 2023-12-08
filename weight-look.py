from keras.models import load_model

# Modeli yükleyin (modelin mimarisini ve ağırlıklarını içerir)
model = load_model('model_with_weights.h5')

# Model özeti
model.summary()

# Modelin ağırlıklarını alın
weights = model.get_weights()

#agirliklar
print(weights)