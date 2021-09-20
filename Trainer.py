import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

# First we import our dataset as training and validation
training_set = tf.keras.preprocessing.image_dataset_from_directory(
    "Spectrograms", labels="inferred", label_mode="categorical", subset="training", validation_split=0.2, seed=1
) 
validation_set = tf.keras.preprocessing.image_dataset_from_directory(
    "Spectrograms", labels="inferred", label_mode="categorical", subset="validation", validation_split=0.2, seed=1
) 

# Following tf's recommendation to cache images so it doesn't have to continuously open/close
AUTOTUNE = tf.data.AUTOTUNE
training_set = training_set.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_set = validation_set.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)


WORDS = 2

# Let's use VGG-11
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape = (256,256,3)),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(256, 3, padding="same", activation="relu"),
    layers.Conv2D(256, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(512, 3, padding="same", activation="relu"),
    layers.Conv2D(512, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(512, 3, padding="same", activation="relu"),
    layers.Conv2D(512, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(1024, activation="relu"),
    layers.Dense(1024, activation="relu"),
    layers.Dense(WORDS)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

history = model.fit(
  training_set,
  validation_data=validation_set,
  epochs=15
)