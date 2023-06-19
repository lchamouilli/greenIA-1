import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping


# Set mixed precision policy
tf.keras.mixed_precision.set_global_policy('mixed_float16')
# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Normalize pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0
# Define model
model = Sequential([
  Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
  BatchNormalization(),
  MaxPooling2D((2, 2)),
  Conv2D(64, (3, 3), activation='relu'),
  BatchNormalization(),
  MaxPooling2D((2, 2)),
  Conv2D(64, (3, 3), activation='relu'),
  BatchNormalization(),
  Flatten(),
  Dense(64, activation='relu'),
  BatchNormalization(),
  Dense(10)
])

# Define early stopping callback
early_stopping = EarlyStopping(
    monitor='val_accuracy',   # Monitor validation accuracy
    mode='max',               # Maximize the monitored quantity
    patience=5              # Number of epochs with no improvement before stopping
)


# Compile and train model
model.compile(optimizer='adam',
       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
       metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, callbacks=[early_stopping])
