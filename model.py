import tensorflow as tf

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define the CNN model
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(6, activation='softmax'))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', metrics=['accuracy'])

# Data preprocessing using ImageDataGenerator
train_datagen = ImageDataGenerator(validation_split=0.2,
                                   rotation_range=10,
   width_shift_range=0.1,
   height_shift_range=0.1,
   shear_range=0.1,
   zoom_range=0.1,
   channel_shift_range=0.1,
   fill_mode='nearest',
   cval=0.1,
   horizontal_flip=True,
   vertical_flip=True,)
# test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'class_data/labelled data',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset = 'training')

validation_generator = train_datagen.flow_from_directory(
    'class_data/labelled data',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset = 'validation')

# validation_generator = test_datagen.flow_from_directory(
#     'path/to/validation_data',
#     target_size=(150, 150),
#     batch_size=32,
#     class_mode='multi'
# )

# Train the model
history = model.fit(
    train_generator,
    # steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=100,
    validation_data=validation_generator,
    # validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Evaluate the model
test_loss, test_acc = model.evaluate(validation_generator)
print(f'Test accuracy: {test_acc}')

# Plot training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('training accuracy.jpg')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('training loss.jpg')

plt.show()
