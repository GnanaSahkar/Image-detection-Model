import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from build_model import create_model

# Load data
data = np.load('../data.npz')
images = data['images']
labels = data['labels']
class_names = data['class_names']

# Define model
input_shape = (224, 224, 3)
num_classes = len(class_names)
model = create_model(input_shape, num_classes)

# Define the file path for saving the best model with .keras extension
model_checkpoint_path = '../models/best_model.keras'

# Model checkpoint callback
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(model_checkpoint_path, monitor='val_loss', save_best_only=True)
]

# Training
history = model.fit(images, labels, validation_split=0.2, epochs=100, batch_size=32, callbacks=callbacks)
