# scripts/build_model.py

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

def create_model(input_shape, num_classes):
    base_model = VGG16(include_top=False, input_shape=input_shape, weights='imagenet')
    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    input_shape = (224, 224, 3)
    num_classes = 150  # Update this based on the actual number of classes
    model = create_model(input_shape, num_classes)
    model.summary()
