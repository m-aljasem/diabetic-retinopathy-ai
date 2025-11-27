"""InceptionV3-based model for retinopathy detection"""
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam


def build_retinopathy_model(input_shape=(299, 299, 3), num_classes=5):
    """Build InceptionV3 model for diabetic retinopathy classification."""
    base = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False
    
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base.input, outputs=x)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

