from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Input, BatchNormalization, ReLU, GlobalAveragePooling2D
from tensorflow.keras import Model


def create_model(include_top=True):
    input = Input(shape=(150, 150, 3))

    # Block 1
    x = Conv2D(32, (3, 3), padding='same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)

    # Block 2
    x = Conv2D(64, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((4, 4))(x)

    # Block 3
    x = Conv2D(128, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((4, 4))(x)

    # Block 4
    x = Conv2D(256, (3, 3), padding='same', name='conv_block_4')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2), name='pooling_layer_4')(x)

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.25)(x)
        x = Dense(512, activation='relu', kernel_regularizer='l2')(x)  # Bottleneck layer
        x = Dropout(0.25)(x)
        outputs = Dense(38, activation='softmax')(x)
    else:
        outputs = x
    return Model(inputs=input, outputs=outputs)

def load_base_model():
    base_model = create_model(include_top=False)
    base_model.load_weights('base_model_weights.weights.h5')
    for layer in base_model.layers:
        if layer.name == 'conv_block_4':
            break
        layer.trainable = False
    return base_model

