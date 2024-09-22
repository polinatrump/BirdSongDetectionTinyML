from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch

class BinaryCNNHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = tf.keras.models.Sequential()
        
        # Hyperparameters
        filters1 = hp.Int('filters1', min_value=2, max_value=8, step=2)
        filters2 = hp.Int('filters2', min_value=2, max_value=16, step=2)
        kernel_size = hp.Choice('kernel_size', values=[2, 3, 5])
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        # batch_size = hp.Int('batch_size', min_value=16, max_value=64, step=16)
        
        # Model architecture
        model.add(lq.layers.QuantConv2D(
            filters=filters1, kernel_size=(kernel_size, kernel_size),
            input_shape=self.input_shape,
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            use_bias=False
        ))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.BatchNormalization(scale=False))
        model.add(lq.layers.QuantConv2D(
            filters=filters2, kernel_size=(kernel_size, kernel_size),
            input_quantizer="ste_sign",
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            use_bias=False
        ))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.BatchNormalization(scale=False))
        model.add(tf.keras.layers.Flatten())
        model.add(lq.layers.QuantDense(
            units=64,
            kernel_quantizer="ste_sign",
            input_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            use_bias=False
        ))
        model.add(tf.keras.layers.BatchNormalization(scale=False))
        model.add(lq.layers.QuantDense(
            units=self.num_classes,
            kernel_quantizer="ste_sign",
            input_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            use_bias=False
        ))
        model.add(tf.keras.layers.BatchNormalization(scale=False))
        model.add(tf.keras.layers.Activation("softmax"))
        
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )
        return model

input_shape = (184, 80, 1)
num_classes = 2

tuner = RandomSearch(
    BinaryCNNHyperModel(input_shape, num_classes),
    objective='val_accuracy',
    max_trials=2,
    executions_per_trial=2,
    # directory='my_dir',
    project_name='binary_cnn_tuning'
)

tuner.search(train_spectrogram_ds, epochs=1, validation_data=val_spectrogram_ds)
best_model = tuner.get_best_models(num_models=1)[0]



class CNNHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = tf.keras.Sequential()
        
        # Hyperparameters
        # num_conv_layers = hp.Int('num_conv_layers', min_value=1, max_value=3, step=1)
        # print("num_conv_layers", num_conv_layers)
        # filters = [hp.Int(f'filters_{i}', min_value=2, max_value=8, step=2) for i in range(num_conv_layers)]
        # print("filters ", filters)
        kernel_size = hp.Choice('kernel_size', values=[2, 3, 5])
        dense_units = hp.Int('1st_dense_units', min_value=4, max_value=32, step=4)
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        dense_activation = hp.Choice('2nd_dense_activation', values=['softmax', 'linear'])
        
        # Model architecture
        model.add(tf.keras.layers.Input(shape=(184, 80, 1)))

        # Tune number of layers
        for i in range(hp.Int('num_conv_layers', min_value=1, max_value=3, step=1)):
            model.add(tf.keras.layers.Conv2D(
                filters=hp.Int(f'filters_{i}', min_value=2, max_value=8, step=2), kernel_size=(kernel_size, kernel_size),
                activation=hp.Choice("activation", ["relu", "tanh"])
            ))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(
            units=dense_units,
            activation='relu'
        ))

        model.add(lq.layers.QuantDense(
            units=self.num_classes,
            activation=dense_activation
        ))
       
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )
        return model

input_shape = (184, 80, 1)
num_classes = 2

tuner = RandomSearch(
    CNNHyperModel(input_shape, num_classes),
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='hpo_tuner/cnn',
    project_name='cnn_tuning'
)