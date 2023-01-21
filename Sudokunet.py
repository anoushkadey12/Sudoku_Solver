import tensorflow as tf
class SudokuNet:
    
    def build(width, height, depth, classes):
        model=tf.keras.models.Sequential()
        inputShape=(height, width, depth)

        model.add(tf.keras.layers.Conv2D(32,(5,5), padding="same",input_shape=inputShape))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

        model.add(tf.keras.layers.Conv2D(32,(3,3), padding="same"))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(64))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(classes))
        model.add(tf.keras.layers.Activation("softmax"))
        
        return model

