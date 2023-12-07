# custom layer 
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model


class CustomResNet50Model:
    def __init__(self, num_classes):
        self.num_classes=num_classes
        self.base_model = tf.keras.applications.ResNet50(weights='imagenet',include_Top=False)
        self.model = self._create_model()
    
    def _create_model(self):
        #custom layer
        x=self.base_model.output
        x=Flatten()(x)
        x=Dense(1024, activation='relu')(x)
        x=Dropout(0.5)(x)
        x=Dense(512, activation='relu')(x)
        x=Dropout(0.5)(x)
        output_layer = Dense(self.num_classes, activation='softmax')(x)

        model= Model(input=self.base_model.input, outputs=output_layer)

        for layer in self.base_model.layers:
            layer.trainable = False
        return model
     
    def get_base_model_output(self, input_data):
        base_model_output = Model(input=self.base_model.input, outputs=self.base_model.output)
        return base_model_output.predict(input_data)
    
    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimimer, loss=loss, metrics=metrics)
    
    def summary(self):
        return self.model.summary()
    
