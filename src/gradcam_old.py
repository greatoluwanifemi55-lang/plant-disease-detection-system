import tensorflow as tf
import numpy as np
import cv2

from keras.utils import load_img, img_to_array


class GradCAM:

    def __init__(self, model, last_conv_layer):

        self.model = model
        self.last_conv_layer = last_conv_layer

    def make_gradcam_heatmap(self, image):

        # Build a model that returns BOTH:
        # 1. Feature maps
        # 2. Predictions

        grad_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[
                self.model.get_layer(self.last_conv_layer).output,
                self.model.output
            ]
        )

        with tf.GradientTape() as tape:

            conv_outputs, predictions = grad_model(image)

            predicted_class = tf.argmax(predictions[0])

            loss = predictions[:, predicted_class]

        grads = tape.gradient(loss, conv_outputs)

        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0, 1, 2)
        )

        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0)

        heatmap /= tf.reduce_max(heatmap)

        return heatmap.numpy()

    def overlay_heatmap(
        self,
        image_path,
        heatmap,
        output_path
    ):

        image = cv2.imread(str(image_path))

        image = cv2.resize(
            image,
            (224, 224)
        )

        heatmap = cv2.resize(
            heatmap,
            (224, 224)
        )

        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        superimposed = cv2.addWeighted(
            image,
            0.6,
            heatmap,
            0.4,
            0
        )

        cv2.imwrite(
            str(output_path),
            superimposed
        )