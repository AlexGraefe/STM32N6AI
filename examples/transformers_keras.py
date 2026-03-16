import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
import keras_hub
import numpy as np
import tensorflow as tf

import functools
import subprocess

def representative_dataset(data):
    for i in range(min(10, data.shape[0])):
        yield [data[i : i + 1]]

os.makedirs("generated_files", exist_ok=True)
tflite_model_path = os.path.join("generated_files", "transformer_keras.tflite")
tflite_quant_model_path = os.path.join(
    "generated_files", "transformer_keras_quant.tflite"
)

model = keras.Sequential(
    [
        keras.layers.Input(batch_shape=(1, 128, 128)),
        keras_hub.layers.TransformerEncoder(
            intermediate_dim=128,
            num_heads=4
        ),
    ]
)

inputs = np.random.randn(10, 128, 128).astype(np.float32)

converter = tf.lite.TFLiteConverter.from_keras_model(model) 
tflite_model = converter.convert()


with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = functools.partial(
    representative_dataset, np.array(inputs)
)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_quant_model = converter.convert()
with open(tflite_quant_model_path, "wb") as f:
    f.write(tflite_quant_model)

print("Exported TFLite models to generated_files.")

subprocess.run(
    ["./generate_nn_code.sh", "generated_files/transformer_keras_quant.tflite"],
    check=True,
)


