import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
import numpy as np
import tensorflow as tf

import functools

def representative_dataset(data):
    for i in range(min(1000, data.shape[0])):
        yield [data[i : i + 1]]

os.makedirs("generated_files", exist_ok=True)
tflite_model_path = os.path.join("generated_files", "simple_fc_keras.tflite")
tflite_quant_model_path = os.path.join(
    "generated_files", "simple_fc_keras_quant.tflite"
)

model = keras.Sequential(
    [
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(1),
    ]
)

print(model.summary())

inputs = np.random.randn(10000, 2).astype(np.float32)
targets = (inputs[:, 0] + inputs[:, 1] * 2).reshape(-1, 1).astype(np.float32)

model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
)

model.fit(
    inputs,
    targets,
    batch_size=64,
    epochs=3,
    validation_split=0.15,
)

model_trained = model

model = keras.Sequential(
    [
        keras.layers.Input(batch_shape=(1, 2)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(1),
    ]
)

model.set_weights(model_trained.get_weights())

print(model.summary())
x_input = np.array([[-1.0, -1.0]], dtype=np.float32)
print(model.predict(x_input))

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

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
interpreter.set_tensor(input_details[0]["index"], x_input)
interpreter.invoke()
result = interpreter.get_tensor(output_details[0]["index"])
print(result)

interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]["index"], x_input)
interpreter.invoke()
result = interpreter.get_tensor(output_details[0]["index"])
print(result)


