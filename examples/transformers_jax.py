import functools
import os
import time

from jax.experimental import jax2tf

import jax.numpy as jnp
from flax import nnx
import numpy as np
import optax
import tensorflow as tf


class SimpleFC(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, hidden_dim: int = 128):
        self.fc1 = nnx.Linear(2, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.fc3 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.fc4 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.fc5 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.fc6 = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))
        x = nnx.relu(self.fc3(x))
        x = nnx.relu(self.fc4(x))
        x = nnx.relu(self.fc5(x))
        x = self.fc6(x)
        return x


@nnx.jit
def train_step(model, optimizer, x, y):
    def loss_fn(m):
        preds = m(x)
        return jnp.mean((preds - y) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def representative_dataset(data):
    for i in range(min(1000, data.shape[0])):
        yield [data[i : i + 1]]


if __name__ == "__main__":
    os.makedirs("generated_files", exist_ok=True)
    tflite_model_path = os.path.join("generated_files", "simple_fc_jax.tflite")
    tflite_quant_model_path = os.path.join(
        "generated_files", "simple_fc_jax_quant.tflite"
    )

    np.random.seed(0)
    model = SimpleFC(rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    num_epochs = 1000
    batch_size = 64

    print("\nStarting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        inputs = np.random.randn(batch_size, 2).astype(np.float32)
        targets = (inputs[:, 0] + inputs[:, 1] * 2).reshape(-1, 1).astype(np.float32)

        x = jnp.array(inputs)
        y = jnp.array(targets)
        loss = train_step(model, optimizer, x, y)

        if (epoch + 1) % 100 == 0:
            epoch_time = time.time() - start_time
            print(
                f"Epoch [{epoch+1}/{num_epochs}] in {epoch_time:.2f}s, Loss: {loss:.6f}"
            )
    
    @tf.function(autograph=False, input_signature=[tf.TensorSpec(shape=(1, 2), dtype=tf.float32)])
    def predict(x):
        return model(x)

    m = tf.Module()
    # Wrap the JAX state in `tf.Variable` (needed when calling the converted JAX function.
    params = nnx.state(model, nnx.Param)
    print(params)
    state_vars = tf.nest.map_structure(tf.Variable, params)
    # Keep the wrapped state as flat list (needed in TensorFlow fine-tuning).
    m.vars = tf.nest.flatten(state_vars)
    # Convert the desired JAX function (`model.predict`).
    predict_fn = jax2tf.convert(model.predict)
    # Wrap the converted function in `tf.function` with the correct `tf.TensorSpec` (necessary for dynamic shapes to work).
    m.predict = predict
    converter = tf.lite.TFLiteConverter.from_concrete_functions(predict, m)
    tflite_model = converter.convert()
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = functools.partial(
        representative_dataset, jnp.array(inputs)
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    tflite_quant_model = converter.convert()
    with open(tflite_quant_model_path, "wb") as f:
        f.write(tflite_quant_model)

    print("Exported TFLite models to generated_files.")

    expected = serving_func(x_input)

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], np.array(x_input))
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]["index"])
    np.testing.assert_allclose(expected, result, rtol=1e-4, atol=1e-4)

    interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], np.array(x_input))
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]["index"])
    np.testing.assert_allclose(expected, result, rtol=1e-2, atol=1e-2)
