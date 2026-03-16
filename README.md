# Installation
Installation takes a while, as you need to install all tools from ST and then the VSCode extension. This website explains you everything you need to install for the VSCode Extension

https://learn.arm.com/install-guides/stm32_vs/

Details (in case the website is gone). Install the following tools from ST:
- [STM32CubeCLT](https://www.st.com/en/development-tools/stm32cubeclt.html)
- [STM32CubeMX](https://www.st.com/en/development-tools/stm32cubemx.html) (not really needed just to avoid errors)
- [ST-MCU-FINDER-PC](https://www.st.com/en/development-tools/st-mcu-finder-pc.html)
- [ST-CORE-AI](https://www.st.com/en/development-tools/stedgeai-core.html#st-get-software) Download it and run ./stedgeai-linux-onlineinstaller. The UI will guide you through the installation process. 

Add the following directory to your PATH variable (e.g., in ~/.bashrc):
- /usr/local/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin
- /opt/ST/STEdgeAI/<Version, e.g., 3.0>/Utilities/linux


Then, install the VSCode extension. 
1. Open the `Extensions` view in Visual Studio Code
2. Search for `STM32 VS Code Extension`
3. Click the `Install` button

Open the project in VSCode. A dialog will pop up in the right lower corner of the window asking to configure the project as an STM32 project. Click yes. The windowd will change with a button called configure. Press enter.
Then, in the same window, type "STM32N6570-DK" under Board / Device and GCC under Toolchain. Add three projects. One called Appli, one Apli2 and one called FSBL. Select secure under security model. Now you should have configured the project.

Generate a virtual environment with Python 3.12:
1. Init venv:
    ```bash
    python3.12 -m venv ./venv
    ```
2. Activate the virtual environment
    ```bash
    source ./venv/bin/activate
    ```
3. Install torch: https://pytorch.org/get-started/locally/
4. Install the requirements
    ```bash
    pip install -r requirements.txt
    ```

# Run a simple example:

1. Activate the virtual environment
    ```bash
    source ./venv/bin/activate
    ```
2. Generate the .onnx file of a simple NN
    ```bash
    python -m examples.simple_fc
    ```
3. Generate the nn files
    ```bash
    ./generate_nn_code.sh generated_files/simple_fc_quant.onnx
    ```
4. Press the Build button on the bottom left part of your VSCode window. This will compile the project and also automatically sign the binary. For a clean rebuild, type ctrl + shift + P and search for CMake: Claen Rebuild.
5. Program the STM32N6. Set Boot1 to 1 and reset
    ```bash
    ./program.sh
    ```
    Set Boot 1 to 0 and reset.
6. Open a serial terminal and connect to the STM32N6 (baudrate 115200). You should see some prints.

# Explanation

## Memory layout
We place the Application into Flash and the FSBL executes it from Flash, i.e., we do not copy the application into RAM, as we need the space. The application itself only uses AXISRAM_1 and leaves place on the other RAMs for the NN.

The NN's memory layout is defined in [examples/my_mpools/stm32n6-app2_STM32N6570-DK.mpool](./examples/my_mpools/stm32n6-app2_STM32N6570-DK.mpool), which states that the activations are saved in the RAM starting with AXISRAM_2 and the weights are saved onto the flash starting with address 0x70380000.

## Neural Network Code Generation
We generate the code for the NN using STEdgeAI. Our example script [examples/simple_fc.py](./examples/simple_fc.py) transforms a PyTorch model into a quantized ONNX graph and saves it. Using STEdgeAi, we then transform this ONNX graph into NN weights and code needed for the inference on the STM32N6 ([generate_nn_code.sh](./generate_nn_code.sh)). The script places all generated files in ./generated_files and copies the coresponding C-files to the folder that the applicaiton links to.

When flashing the STM32N6, in [program.sh](./program.sh), we flash the generated weights (called network_atonbuf.xSPI2.raw) to the address 0x70380000, which is equal to the address we defined above to be the starting address of the NN's weights.

On how to call the NN, refer to [Appli/Core/Src/main.c](./Appli/Core/Src/main.c). The most important part is to invalidate the Cache as otherwise updates on the NN's inputs and outputs might be dangeling inside the cashe and not updated in the RAM (CPU and NPU do share RAM, but not caches).

# How to use peripherals.
Use CubeMX to generate the code to initialize the peripherals and then copy it. In [`./Core/Inc/stm32n6xx_hal_conf.h`](./Core/Inc/stm32n6xx_hal_conf.h) uncomment the line #define HAL_<peripherl>_MODULE_ENABLED
