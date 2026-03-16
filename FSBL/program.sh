#!/bin/bash
STM32_Programmer_CLI -c port=SWD  -d ./build/Debug/FSBL-trusted.bin 0x70000000 -el /usr/local/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/ExternalLoader/MX66UW1G45G_STM32N6570-DK.stldr 
if [ $? -ne 0 ]; then
    echo ""
    echo -e "\033[0;31mError: Failed to program FSBL. Please see error messages above.\033[0m" >&2
    exit 1
fi

echo ""
echo -e "\033[0;32mSuccessfully wrote FSBL into memory!\033[0m"
