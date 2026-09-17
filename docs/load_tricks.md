# Loading and flashing notes

This is a collection of notes I made along the way on testing and flashing the STM32N6570-DK board.

Run commands from the repo root and adjust the ST tool paths for your installation.
See [the firmware guide](../firmware/README.md) for the missing SDK libraries.
The serial validation firmware and LCD app are separate builds.

## Just validation through stm_ai_runner:

generate c code stuff:
```shell
stedgeai generate -m .\training\quant_model\test-unet-tr_PerChannel_quant_oscd_44_npz_1.onnx --target stm32n6 --st-neural-art
```

load on device with
```shell
python C:\ST\STEdgeAI\4.0\scripts\N6_scripts\n6_loader.py --config-file "C:\ST\STEdgeAI\4.0\scripts\N6_scripts\config_n6l.json"
```

where you need to modify config.json to contain paths to CUBE IDE, and config_n6l.json to point to network.c output from steadgeai and also path to  `STEdgeAI/4.0/Projects/STM32N6570-DK/Applications/NPU_Validation`

Then you can run the validation on the device with run.py in `host`

---

## STM Cube app (harder option)

Using STM AI studio, load the quantized model and prepare application.

To debug directly you need to load signed Appli to specific address etc:

post-build inside rigth click on STM32N6...Appli, properties, C/C++ Build, settings, build steps, post-build command: (where the signign tool path migth be different for you)
```text
"C:\ST\STM32CubeIDE_2.2.0\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.win32_2.2.500.202603051304\tools\bin\STM32_SigningTool_CLI.exe" -s -bin "${BuildArtifactFileBaseName}.bin" -nk -t fsbl -hv 2.3 -align -o "${BuildArtifactFileBaseName}-Trusted.bin"
```

Then add binary to offset (right click on appli, run as, run config and in new window under startups):
appli download false, symbols true
FSBL download true, symbols true

add binary: download true, symbols false, address/offset 0x70100000

To setup everything needed for LCD, touchscreen etc:
Add to Application/User/Core:
 - BTC_STM/Drivers/BSP/STM32N6570-DK/stm32n6570_discovery_lcd.c

from hal:
  - stm32n6xx_hal_dma2d.c
  - stm32n6xx_hal_ltdc.c
  - stm32n6xx_hal_ltdc_ex.c

for touchscreen also:
  - stm32n6570_discovery_bus.c
 - stm32n6570_discovery_ts.c

from hal:
 - stm32n6xx_hal_i2c.c
 - stm32n6xx_hal_i2c_ex.c

also inside BTC_STM/Drivers/BSP/Components:
 - gt911.c
 - gt911_reg.c

For these two also add to compile include target.

There's ram map issue by default for LCD:
firmware/BTC_STM/STM32CubeIDE/Appli/STM32N657X0HXQ_ROMxspi2.ld

Specifically, config like this to work:
    - LCD_FB reserves the framebuffer at 0x34000000.
    - RAM now starts at 0x340C0000.
    - .lcd_framebuffer reserves the 800×480 RGB565 buffer.

At inference when you put the image in buffer (now also part of code) you need to clean cache:

SCB_CleanDCache_by_Addr((uint32_t *)stai_input[0],
                            STAI_NETWORK_IN_1_SIZE_BYTES);


To actually flash the stuff on the board:

```shell
$cp = "C:\ST\STM32CubeIDE_2.2.0\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.win32_2.2.500.202603051304\tools\bin\STM32_Programmer_CLI.exe"

$fsbl = "$pwd\firmware\BTC_STM\STM32CubeIDE\FSBL\Debug\STM32N6570-DK_FSBL-Trusted.bin"
$app  = "$pwd\firmware\BTC_STM\STM32CubeIDE\Appli\Debug\STM32N6570-DK_Appli-Trusted.bin"
```
Connect to the board in SWD mode:
```shell
& $cp -c port=SWD mode=HOTPLUG
```
Program the trusted FSBL at external-flash base:
```shell
& $cp -c port=SWD mode=HOTPLUG -d $fsbl 0x70000000
```
Program the trusted application at its application offset:
```shell
& $cp -c port=SWD mode=HOTPLUG -d $app 0x70100000
```

Then disconnect, power-cycle the board, and return BOOT1 to the normal boot setting. BOOT1=high is only for the
debug/programming workflow.
