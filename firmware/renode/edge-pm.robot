*** Settings ***
Documentation    Emulated bring-up of the edge-pm `--features sim` firmware on an STM32F411.
...              Boots the ELF in Renode, feeds the flash-baked `bearing_stream` through the
...              on-device pipeline, and asserts the UART shows the expected alert trajectory:
...              latch on `outer_race`, then clear via the normal-window hysteresis.
...
...              Build the ELF first:  cd firmware && cargo build --features sim
...              Run this test:        renode-test firmware/renode/edge-pm.robot
Suite Setup      Setup
Suite Teardown   Teardown
Test Setup       Reset Emulation
Resource         ${RENODEKEYWORDS}

*** Variables ***
${UART}          sysbus.usart2
${PLATFORM}      ${CURDIR}/stm32f411.repl
${ELF}           ${CURDIR}/../target/thumbv7em-none-eabihf/debug/firmware

*** Test Cases ***
Sim Stream Latches Then Clears An Alert
    Execute Command           mach create "edge-pm"
    Execute Command           machine LoadPlatformDescription @${PLATFORM}
    Execute Command           sysbus LoadELF @${ELF}

    Execute Command           showAnalyzer ${UART}
    Create Terminal Tester    ${UART}

    Start Emulation

    # Boot + source selection (the sim build has no ADXL345).
    Wait For Line On Uart     edge-pm: boot
    Wait For Line On Uart     edge-pm: sim source (baked bearing_stream)

    # The impulsive windows drive z-axis kurtosis past threshold -> outer_race alert latches.
    # conf=1.000 is the fp32 softmax PyTorch produced on the impulsive windows (see
    # tools/make_stream.py), reproduced on the Cortex-M4F.
    Wait For Line On Uart     ALERT outer_race conf=1.000    treatAsRegex=true

    # Three consecutive normal windows clear it (the AlertMachine hysteresis).
    Wait For Line On Uart     CLEAR

    # The whole baked stream (8 windows) was consumed without a fault/panic.
    Wait For Line On Uart     edge-pm: sim stream complete
