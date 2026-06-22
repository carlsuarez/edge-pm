*** Settings ***
Documentation    Emulated bring-up of the edge-pm `--features sim,q8` firmware on an STM32F411.
...              The Milestone-F integer-only int8 build: bakes the quantized v2
...              `bearing_stream_q8` checkpoint and runs `process_window_q8` on the Cortex-M4F
...              — int8 activations, i32 accumulation, fixed-point requantization, no float
...              between layers. Asserts the UART shows the same alert trajectory the fp32
...              build does — latch on `outer_race`, then clear — proving the integer-only
...              forward pass reproduces the host decision.
...
...              Build the ELF first:  cd firmware && cargo build --features sim,q8
...              Run this test:        renode-test firmware/renode/edge-pm-q8.robot
Suite Setup      Setup
Suite Teardown   Teardown
Test Setup       Reset Emulation
Resource         ${RENODEKEYWORDS}

*** Variables ***
${UART}          sysbus.usart2
${PLATFORM}      ${CURDIR}/stm32f411.repl
${ELF}           ${CURDIR}/../target/thumbv7em-none-eabihf/debug/firmware

*** Test Cases ***
Int8 Sim Stream Latches Then Clears An Alert
    Execute Command           mach create "edge-pm-q8"
    Execute Command           machine LoadPlatformDescription @${PLATFORM}
    Execute Command           sysbus LoadELF @${ELF}

    Execute Command           showAnalyzer ${UART}
    Create Terminal Tester    ${UART}

    Start Emulation

    # Boot + source selection (the sim build has no ADXL345).
    Wait For Line On Uart     edge-pm: boot
    Wait For Line On Uart     edge-pm: sim source (baked bearing_stream)

    # The integer-only forward pass classifies the impulsive windows as outer_race and latches.
    # conf=0.996 is the softmax the host (`host-sim replay`) produced, reproduced on the
    # Cortex-M4F purely in integer arithmetic (int8 weights+activations, i32 accumulation,
    # fixed-point requant) — float appears only at the final logit dequantize + softmax.
    Wait For Line On Uart     ALERT outer_race conf=0.996    treatAsRegex=true

    # Three consecutive normal windows clear it (the AlertMachine hysteresis).
    Wait For Line On Uart     CLEAR

    # The whole baked stream (8 windows) was consumed without a fault/panic.
    Wait For Line On Uart     edge-pm: sim stream complete
