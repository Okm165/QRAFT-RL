# Aurora-1 32-bit Microcontroller (synthetic example)

> Synthetic document for pipeline testing. Not a real device.

## Features

- 32-bit RISC core, up to 120 MHz
- 512 KB embedded flash, 128 KB SRAM
- 12-bit ADC with 16 input channels
- Four USARTs, two SPI controllers, two I2C controllers
- Eight-channel DMA controller
- 1.8 V to 3.6 V supply, -40 °C to +85 °C
- 64-pin LQFP package

## Memory map

| Region | Start address | Size |
|---|---:|---:|
| Flash | 0x0000_0000 | 512 KB |
| SRAM | 0x2000_0000 | 128 KB |
| Peripherals | 0x4000_0000 | 64 KB |
| DMA registers | 0x4001_0000 | 4 KB |

## Pin multiplexing

| Pin | Primary signal | Alternate function |
|---|---|---|
| PA0 | USART0_TXD | ADC0 |
| PA1 | USART0_RXD | ADC1 |
| PA2 | SPI0_SCK | I2C0_SCL |
| PA3 | SPI0_MOSI | I2C0_SDA |
| PB0 | DMA_REQ0 | GPIO |
| PB1 | DMA_REQ1 | GPIO |

## DMA controller

The DMA controller has eight independent channels. Each channel supports
memory-to-memory, memory-to-peripheral, and peripheral-to-memory transfers.
Arbitration is round-robin by default. A channel can be assigned a fixed
priority by setting the PRIORITY field in its control register. The controller
raises a transfer-complete interrupt after the programmed transfer count
reaches zero.

## ADC conversion

1. Enable the ADC clock in the peripheral clock register.
2. Select the input channel in ADC_CHANNEL.
3. Set ADC_START to begin one conversion.
4. Wait for ADC_READY.
5. Read the 12-bit result from ADC_DATA.

The ADC reference is the supply voltage. The maximum conversion rate is
one million samples per second.

## USART baud rate

The baud rate is generated from the peripheral clock:

```
baud = peripheral_clock / (16 * (BRR + 1))
```

For a 48 MHz peripheral clock and 115200 baud, BRR is 25. The transmitter and
receiver can be enabled independently.

## SPI operation

The SPI controller supports master and slave modes, four wire modes, and
8-bit or 16-bit transfers. In master mode, the clock polarity and phase are
selected by CPOL and CPHA. The transfer-complete flag is set after the last
bit is shifted out.

## Electrical characteristics

| Parameter | Minimum | Typical | Maximum |
|---|---:|---:|---:|
| Supply voltage | 1.8 V | 3.3 V | 3.6 V |
| Operating temperature | -40 °C | 25 °C | +85 °C |
| Active current | - | 18 mA | 25 mA |
| Sleep current | - | 12 µA | 20 µA |

## Ordering information

The Aurora-1 is available in a 64-pin LQFP package. The part number suffix
encodes the flash size and temperature range. Contact the distributor for
availability.
