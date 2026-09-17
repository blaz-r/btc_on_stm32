/*
 * Application copy of the STM32N6570-DK BSP configuration.
 *
 * ST's BSP integration requires this file in the application's include
 * directory.  Keeping it here makes the configuration selected by every BSP
 * component explicit (including the GT911 touch driver).
 */
#ifndef STM32N6570_DISCOVERY_CONF_H
#define STM32N6570_DISCOVERY_CONF_H

#include "stm32n6xx_hal.h"

#define USE_COM_LOG                         0U
#define USE_BSP_COM_FEATURE                 0U

/* The N6570-DK touch panel is GT911; this legacy BSP switch is retained by
   ST's template for compatibility with the common TS interface. */
#define USE_FT5336_TS_CTRL                  1U
#define USE_TS_GESTURE                      1U
#define USE_TS_MULTI_TOUCH                  1U
#define TS_TOUCH_NBR                        2U

#define LCD_LAYER_0_ADDRESS                 0x34000000
#define LCD_LAYER_1_ADDRESS                 0x340C0000

#define USE_AUDIO_CODEC_WM8904
#define DEFAULT_AUDIO_IN_BUFFER_SIZE        2048U

#define BSP_SDRAM_IT_PRIORITY               15U
#define BSP_BUTTON_USER1_IT_PRIORITY        15U
#define BSP_BUTTON_USER2_IT_PRIORITY        15U
#define BSP_BUTTON_TAMP_IT_PRIORITY         15U
#define BSP_AUDIO_OUT_IT_PRIORITY           14U
#define BSP_AUDIO_IN_IT_PRIORITY            15U
#define BSP_SD_IT_PRIORITY                  14U
#define BSP_SD_RX_IT_PRIORITY               14U
#define BSP_SD_TX_IT_PRIORITY               15U
#define BSP_TS_IT_PRIORITY                  15U

#define BSP_CAMERA_ISP_DEFAULT_WHITE_BALANCE    255U
#define BSP_CAMERA_ISP_DEFAULT_EXPOSURE         128U
#define BSP_CAMERA_ISP_DEFAULT_CONTRAST         130U
#define BSP_CAMERA_ISP_STATISTICS_AREA_HEIGHT   1940
#define BSP_CAMERA_ISP_STATISTICS_AREA_WIDTH    2592

#endif
