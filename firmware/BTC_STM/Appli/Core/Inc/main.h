/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined ( __ICCARM__ )
#  define CMSE_NS_CALL  __cmse_nonsecure_call
#  define CMSE_NS_ENTRY __cmse_nonsecure_entry
#else
#  define CMSE_NS_CALL  __attribute((cmse_nonsecure_call))
#  define CMSE_NS_ENTRY __attribute((cmse_nonsecure_entry))
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32n6xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* Function pointer declaration in non-secure*/
#if defined ( __ICCARM__ )
typedef void (CMSE_NS_CALL *funcptr)(void);
#else
typedef void CMSE_NS_CALL (*funcptr)(void);
#endif

/* typedef for non-secure callback functions */
typedef funcptr funcptr_NS;

/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define LCD_BL_CTRL_Pin GPIO_PIN_6
#define LCD_BL_CTRL_GPIO_Port GPIOQ
#define OSC_IN_Pin GPIO_PIN_0
#define OSC_IN_GPIO_Port GPIOH
#define USB1_INT_Pin GPIO_PIN_10
#define USB1_INT_GPIO_Port GPIOD
#define OSC_OUT_Pin GPIO_PIN_1
#define OSC_OUT_GPIO_Port GPIOH
#define VCP_TX_Pin GPIO_PIN_5
#define VCP_TX_GPIO_Port GPIOE
#define EN_MODULE_Pin GPIO_PIN_2
#define EN_MODULE_GPIO_Port GPIOD
#define TOF_LPn_Pin GPIO_PIN_5
#define TOF_LPn_GPIO_Port GPIOQ
#define NRST_CAM_Pin GPIO_PIN_8
#define NRST_CAM_GPIO_Port GPIOC
#define VCP_RX_Pin GPIO_PIN_6
#define VCP_RX_GPIO_Port GPIOE
#define PWR_USB2_EN_Pin GPIO_PIN_9
#define PWR_USB2_EN_GPIO_Port GPIOB
#define IMU_INT2_Pin GPIO_PIN_2
#define IMU_INT2_GPIO_Port GPIOQ
#define IMU_INT1_Pin GPIO_PIN_1
#define IMU_INT1_GPIO_Port GPIOQ
#define TOF_INT_Pin GPIO_PIN_0
#define TOF_INT_GPIO_Port GPIOQ
#define LCD_NRST_Pin GPIO_PIN_1
#define LCD_NRST_GPIO_Port GPIOE
#define PWR_SD_EN_Pin GPIO_PIN_7
#define PWR_SD_EN_GPIO_Port GPIOQ
#define SD_SEL_Pin GPIO_PIN_5
#define SD_SEL_GPIO_Port GPIOO
#define AUDIO_RST_Pin GPIO_PIN_1
#define AUDIO_RST_GPIO_Port GPIOB
#define LED2_Pin GPIO_PIN_10
#define LED2_GPIO_Port GPIOG
#define USB1_OCP_Pin GPIO_PIN_7
#define USB1_OCP_GPIO_Port GPION

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
