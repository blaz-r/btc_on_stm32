/**
  ******************************************************************************
  * @file    npu_init.h
  * @author  GPM/AIS Application Team
  * @brief   Header file for npu_init.c
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

#ifndef NPU_INIT_H
#define NPU_INIT_H

#include <string.h>
#include "npu_init.h"
#include "app_config.h"
#include "npu_cache.h"
#include "stm32n6xx_hal.h"

#ifdef __cplusplus
extern "C" {
#endif


void Set_CLK_Sleep_Mode(void);
void UART_Config(void);
#if defined(LL_ATON_PLATFORM)
void NPU_Config(void);
#endif
void aiPreInitialize(void);
void RISAF_Config(void);
void SystemInit_POST(void);

#ifdef __cplusplus
}
#endif

#endif // NPU_INIT_H
