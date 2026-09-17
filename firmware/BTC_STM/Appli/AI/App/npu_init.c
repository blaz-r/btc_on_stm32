  /**
  ******************************************************************************
  * @file    npu_init.c
  * @author  GPM/AIS Application Team
  * @brief   Collection of functions to perform main configurations in main.c
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

#include <string.h>     // Used for memset

#include "npu_init.h"
#include "app_config.h"
#include "npu_cache.h"  // Used in NPU_config

#define USE_NPU_CACHE



static uint32_t Get_RISAF_Max_Addr(RISAF_TypeDef *risaf)
{
  uint32_t max_addr = 0U;
  if      ((risaf == RISAF1_S)  || (risaf == RISAF1_NS))  {max_addr = RISAF1_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF2_S)  || (risaf == RISAF2_NS))  {max_addr = RISAF2_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF3_S)  || (risaf == RISAF3_NS))  {max_addr = RISAF3_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF4_S)  || (risaf == RISAF4_NS))  {max_addr = RISAF4_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF5_S)  || (risaf == RISAF5_NS))  {max_addr = RISAF5_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF6_S)  || (risaf == RISAF6_NS))  {max_addr = RISAF6_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF7_S)  || (risaf == RISAF7_NS))  {max_addr = RISAF7_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF8_S)  || (risaf == RISAF8_NS))  {max_addr = RISAF8_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF9_S)  || (risaf == RISAF9_NS))  {max_addr = RISAF9_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF11_S) || (risaf == RISAF11_NS)) {max_addr = RISAF11_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF12_S) || (risaf == RISAF12_NS)) {max_addr = RISAF12_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF13_S) || (risaf == RISAF13_NS)) {max_addr = RISAF13_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF14_S) || (risaf == RISAF14_NS)) {max_addr = RISAF14_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF15_S) || (risaf == RISAF15_NS)) {max_addr = RISAF15_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF21_S) || (risaf == RISAF21_NS)) {max_addr = RISAF21_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF22_S) || (risaf == RISAF22_NS)) {max_addr = RISAF22_LIMIT_ADDRESS_SPACE_SIZE;}
  else if ((risaf == RISAF23_S) || (risaf == RISAF23_NS)) {max_addr = RISAF23_LIMIT_ADDRESS_SPACE_SIZE;}
  return max_addr;
}

static void Set_RISAF_Default(RISAF_TypeDef *risaf)
{
  RISAF_BaseRegionConfig_t risaf_conf;  
  risaf_conf.StartAddress = 0x0;
  risaf_conf.EndAddress   = Get_RISAF_Max_Addr(risaf); /* as the default config */
  risaf_conf.Filtering    = RISAF_FILTER_ENABLE; // Base region enable (otherwise access control is secure, privileged, trusted domain CID = 1)
  risaf_conf.PrivWhitelist  = RIF_CID_NONE; // apps running in all compartments can access to region in priv/unpriv mode
  risaf_conf.ReadWhitelist  = RIF_CID_MASK; // apps running in all compartments can R in this region
  risaf_conf.WriteWhitelist = RIF_CID_MASK; // apps running in all compartments can W in this region
  // Configure 2 regions with this config, fully overlapping, one for secure one for non secure accesses:
  risaf_conf.Secure = RIF_ATTRIBUTE_SEC;    // Only secure requests can access this region
  HAL_RIF_RISAF_ConfigBaseRegion(risaf, 0, &risaf_conf);
  risaf_conf.Secure = RIF_ATTRIBUTE_NSEC;    // Only non-secure requests can access this region
  HAL_RIF_RISAF_ConfigBaseRegion(risaf, 1, &risaf_conf);
}


void Set_CLK_Sleep_Mode(void)
{
  /* Leave clocks enabled in Low Power modes */
  // Low-power clock enable misc
  __HAL_RCC_DBG_CLK_SLEEP_ENABLE();
  __HAL_RCC_XSPIPHYCOMP_CLK_SLEEP_ENABLE();
  
  // Low-power clock enable for memories
  __HAL_RCC_AXISRAM1_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_AXISRAM2_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_AXISRAM3_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_AXISRAM4_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_AXISRAM5_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_AXISRAM6_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_FLEXRAM_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_CACHEAXIRAM_MEM_CLK_SLEEP_ENABLE();
  // LP clock AHB1: None
  // LP clock AHB2: None
  // LP clock AHB3
  __HAL_RCC_RIFSC_CLK_SLEEP_ENABLE();
  __HAL_RCC_RISAF_CLK_SLEEP_ENABLE();
  __HAL_RCC_IAC_CLK_SLEEP_ENABLE();
  // LP clock AHB4: None
  // LP clocks AHB5
  __HAL_RCC_XSPI1_CLK_SLEEP_ENABLE();
  __HAL_RCC_XSPI2_CLK_SLEEP_ENABLE();
  __HAL_RCC_CACHEAXI_CLK_SLEEP_ENABLE();
  __HAL_RCC_NPU_CLK_SLEEP_ENABLE();
  // LP clocks APB1: None
  // LP clocks APB2
  __HAL_RCC_USART1_CLK_SLEEP_ENABLE();
  // LP clocks APB4: None
  // LP clocks APB5: None
}


#if defined(LL_ATON_PLATFORM)
void NPU_Config(void)
{
  // Enable NPU
  __HAL_RCC_NPU_CLK_ENABLE();
  __HAL_RCC_NPU_FORCE_RESET();
  __HAL_RCC_NPU_RELEASE_RESET();
  // Enable Cache-AXI
  __HAL_RCC_CACHEAXI_CLK_ENABLE();
  __HAL_RCC_CACHEAXI_FORCE_RESET();
  __HAL_RCC_CACHEAXI_RELEASE_RESET();
  
  // __HAL_RCC_CACHEAXI_CLK_SLEEP_DISABLE();
  // __HAL_RCC_NPU_CLK_SLEEP_DISABLE();
  // __HAL_RCC_RAMCFG_CLK_SLEEP_DISABLE();
  
#ifdef USE_NPU_CACHE
   npu_cache_enable(); // Useless: already enabled by init
#else
   npu_cache_disable();
#endif

  RIMC_MasterConfig_t master_conf;
  /* Enable Secure access for NPU */
  master_conf.MasterCID = RIF_CID_1;    // Master CID = 1
  master_conf.SecPriv = RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_PRIV; // Priviledged secure
  HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_NPU, &master_conf);  
  HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_NPU, RIF_ATTRIBUTE_PRIV | RIF_ATTRIBUTE_SEC);
}
#endif

void RISAF_Config(void)
{
  /*
  *  Note: before to set a risaf for a given IP, the IP
  *        should be clocked.
  */
  Set_RISAF_Default(RISAF2_S);          /* SRAM1_AXI */
  Set_RISAF_Default(RISAF3_S);          /* SRAM2_AXI */

#if defined(LL_ATON_PLATFORM)
  Set_RISAF_Default(RISAF4_S);          /* NPU MST0 */
  Set_RISAF_Default(RISAF5_S);          /* NPU MST1 */
#endif
  
  Set_RISAF_Default(RISAF6_S);          /* SRAM3,4,5,6_AXI */
  Set_RISAF_Default(RISAF7_S);          /* FLEXMEM */
  
#if defined(LL_ATON_PLATFORM)
#ifdef USE_NPU_CACHE
  Set_RISAF_Default(RISAF8_S);          /* NPU_CACHE */
  Set_RISAF_Default(RISAF15_S);         /* NPU_CACHE config */
#endif  
#endif
  
  // Set_RISAF_Default(RISAF9_S);       /* VENC */
  

#if (USE_EXTERNAL_RAM)
  Set_RISAF_Default(RISAF11_S);         /* OCTOSPI1 0x9000 0000 */
#endif
  Set_RISAF_Default(RISAF12_S);         /* OCTOSPI2 0x7000 0000 */
  // Set_RISAF_Default(RISAF13_S);      /* OCTOSPI3 0x8000 0000 */
  
}



void SystemInit_POST(void)
{  
  __HAL_RCC_SYSCFG_CLK_ENABLE();
  __HAL_RCC_CRC_CLK_ENABLE();
   
  /* Enable NPU RAMs (4x448KB) + CACHEAXI */
  RCC->MEMENR |= RCC_MEMENR_AXISRAM3EN | RCC_MEMENR_AXISRAM4EN | RCC_MEMENR_AXISRAM5EN | RCC_MEMENR_AXISRAM6EN;
  RCC->MEMENR |= RCC_MEMENR_CACHEAXIRAMEN; // RCC_MEMENR_NPUCACHERAMEN;
  
  RAMCFG_SRAM2_AXI->CR &= ~RAMCFG_CR_SRAMSD;
  RAMCFG_SRAM3_AXI->CR &= ~RAMCFG_CR_SRAMSD;
  RAMCFG_SRAM4_AXI->CR &= ~RAMCFG_CR_SRAMSD;
  RAMCFG_SRAM5_AXI->CR &= ~RAMCFG_CR_SRAMSD;
  RAMCFG_SRAM6_AXI->CR &= ~RAMCFG_CR_SRAMSD;

  // Set low-power mode for clocks
  Set_CLK_Sleep_Mode();

  /* Allow caches to be activated. Default value is 1, but the current boot sets it to 0 */
  MEMSYSCTL->MSCR |= MEMSYSCTL_MSCR_DCACTIVE_Msk | MEMSYSCTL_MSCR_ICACTIVE_Msk;
}

void aiPreInitialize(void)
{
    SystemInit_POST();
#if defined(LL_ATON_PLATFORM)
  NPU_Config();
#endif
  RISAF_Config();
}