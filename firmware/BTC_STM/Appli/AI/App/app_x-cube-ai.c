
/**
  ******************************************************************************
  * @file    app_x-cube-ai.c
  * @author  X-CUBE-AI C code generator
  * @brief   AI program body
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

  /**
    * Description
    * Minimum template to show how to use the Neural-ART Embedded Client API
    *          Re-target of the printf function is out-of-scope.
    *
    *
    */

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/


/* System headers */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include "app_x-cube-ai.h"
#include "bsp_ai.h"

#include "stai.h"
#include "npu_init.h"



/* USER CODE BEGIN includes */

/* USER CODE END includes */

/* IO buffers ----------------------------------------------------------------*/


/* Input defs ----------------------------------------------------------------*/
#include "aiTestUtility.h"
/**

// Array to store the data of the input tensor
stai_ptr data_ins[] = {
}; 
*/

/* Output defs ----------------------------------------------------------------*/

/**

// c-array to store the data of the output tensor
stai_ptr data_outs[] = {
}; 
*/




/* Global byte buffer to save instantiated C-model network context */
STAI_NETWORK_CONTEXT_DECLARE(network_context, STAI_NETWORK_CONTEXT_SIZE)

/* Activations buffers -------------------------------------------------------*/




/* Entry points --------------------------------------------------------------*/

/* Array of pointer to manage the model's input/output tensors */
static stai_size in_length, out_length;
static stai_ptr stai_input[STAI_NETWORK_IN_NUM];
static stai_ptr stai_output[STAI_NETWORK_OUT_NUM];


/* 
 * Bootstrap
 */
int aiInit(void) {
  stai_return_code ret_code;
  
  /* 1: Initialize runtime library */
  ret_code = stai_runtime_init();
  if (ret_code != STAI_SUCCESS) {
    return (int)ret_code;
  }
  
  /* 2: Initialize network model context */
  ret_code = user_stai_network_init(network_context);
  if (ret_code != STAI_SUCCESS) {
    return (int)ret_code;
  }

  ret_code = stai_network_get_inputs(network_context, stai_input, &in_length);
  if (ret_code != STAI_SUCCESS) {
    return (int)ret_code;
  }
  
  ret_code = stai_network_get_outputs(network_context, stai_output, &out_length);
  if (ret_code != STAI_SUCCESS) {
    return (int)ret_code;
  }
  
  return (int)STAI_SUCCESS;
}

int aiDeinit(void) {
  stai_return_code ret_code;

  /* 1: Deinitialize network model context */
  ret_code = stai_network_deinit(network_context);
  
  /* 2: Deinitialize runtime library */
  ret_code = stai_runtime_deinit();

  return 0;
}

/* 
 * Run inference
 */
stai_return_code aiRun() {
  stai_return_code ret_code;

  /** Profiling code to calculate the inference time of the model. You can remove it if not needed */
  static uint32_t inference_nb = 0;
  static uint32_t total_cycles = 0;
  uint32_t start_tick, end_tick, end_dwt = 0;
  struct dwtTime t;
  cyclesCounterInit();

  LC_PRINT("---- Inference number %" PRIu32 " ----\r\n", inference_nb);
  LC_PRINT("Results for network \"%s\"\r\nRunning...\r\n", STAI_NETWORK_C_MODEL_NAME);
  cyclesCounterStart();
  start_tick = HAL_GetTick();


  /* Perform the inference */
  ret_code = stai_network_run(network_context, STAI_MODE_SYNC);
  if (ret_code != STAI_SUCCESS) {
      ret_code = stai_network_get_error(network_context);
      LC_PRINT("Inference failed with error code %s\r\n", stai_get_return_code_name(ret_code));
  };
  /** End of inference */
  
  /** Continue profiling */
  end_dwt = cyclesCounterEnd();
  total_cycles += end_dwt;
  end_tick = HAL_GetTick();
  dwtCyclesToTime(end_dwt, &t);

  LC_PRINT(" duration DWT    : %d.%03d ms\r\n", t.s * 1000 + t.ms, t.us);
  LC_PRINT(" duration SysTick: %" PRIu32" ms\r\n", end_tick - start_tick);
  LC_PRINT(" CPU cycles      : %" PRIu32 "\r\n", end_dwt);
  LC_PRINT(" CPU cycles (avg): %" PRIu32 "\r\n", total_cycles / ++inference_nb);
  LC_PRINT(" Sleep for 5s...\r\n");
  HAL_Delay(5000);

  return ret_code;
}


int acquire_and_process_data()
{
  /* USER CODE BEGIN acquire_and_process_data */
  /* fill the inputs of the c-model 
  for (int idx=0; idx < STAI_NETWORK_IN_NUM; idx++ )
  {
      stai_input[idx] = ....
  }

  */
  return 0;
  /* USER CODE END acquire_and_process_data */
}

int post_process()
{
  /* USER CODE BEGIN post_process */
  /* process the predictions
  for (int idx=0; idx < STAI_NETWORK_OUT_NUM; idx++ )
  {
      stai_output[idx] = ....
  }

  */
  return 0;
  /* USER CODE END post_process */
}



/* 
 * Example of main loop function
 */
void main_loop() {
  /* USER CODE BEGIN main_loop */
  while (1) {
    /* 1 - Acquire, pre-process and fill the input buffers */
    acquire_and_process_data();

    /* 2 - Call inference engine */
    aiRun();

    /* 3 - Post-process the predictions */
    post_process();
  }
  /* USER CODE END main_loop */
}


/* Entry points --------------------------------------------------------------*/



stai_return_code STM32CubeAI_Studio_AI_Init(void)
{
    stai_return_code ret_code;

    MX_UARTx_Init();
    aiPreInitialize();
    ret_code = (stai_return_code)aiInit();
    /* USER CODE BEGIN init */
    /* USER CODE END init */
    return ret_code;
}

void STM32CubeAI_Studio_AI_Process(void)
{
    main_loop();
} 

stai_return_code STM32CubeAI_Studio_AI_RunImagePair(const int8_t *input,
                                                     const int8_t **output)
{
    stai_return_code ret_code;

    if ((input == NULL) || (output == NULL))
    {
        return STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS;
    }

    memcpy(stai_input[0], input, STAI_NETWORK_IN_1_SIZE_BYTES);

    /* The NPU is a separate bus master.  Publish CPU-written input data before
       starting inference when the input region is D-cacheable. */
    SCB_CleanDCache_by_Addr((uint32_t *)stai_input[0],
                            STAI_NETWORK_IN_1_SIZE_BYTES);

    ret_code = stai_network_run(network_context, STAI_MODE_SYNC);
    if (ret_code == STAI_SUCCESS)
    {
        /* Fetch NPU-written output data before the CPU post-processes it. */
        SCB_InvalidateDCache_by_Addr((uint32_t *)stai_output[0],
                                     STAI_NETWORK_OUT_1_SIZE_BYTES);
        *output = (const int8_t *)stai_output[0];
    }
    return ret_code;
}

void STM32CubeAI_Studio_AI_Deinit(void)
{
    aiDeinit();
} 


#ifdef __cplusplus
}
#endif
