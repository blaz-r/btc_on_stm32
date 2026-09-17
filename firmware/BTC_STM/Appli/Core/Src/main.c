/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
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
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

#include "app_x-cube-ai.h"
#include "stm32n6570_discovery_lcd.h"
#include "stm32n6570_discovery_ts.h"
#include "stm32n6570_discovery_bus.h"
#include <math.h>

#include "images.h"

/* The LCD BSP is linked in a different XIP/RAM region than main().  Force
   this cross-region call to use an absolute veneer instead of a short branch. */
extern int32_t BSP_LCD_FillRect(uint32_t Instance, uint32_t Xpos,
                                uint32_t Ypos, uint32_t Width,
                                uint32_t Height, uint32_t Color)
                                __attribute__((long_call));
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* STM32N6570-DK green user LED (LED1): PO1, active high. */
#define USER_LED_GPIO_PORT       GPIOO
#define USER_LED_PIN             GPIO_PIN_1
#define USER_BUTTON_GPIO_PORT    GPIOC
#define USER_BUTTON_PIN          GPIO_PIN_13
#define AI_INIT_ERROR_BLINK_MS   100U
#define USER_BUTTON_DEBOUNCE_MS  50U

#define LCD_CLEAR_COLOR          LCD_COLOR_RGB565_WHITE
#define IMAGE_PAIR_BYTES         (6U * 96U * 96U)
#define LCD_FRAMEBUFFER_PIXELS   (LCD_DEFAULT_WIDTH * LCD_DEFAULT_HEIGHT)
#define NETWORK_OUTPUT_WIDTH      96U
#define NETWORK_OUTPUT_HEIGHT     96U
#define PREDICTION_DISPLAY_SIZE   256U
#define OUTPUT_QUANT_SCALE        0.0626182854175568f
#define OUTPUT_QUANT_ZERO_POINT   (-69.0f)
#define IMAGE_A_DISPLAY_X         0U
#define IMAGE_B_DISPLAY_X         (IMAGE_A_DISPLAY_X + PREDICTION_DISPLAY_SIZE)
#define PREDICTION_DISPLAY_X      (IMAGE_B_DISPLAY_X + PREDICTION_DISPLAY_SIZE)
#define LCD_STATUS_TEXT_X         24U
#define LCD_STATUS_TEXT_Y         300U
#define LCD_STATUS_TEXT_SCALE     4U
#define NAV_BUTTON_Y              365U
#define NAV_BUTTON_WIDTH          130U
#define NAV_BUTTON_HEIGHT         90U
#define NAV_LEFT_X                20U
#define NAV_RIGHT_X               (LCD_DEFAULT_WIDTH - NAV_LEFT_X - NAV_BUTTON_WIDTH)

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
CACHEAXI_HandleTypeDef hcacheaxi;

UART_HandleTypeDef huart1;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
static void MX_GPIO_Init(void);
static void MX_CACHEAXI_Init(void);
void MX_USART1_UART_Init(void);
static void SystemIsolation_Config(void);
static void LCD_ClearFrameBuffer(uint16_t color);
static void LCD_FlushFrameBuffer(void);
static void RunSingleImage(void);
static void LCD_DrawRgbImage(const uint8_t *image, uint32_t destination_x);
static void LCD_ShowReadyScreen(void);
static void LCD_ShowPrediction(const int8_t *output, uint32_t elapsed_us);
static void LCD_DrawText(const char *text, uint32_t x, uint32_t y, uint32_t scale);
static void LCD_DrawInferenceTime(uint32_t elapsed_us);
static void LCD_DrawNavigation(void);
static void LCD_FillRectangle(uint32_t x, uint32_t y, uint32_t width,
                              uint32_t height, uint16_t color);
static void InferenceTimer_Init(void);
static uint8_t UserButtonWasPressed(void);
static int8_t TouchNavigationWasPressed(void);
static int8_t image_pair_buffer[IMAGE_PAIR_BYTES] __attribute__((aligned(32)));
static uint32_t selected_image_pair = 0U;
static uint8_t touchscreen_ready = 0U;

/* Select the GT911's documented 0x5D (8-bit HAL address 0xBA) address.
   GT911 samples INT while RESET is asserted; merely releasing RESET, as the
   generic BSP does, is not sufficient when the panel has no fixed strap. */
static void Touchscreen_SelectGT911Address(void)
{
  GPIO_InitTypeDef gpio = {0};

  TS_INT_GPIO_CLK_ENABLE();
  TS_NRST_GPIO_CLK_ENABLE();

  gpio.Pin = TS_INT_PIN;
  gpio.Mode = GPIO_MODE_OUTPUT_PP;
  gpio.Pull = GPIO_NOPULL;
  gpio.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(TS_INT_GPIO_PORT, &gpio);
  HAL_GPIO_WritePin(TS_INT_GPIO_PORT, TS_INT_PIN, GPIO_PIN_RESET);

  gpio.Pin = TS_NRST_PIN;
  gpio.Mode = GPIO_MODE_OUTPUT_PP;
  gpio.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(TS_NRST_GPIO_PORT, &gpio);
  HAL_GPIO_WritePin(TS_NRST_GPIO_PORT, TS_NRST_PIN, GPIO_PIN_RESET);
  HAL_Delay(5U);
  HAL_GPIO_WritePin(TS_NRST_GPIO_PORT, TS_NRST_PIN, GPIO_PIN_SET);
  HAL_Delay(50U);

  /* INT becomes the controller's touch interrupt input after address select. */
  gpio.Pin = TS_INT_PIN;
  gpio.Mode = GPIO_MODE_INPUT;
  gpio.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(TS_INT_GPIO_PORT, &gpio);
  HAL_Delay(5U);
}
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  stai_return_code ai_init_status = STAI_ERROR_GENERIC;

  /* USER CODE END 1 */

  /* Enable the CPU Cache */

  /* Enable I-Cache---------------------------------------------------------*/
  SCB_EnableICache();

  /* Enable D-Cache---------------------------------------------------------*/
  SCB_EnableDCache();

  /* MCU Configuration--------------------------------------------------------*/
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_CACHEAXI_Init();
  MX_USART1_UART_Init();
  SystemIsolation_Config();
  /* USER CODE BEGIN 2 */
  /* Bring up the panel and present a deterministic blank frame while the
     network status LED continues to report initialization state. */
  if (BSP_LCD_Init(0, LCD_ORIENTATION_LANDSCAPE) != BSP_ERROR_NONE)
  {
    Error_Handler();
  }
  if (BSP_LCD_DisplayOn(0) != BSP_ERROR_NONE)
  {
    Error_Handler();
  }
  HAL_GPIO_WritePin(LCD_BL_CTRL_GPIO_Port, LCD_BL_CTRL_Pin, GPIO_PIN_SET);
  if (BSP_LCD_FillRect(0, 0, 0, LCD_DEFAULT_WIDTH, LCD_DEFAULT_HEIGHT,
                       LCD_CLEAR_COLOR) != BSP_ERROR_NONE)
  {
    Error_Handler();
  }
  LCD_ClearFrameBuffer(LCD_CLEAR_COLOR);
  {
    TS_NRST_GPIO_CLK_ENABLE();
    Touchscreen_SelectGT911Address();
    TS_Init_t touch_init =
    {
      .Width = LCD_DEFAULT_WIDTH,
      .Height = LCD_DEFAULT_HEIGHT,
      .Orientation = TS_SWAP_NONE,
      .Accuracy = 0U
    };
    /* The BSP expects the application to select the I2C2 kernel clock. */
    __HAL_RCC_I2C2_CONFIG(RCC_I2C2CLKSOURCE_PCLK1);
    const int32_t ts_status = BSP_TS_Init(0U, &touch_init);
    touchscreen_ready = (ts_status == BSP_ERROR_NONE) ? 1U : 0U;
  }
  ai_init_status = STM32CubeAI_Studio_AI_Init();

  if (ai_init_status == STAI_SUCCESS)
  {
    InferenceTimer_Init();
    LCD_ShowReadyScreen();
  }
  else
  {
    while (1)
    {
      HAL_GPIO_TogglePin(USER_LED_GPIO_PORT, USER_LED_PIN);
      HAL_Delay(AI_INIT_ERROR_BLINK_MS);
    }
  }
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */

  while (1)
  {
    if ((ai_init_status == STAI_SUCCESS) && (UserButtonWasPressed() != 0U))
      RunSingleImage();
    else if (ai_init_status == STAI_SUCCESS)
    {
      const int8_t navigation = TouchNavigationWasPressed();
      if (navigation < 0)
      {
        selected_image_pair = (selected_image_pair + IMAGE_PAIR_COUNT - 1U) % IMAGE_PAIR_COUNT;
        RunSingleImage();
      }
      else if (navigation > 0)
      {
        selected_image_pair = (selected_image_pair + 1U) % IMAGE_PAIR_COUNT;
        RunSingleImage();
      }
      else
      {
        HAL_Delay(1U);
      }
    }
    else if (ai_init_status != STAI_SUCCESS)
    {
      HAL_GPIO_TogglePin(USER_LED_GPIO_PORT, USER_LED_PIN);
      HAL_Delay(AI_INIT_ERROR_BLINK_MS);
    }
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief CACHEAXI Initialization Function
  * @param None
  * @retval None
  */
static void MX_CACHEAXI_Init(void)
{

  /* USER CODE BEGIN CACHEAXI_Init 0 */

  /* USER CODE END CACHEAXI_Init 0 */

  /* USER CODE BEGIN CACHEAXI_Init 1 */

  /* USER CODE END CACHEAXI_Init 1 */
  hcacheaxi.Instance = CACHEAXI;
  if (HAL_CACHEAXI_Init(&hcacheaxi) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN CACHEAXI_Init 2 */

  /* USER CODE END CACHEAXI_Init 2 */

}

/**
  * @brief RIF Initialization Function
  * @param None
  * @retval None
  */
  static void SystemIsolation_Config(void)
{

  /* USER CODE BEGIN RIF_Init 0 */

  /* USER CODE END RIF_Init 0 */

  /* set all required IPs as secure privileged */
  __HAL_RCC_RIFSC_CLK_ENABLE();

  /*RIMC configuration*/
  RIMC_MasterConfig_t RIMC_master = {0};
  RIMC_master.MasterCID = RIF_CID_1;
  RIMC_master.SecPriv = RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_NPRIV;
  HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_ETH1, &RIMC_master);

  HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_SDMMC2, &RIMC_master);
  HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_DMA2D, &RIMC_master);
  HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_LTDC1, &RIMC_master);
  HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_LTDC2, &RIMC_master);

  HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_DMA2D,
                                        RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_NPRIV);
  HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_LTDC,
                                        RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_NPRIV);
  HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_LTDCL1,
                                        RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_NPRIV);
  HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_LTDCL2,
                                        RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_NPRIV);
  /* The GT911 touchscreen controller is connected through I2C2.  Give the
     secure application the same access to that peripheral as the LCD blocks. */
  HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_I2C2,
                                        RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_NPRIV);

  /* RIF-Aware IPs Config */

  /* set up GPIO configuration */
  HAL_GPIO_ConfigPinAttributes(GPIOA,GPIO_PIN_11,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOB,GPIO_PIN_0,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOB,GPIO_PIN_1,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOB,GPIO_PIN_6,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOB,GPIO_PIN_7,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOB,GPIO_PIN_9,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOC,GPIO_PIN_0,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOC,GPIO_PIN_1,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOC,GPIO_PIN_2,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOC,GPIO_PIN_3,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOC,GPIO_PIN_4,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOC,GPIO_PIN_5,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOC,GPIO_PIN_8,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOC,GPIO_PIN_13,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOD,GPIO_PIN_2,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOD,GPIO_PIN_4,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOD,GPIO_PIN_10,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOD,GPIO_PIN_14,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_1,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_2,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_3,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_4,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_5,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_6,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_7,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_8,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOF,GPIO_PIN_4,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOG,GPIO_PIN_7,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOG,GPIO_PIN_10,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOH,GPIO_PIN_9,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_0,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_1,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_2,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_3,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_4,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_5,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_6,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_7,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_8,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_9,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_10,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_11,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOO,GPIO_PIN_0,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOO,GPIO_PIN_2,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOO,GPIO_PIN_3,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOO,GPIO_PIN_4,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOO,GPIO_PIN_5,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_0,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_1,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_2,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_3,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_4,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_5,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_6,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_7,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_8,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_9,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_10,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_11,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_12,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_13,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_14,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_15,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOQ,GPIO_PIN_0,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOQ,GPIO_PIN_1,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOQ,GPIO_PIN_2,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOQ,GPIO_PIN_3,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOQ,GPIO_PIN_4,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOQ,GPIO_PIN_5,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOQ,GPIO_PIN_6,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOQ,GPIO_PIN_7,GPIO_PIN_SEC|GPIO_PIN_NPRIV);

  /* USER CODE BEGIN RIF_Init 1 */

  /* Give the secure application access to the green user LED pin. */
  HAL_GPIO_ConfigPinAttributes(USER_LED_GPIO_PORT, USER_LED_PIN,
                               GPIO_PIN_SEC | GPIO_PIN_NPRIV);

  /* USER CODE END RIF_Init 1 */
  /* USER CODE BEGIN RIF_Init 2 */

  /* USER CODE END RIF_Init 2 */

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 921600;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart1, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart1, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOQ_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOO_CLK_ENABLE();
  __HAL_RCC_GPIOG_CLK_ENABLE();
  __HAL_RCC_GPION_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOQ, LCD_BL_CTRL_Pin|GPIO_PIN_3|PWR_SD_EN_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(EN_MODULE_GPIO_Port, EN_MODULE_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, PWR_USB2_EN_Pin|AUDIO_RST_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LCD_NRST_GPIO_Port, LCD_NRST_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(SD_SEL_GPIO_Port, SD_SEL_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(USB1_OCP_GPIO_Port, USB1_OCP_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pins : LCD_BL_CTRL_Pin PQ3 PWR_SD_EN_Pin */
  GPIO_InitStruct.Pin = LCD_BL_CTRL_Pin|GPIO_PIN_3|PWR_SD_EN_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOQ, &GPIO_InitStruct);

  /*Configure GPIO pin : USB1_INT_Pin */
  GPIO_InitStruct.Pin = USB1_INT_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(USB1_INT_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : EN_MODULE_Pin */
  GPIO_InitStruct.Pin = EN_MODULE_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(EN_MODULE_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : PQ4 TOF_LPn_Pin IMU_INT2_Pin IMU_INT1_Pin
                           TOF_INT_Pin */
  GPIO_InitStruct.Pin = GPIO_PIN_4|TOF_LPn_Pin|IMU_INT2_Pin|IMU_INT1_Pin
                          |TOF_INT_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOQ, &GPIO_InitStruct);

  /*Configure GPIO pin : NRST_CAM_Pin */
  GPIO_InitStruct.Pin = NRST_CAM_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(NRST_CAM_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : PWR_USB2_EN_Pin AUDIO_RST_Pin */
  GPIO_InitStruct.Pin = PWR_USB2_EN_Pin|AUDIO_RST_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pin : LCD_NRST_Pin */
  GPIO_InitStruct.Pin = LCD_NRST_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LCD_NRST_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : SD_SEL_Pin */
  GPIO_InitStruct.Pin = SD_SEL_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(SD_SEL_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : LED2_Pin */
  GPIO_InitStruct.Pin = LED2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(LED2_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : USB1_OCP_Pin */
  GPIO_InitStruct.Pin = USB1_OCP_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(USB1_OCP_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* Configure LED1 (green user LED) as an active-high push-pull output. */
  __HAL_RCC_GPIOO_CLK_ENABLE();
  HAL_GPIO_WritePin(USER_LED_GPIO_PORT, USER_LED_PIN, GPIO_PIN_RESET);

  GPIO_InitStruct.Pin = USER_LED_PIN;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(USER_LED_GPIO_PORT, &GPIO_InitStruct);

  /* USER1 (B2) is PC13 on STM32N6570-DK and is active high when pressed. */
  GPIO_InitStruct.Pin = USER_BUTTON_PIN;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLDOWN;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(USER_BUTTON_GPIO_PORT, &GPIO_InitStruct);

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

static void LCD_ClearFrameBuffer(uint16_t color)
{
  volatile uint16_t *framebuffer = (volatile uint16_t *)LCD_LAYER_0_ADDRESS;

  for (uint32_t pixel = 0; pixel < LCD_FRAMEBUFFER_PIXELS; ++pixel)
  {
    framebuffer[pixel] = color;
  }

  LCD_FlushFrameBuffer();
}

#define HW (NETWORK_OUTPUT_WIDTH * NETWORK_OUTPUT_HEIGHT)

static void prepare_ai_input(int8_t *dst)
{
    const ImagePair *pair = &image_pairs[selected_image_pair];

    for (uint32_t y = 0; y < NETWORK_OUTPUT_HEIGHT; y++)
    {
        for (uint32_t x = 0; x < NETWORK_OUTPUT_WIDTH; x++)
        {
            uint32_t pix = (y * NETWORK_OUTPUT_WIDTH + x);
            int rgb = pix * 3;

            /* Image A */
            dst[0 * HW + pix] = (int8_t)((int16_t)pair->a_rgb[rgb + 0] - 128);
            dst[1 * HW + pix] = (int8_t)((int16_t)pair->a_rgb[rgb + 1] - 128);
            dst[2 * HW + pix] = (int8_t)((int16_t)pair->a_rgb[rgb + 2] - 128);

            /* Image B */
            dst[3 * HW + pix] = (int8_t)((int16_t)pair->b_rgb[rgb + 0] - 128);
            dst[4 * HW + pix] = (int8_t)((int16_t)pair->b_rgb[rgb + 1] - 128);
            dst[5 * HW + pix] = (int8_t)((int16_t)pair->b_rgb[rgb + 2] - 128);
        }
    }
}

static void RunSingleImage(void)
{
  const int8_t *output = NULL;
  uint32_t start_cycles;
  uint32_t elapsed_cycles;
  uint32_t elapsed_us;
  const uint32_t core_clock_hz = HAL_RCC_GetHCLKFreq();

  prepare_ai_input(image_pair_buffer);

  /* Measure the whole model invocation (input copy/cache maintenance, NPU run,
     and output cache invalidation), but not the LCD rendering below. */
  start_cycles = DWT->CYCCNT;
  if (STM32CubeAI_Studio_AI_RunImagePair(image_pair_buffer, &output) == STAI_SUCCESS)
  {
    elapsed_cycles = DWT->CYCCNT - start_cycles;
    elapsed_us = (uint32_t)(((uint64_t)elapsed_cycles * 1000000ULL +
                             (core_clock_hz / 2U)) / core_clock_hz);
    LCD_ShowPrediction(output, elapsed_us);
  }
}


static void LCD_ShowReadyScreen(void)
{
  const ImagePair *pair = &image_pairs[selected_image_pair];

  LCD_ClearFrameBuffer(LCD_COLOR_RGB565_WHITE);
  LCD_DrawRgbImage(pair->a_rgb, IMAGE_A_DISPLAY_X);
  LCD_DrawRgbImage(pair->b_rgb, IMAGE_B_DISPLAY_X);
  LCD_DrawText("PRESS USER BUTTON", LCD_STATUS_TEXT_X, LCD_STATUS_TEXT_Y,
               LCD_STATUS_TEXT_SCALE);
  LCD_DrawNavigation();

  LCD_FlushFrameBuffer();
}

static void LCD_FlushFrameBuffer(void)
{
  /* Publish the framebuffer before LTDC fetches it. */
  SCB_CleanDCache();
  __DSB();
  __ISB();
}

static void LCD_ShowPrediction(const int8_t *output, uint32_t elapsed_us)
{
  const ImagePair *pair = &image_pairs[selected_image_pair];
  /* The LCD BSP configured layer 0 as RGB565 at LCD_LAYER_0_ADDRESS.
     Direct writes are appropriate here because we own the reserved framebuffer;
     D-cache is cleaned below so LTDC observes the new pixels. */
  volatile uint16_t *fb = (volatile uint16_t *)LCD_LAYER_0_ADDRESS;

  /* Keep the unused display area blank. */
  LCD_ClearFrameBuffer(LCD_COLOR_RGB565_WHITE);

  /* Input assets are HWC/interleaved RGB.  Draw them before the result so
     the panel reads left-to-right: image A, image B, prediction. */
  LCD_DrawRgbImage(pair->a_rgb, IMAGE_A_DISPLAY_X);
  LCD_DrawRgbImage(pair->b_rgb, IMAGE_B_DISPLAY_X);

  for (uint32_t y = 0; y < PREDICTION_DISPLAY_SIZE; ++y)
  {
    const uint32_t source_y = (y * NETWORK_OUTPUT_HEIGHT) / PREDICTION_DISPLAY_SIZE;
    for (uint32_t x = 0; x < PREDICTION_DISPLAY_SIZE; ++x)
    {
      const uint32_t source_x = (x * NETWORK_OUTPUT_WIDTH) / PREDICTION_DISPLAY_SIZE;
      const int8_t quantized_logit = output[source_y * NETWORK_OUTPUT_WIDTH + source_x];

      /* Match contest.py exactly: dequantize the int8 logit, then sigmoid. */
      const float logit = ((float)quantized_logit - OUTPUT_QUANT_ZERO_POINT) * OUTPUT_QUANT_SCALE;
      const float probability = 1.0f / (1.0f + expf(-logit));
      const uint8_t level = (uint8_t)(probability * 255.0f + 0.5f);

      /* A neutral grayscale probability map: black = 0, white = 1. */
      fb[y * LCD_DEFAULT_WIDTH + PREDICTION_DISPLAY_X + x] =
          (uint16_t)(((level & 0xF8U) << 8U) |
                     ((level & 0xFCU) << 3U) |
                     (level >> 3U));
    }
  }

  LCD_DrawInferenceTime(elapsed_us);
  LCD_DrawNavigation();

  /* LTDC reads SRAM directly, whereas the CPU has D-cache enabled. */
  LCD_FlushFrameBuffer();
}

static void LCD_DrawRgbImage(const uint8_t *image, uint32_t destination_x)
{
  volatile uint16_t *fb = (volatile uint16_t *)LCD_LAYER_0_ADDRESS;

  for (uint32_t y = 0; y < PREDICTION_DISPLAY_SIZE; ++y)
  {
    const uint32_t source_y = (y * NETWORK_OUTPUT_HEIGHT) / PREDICTION_DISPLAY_SIZE;
    for (uint32_t x = 0; x < PREDICTION_DISPLAY_SIZE; ++x)
    {
      const uint32_t source_x = (x * NETWORK_OUTPUT_WIDTH) / PREDICTION_DISPLAY_SIZE;
      const uint8_t *pixel = &image[(source_y * NETWORK_OUTPUT_WIDTH + source_x) * 3U];

      /* Convert the HWC RGB888 asset to the RGB565 framebuffer format. */
      fb[y * LCD_DEFAULT_WIDTH + destination_x + x] =
          (uint16_t)(((pixel[0] & 0xF8U) << 8U) |
                     ((pixel[1] & 0xFCU) << 3U) |
                     (pixel[2] >> 3U));
    }
  }
}

static void LCD_DrawText(const char *text, uint32_t x, uint32_t y, uint32_t scale)
{
  volatile uint16_t *fb = (volatile uint16_t *)LCD_LAYER_0_ADDRESS;
  static const char glyph_characters[] = "0123456789BCEFIMNOPRSTU:.";
  static const uint8_t glyph_columns[][5] =
  {
    {0x3EU, 0x51U, 0x49U, 0x45U, 0x3EU}, /* 0 */
    {0x00U, 0x42U, 0x7FU, 0x40U, 0x00U}, /* 1 */
    {0x42U, 0x61U, 0x51U, 0x49U, 0x46U}, /* 2 */
    {0x21U, 0x41U, 0x45U, 0x4BU, 0x31U}, /* 3 */
    {0x18U, 0x14U, 0x12U, 0x7FU, 0x10U}, /* 4 */
    {0x27U, 0x45U, 0x45U, 0x45U, 0x39U}, /* 5 */
    {0x3CU, 0x4AU, 0x49U, 0x49U, 0x30U}, /* 6 */
    {0x01U, 0x71U, 0x09U, 0x05U, 0x03U}, /* 7 */
    {0x36U, 0x49U, 0x49U, 0x49U, 0x36U}, /* 8 */
    {0x06U, 0x49U, 0x49U, 0x29U, 0x1EU}, /* 9 */
    {0x7FU, 0x49U, 0x49U, 0x49U, 0x36U}, /* B */
    {0x3EU, 0x41U, 0x41U, 0x41U, 0x22U}, /* C */
    {0x7FU, 0x49U, 0x49U, 0x49U, 0x41U}, /* E */
    {0x7FU, 0x09U, 0x09U, 0x09U, 0x01U}, /* F */
    {0x00U, 0x41U, 0x7FU, 0x41U, 0x00U}, /* I */
    {0x7FU, 0x02U, 0x0CU, 0x02U, 0x7FU}, /* M */
    {0x7FU, 0x02U, 0x04U, 0x08U, 0x7FU}, /* N */
    {0x3EU, 0x41U, 0x41U, 0x41U, 0x3EU}, /* O */
    {0x7FU, 0x09U, 0x09U, 0x09U, 0x06U}, /* P */
    {0x7FU, 0x09U, 0x19U, 0x29U, 0x46U}, /* R */
    {0x46U, 0x49U, 0x49U, 0x49U, 0x31U}, /* S */
    {0x01U, 0x01U, 0x7FU, 0x01U, 0x01U}, /* T */
    {0x3FU, 0x40U, 0x40U, 0x40U, 0x3FU}, /* U */
    {0x00U, 0x36U, 0x36U, 0x00U, 0x00U}, /* : */
    {0x00U, 0x60U, 0x60U, 0x00U, 0x00U}  /* . */
  };
  static const uint8_t blank_glyph[5] = {0U, 0U, 0U, 0U, 0U};

  while (*text != '\0')
  {
    const uint8_t *glyph = blank_glyph;

    /* 5x7 uppercase glyphs. Each byte is one column, with bit 0 at the top. */
    for (uint32_t index = 0U; glyph_characters[index] != '\0'; ++index)
    {
      if (*text == glyph_characters[index])
      {
        glyph = glyph_columns[index];
        break;
      }
    }

    for (uint32_t column = 0U; column < 5U; ++column)
    {
      for (uint32_t row = 0U; row < 7U; ++row)
      {
        if ((glyph[column] & (1U << row)) != 0U)
        {
          for (uint32_t dy = 0U; dy < scale; ++dy)
          {
            for (uint32_t dx = 0U; dx < scale; ++dx)
            {
              fb[(y + row * scale + dy) * LCD_DEFAULT_WIDTH + x + column * scale + dx] =
                  LCD_COLOR_RGB565_BLACK;
            }
          }
        }
      }
    }
    x += 6U * scale;
    ++text;
  }
}

static void LCD_DrawInferenceTime(uint32_t elapsed_us)
{
  char digits[10];
  uint32_t integer_ms = elapsed_us / 1000U;
  const uint32_t fraction_us = elapsed_us % 1000U;
  uint32_t digit_count = 0U;

  LCD_DrawText("INFERENCE: ", LCD_STATUS_TEXT_X, LCD_STATUS_TEXT_Y,
               LCD_STATUS_TEXT_SCALE);

  do
  {
    digits[digit_count++] = (char)('0' + (integer_ms % 10U));
    integer_ms /= 10U;
  } while (integer_ms != 0U);

  for (uint32_t i = 0U; i < digit_count; ++i)
  {
    char digit[2] = {digits[digit_count - 1U - i], '\0'};
    LCD_DrawText(digit, LCD_STATUS_TEXT_X + (11U + i) * 6U * LCD_STATUS_TEXT_SCALE,
                 LCD_STATUS_TEXT_Y, LCD_STATUS_TEXT_SCALE);
  }

  {
    const uint32_t suffix_x = LCD_STATUS_TEXT_X + (11U + digit_count) * 6U * LCD_STATUS_TEXT_SCALE;
    char fraction[4] =
    {
      (char)('0' + (fraction_us / 100U)),
      (char)('0' + ((fraction_us / 10U) % 10U)),
      (char)('0' + (fraction_us % 10U)),
      '\0'
    };
    LCD_DrawText(".", suffix_x, LCD_STATUS_TEXT_Y, LCD_STATUS_TEXT_SCALE);
    LCD_DrawText(fraction, suffix_x + 6U * LCD_STATUS_TEXT_SCALE,
                 LCD_STATUS_TEXT_Y, LCD_STATUS_TEXT_SCALE);
    LCD_DrawText(" MS", suffix_x + 24U * LCD_STATUS_TEXT_SCALE,
                 LCD_STATUS_TEXT_Y, LCD_STATUS_TEXT_SCALE);
  }
}

static void LCD_FillRectangle(uint32_t x, uint32_t y, uint32_t width,
                              uint32_t height, uint16_t color)
{
  volatile uint16_t *fb = (volatile uint16_t *)LCD_LAYER_0_ADDRESS;

  for (uint32_t row = 0U; row < height; ++row)
  {
    for (uint32_t column = 0U; column < width; ++column)
    {
      fb[(y + row) * LCD_DEFAULT_WIDTH + x + column] = color;
    }
  }
}

static void LCD_DrawNavigation(void)
{
  const uint16_t left_color = LCD_COLOR_RGB565_BLUE;
  const uint16_t right_color = LCD_COLOR_RGB565_RED;
  const uint16_t arrow_color = LCD_COLOR_RGB565_WHITE;
  volatile uint16_t *fb = (volatile uint16_t *)LCD_LAYER_0_ADDRESS;

  LCD_FillRectangle(NAV_LEFT_X, NAV_BUTTON_Y, NAV_BUTTON_WIDTH, NAV_BUTTON_HEIGHT,
                    left_color);
  LCD_FillRectangle(NAV_RIGHT_X, NAV_BUTTON_Y, NAV_BUTTON_WIDTH, NAV_BUTTON_HEIGHT,
                    right_color);

  /* Filled arrows make the touch targets readable without a larger font. */
  for (uint32_t row = 0U; row < 50U; ++row)
  {
    for (uint32_t column = 0U; column <= row; ++column)
    {
      fb[(NAV_BUTTON_Y + 20U + row) * LCD_DEFAULT_WIDTH + NAV_LEFT_X + 75U - column] = arrow_color;
      fb[(NAV_BUTTON_Y + 20U + row) * LCD_DEFAULT_WIDTH + NAV_RIGHT_X + 55U + column] = arrow_color;
    }
  }
}

static void InferenceTimer_Init(void)
{
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0U;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static uint8_t UserButtonWasPressed(void)
{
  static GPIO_PinState previous_state = GPIO_PIN_RESET;
  static uint32_t last_press_ms = 0U;
  const GPIO_PinState current_state = HAL_GPIO_ReadPin(USER_BUTTON_GPIO_PORT,
                                                        USER_BUTTON_PIN);
  const uint32_t now_ms = HAL_GetTick();
  uint8_t pressed = 0U;

  if ((current_state == GPIO_PIN_SET) && (previous_state == GPIO_PIN_RESET) &&
      ((now_ms - last_press_ms) >= USER_BUTTON_DEBOUNCE_MS))
  {
    last_press_ms = now_ms;
    pressed = 1U;
  }
  previous_state = current_state;
  return pressed;
}

static int8_t TouchNavigationWasPressed(void)
{
  static uint8_t touch_was_down = 0U;
  TS_State_t state;
  int8_t result = 0;

  if (touchscreen_ready == 0U)
  {
    return 0;
  }

  if (BSP_TS_GetState(0U, &state) != BSP_ERROR_NONE)
  {
    return 0;
  }

  if (state.TouchDetected == 0U)
  {
    touch_was_down = 0U;
    return 0;
  }

  if (touch_was_down != 0U)
  {
    return 0;
  }
  touch_was_down = 1U;

  /* Accept the normal landscape coordinates and the equivalent XY-swapped
     coordinates used by some panel revisions. */
  const uint8_t normal_y_button =
      (state.TouchY >= NAV_BUTTON_Y) &&
      (state.TouchY < (NAV_BUTTON_Y + NAV_BUTTON_HEIGHT));
  const uint8_t swapped_y_button =
      (state.TouchX >= NAV_BUTTON_Y) &&
      (state.TouchX < (NAV_BUTTON_Y + NAV_BUTTON_HEIGHT));
  const uint32_t button_x = normal_y_button ? state.TouchX : state.TouchY;

  if (normal_y_button || swapped_y_button)
  {
    if ((button_x >= NAV_LEFT_X) && (button_x < (NAV_LEFT_X + NAV_BUTTON_WIDTH)))
    {
      result = -1;
    }
    else if ((button_x >= NAV_RIGHT_X) &&
             (button_x < (NAV_RIGHT_X + NAV_BUTTON_WIDTH)))
    {
      result = 1;
    }
  }

  return result;
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
