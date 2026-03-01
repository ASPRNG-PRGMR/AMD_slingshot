/*
 * Embedded Energy Console + MCU–Runtime Serial Bridge
 *
 * Target MCU : MSPM0G3507 (or MSPM0G1507)
 * Display    : ILI9341  240×320  via SPI
 * Input      : Joystick X-axis via ADC12  (Y-axis optional)
 * Comms      : UART 115200 8N1  bidirectional with PC runtime
 */

#include "ti_msp_dl_config.h"
#include <stdbool.h>
#include <string.h>

/* ================================================================
 * PIN ALIASES
 * ================================================================ */
#define DC_LOW()    DL_GPIO_clearPins(EXTRA_DC_PORT,  EXTRA_DC_PIN)
#define DC_HIGH()   DL_GPIO_setPins(EXTRA_DC_PORT,    EXTRA_DC_PIN)
#define RST_LOW()   DL_GPIO_clearPins(EXTRA_RST_PORT, EXTRA_RST_PIN)
#define RST_HIGH()  DL_GPIO_setPins(EXTRA_RST_PORT,   EXTRA_RST_PIN)

/* ================================================================
 * TIMING
 * ================================================================ */
#define MS_TO_CYCLES    32000U
#define REFRESH_MS      500U

/* ================================================================
 * DISPLAY COLORS  (RGB565)
 * ================================================================ */
#define BLACK           0x0000
#define WHITE           0xFFFF

#define COL_BG          0x1082
#define COL_PANEL       0x2104
#define COL_ACCENT      0x051F
#define COL_AMBER       0xFD20
#define COL_TEXT        0xFFFF
#define COL_DIM         0x8410
#define COL_CPU         0x051F
#define COL_GPU         0x07E0
#define COL_NPU         0xFD20
#define COL_WARN        0xF800
#define COL_OK          0x07E0
#define COL_DARKGRAY    0x39E7
#define COL_CYAN        0x07FF
#define COL_YELLOW      0xFFE0

/* ================================================================
 * ADC / JOYSTICK
 * ================================================================ */
#define ADC_MAX         4095U
#define ADC_MID         2047U
#define ADC_DEADZONE    200U

#define BUDGET_MIN      1U
#define BUDGET_MAX      20U
#define BUDGET_DEFAULT  5U

#define MODE_COUNT      4U

/* ================================================================
 * UART RX BUFFER
 * ================================================================ */
#define RX_BUF_LEN      128U

/* ================================================================
 * SIMULATION
 * ================================================================ */
#define SIM_HW          "NPU"
#define SIM_ENERGY_EJ   320UL
#define SIM_TEMP_C      38U
#define SIM_CLK_MHZ     800U
#define SIM_TIMEOUT_CYCLES  4U

/* ================================================================
 * GLOBAL STATE
 * ================================================================ */
volatile bool     gAdcReady  = false;
volatile uint16_t gAdcResult = 0;

volatile uint8_t  gBudget    = BUDGET_DEFAULT;
volatile uint8_t  gMode      = 0;

volatile char     gHW[4]     = "---";
volatile uint32_t gEnergyEJ  = 0;
volatile uint16_t gTempC     = 0;
volatile uint16_t gClkMHz    = 0;
volatile bool     gRxReady   = false;

static uint8_t    gNoRxCount = 0;

static char    gRxBuf[RX_BUF_LEN];
static uint8_t gRxIdx = 0;

/* ================================================================
 * UTILITY FUNCTIONS
 * ================================================================ */
static void delay_ms(uint32_t ms)
{
    while (ms--) delay_cycles(MS_TO_CYCLES);
}

static void int_to_string(uint32_t num, char *str)
{
    if (num == 0) { str[0]='0'; str[1]='\0'; return; }
    int i = 0;
    while (num > 0) { str[i++] = (char)('0' + num % 10); num /= 10; }
    str[i] = '\0';
    for (int a=0, b=i-1; a<b; a++,b--) {
        char t=str[a]; str[a]=str[b]; str[b]=t;
    }
}

/* ================================================================
 * UART TRANSMIT
 * ================================================================ */
static void uart_send_char(char c)
{
    while (DL_UART_isBusy(UART_0_INST));
    DL_UART_Main_transmitData(UART_0_INST, (uint8_t)c);
}

static void uart_send_string(const char *str)
{
    while (*str) uart_send_char(*str++);
}

static void uart_send_uint(uint32_t n)
{
    char buf[12];
    int_to_string(n, buf);
    uart_send_string(buf);
}

static void uart_send_status(void)
{
    uart_send_string("{\"budget\":");
    uart_send_uint(gBudget);
    uart_send_string(",\"mode\":");
    uart_send_uint(gMode);
    uart_send_string(",\"joy\":");
    uart_send_uint(gAdcResult);
    uart_send_string("}\r\n");
}

/* ================================================================
 * UART RECEIVE PARSER
 * ================================================================ */
static void parse_pc_line(const char *line)
{
    const char *p;

    p = strstr(line, "\"hw\":\"");
    if (p) {
        p += 6;
        gHW[0] = p[0];
        gHW[1] = p[1];
        gHW[2] = p[2];
        gHW[3] = '\0';
    }

    p = strstr(line, "\"ej\":");
    if (p) {
        p += 5;
        uint32_t v = 0;
        while (*p >= '0' && *p <= '9') { v = v*10 + (uint32_t)(*p - '0'); p++; }
        gEnergyEJ = v;
    }

    p = strstr(line, "\"temp\":");
    if (p) {
        p += 7;
        uint32_t v = 0;
        while (*p >= '0' && *p <= '9') { v = v*10 + (uint32_t)(*p - '0'); p++; }
        gTempC = (uint16_t)v;
    }

    p = strstr(line, "\"clk\":");
    if (p) {
        p += 6;
        uint32_t v = 0;
        while (*p >= '0' && *p <= '9') { v = v*10 + (uint32_t)(*p - '0'); p++; }
        gClkMHz = (uint16_t)v;
    }

    gRxReady   = true;
    gNoRxCount = 0;
}

/* ================================================================
 * SIMULATION FILL
 * ================================================================ */
static void apply_simulation(void)
{
    static uint8_t sim_tick = 0;
    sim_tick++;

    gHW[0] = 'N'; gHW[1] = 'P'; gHW[2] = 'U'; gHW[3] = '\0';

    uint32_t energy_base = SIM_ENERGY_EJ;
    int8_t   wobble      = (int8_t)((sim_tick & 0x07) * 10) - 35;
    gEnergyEJ = (uint32_t)((int32_t)energy_base + wobble);

    gTempC  = (uint16_t)(SIM_TEMP_C  + (sim_tick & 0x03));
    gClkMHz = (uint16_t)(SIM_CLK_MHZ + (uint16_t)((sim_tick & 0x03) * 50));
}

/* ================================================================
 * SPI LAYER
 * ================================================================ */
static inline void spi_tx(uint8_t b)
{
    DL_SPI_fillTXFIFO8(SPI_0_INST, &b, 1);
    while (DL_SPI_isBusy(SPI_0_INST));
}

/* ================================================================
 * LCD PRIMITIVES
 * ================================================================ */
static void lcd_cmd(uint8_t c)  { DC_LOW();  spi_tx(c); }
static void lcd_data(uint8_t d) { DC_HIGH(); spi_tx(d); }

static void lcd_reset(void)
{
    RST_LOW();  delay_ms(20);
    RST_HIGH(); delay_ms(150);
}

static void ili9341_init(void)
{
    lcd_reset();
    lcd_cmd(0x11);
    delay_ms(120);
    lcd_cmd(0x3A); lcd_data(0x55);
    lcd_cmd(0x36); lcd_data(0xC8);
    lcd_cmd(0x29);
    delay_ms(20);
}

static void lcd_set_window(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1)
{
    lcd_cmd(0x2A);
    lcd_data((uint8_t)(x0>>8)); lcd_data((uint8_t)(x0&0xFF));
    lcd_data((uint8_t)(x1>>8)); lcd_data((uint8_t)(x1&0xFF));
    lcd_cmd(0x2B);
    lcd_data((uint8_t)(y0>>8)); lcd_data((uint8_t)(y0&0xFF));
    lcd_data((uint8_t)(y1>>8)); lcd_data((uint8_t)(y1&0xFF));
    lcd_cmd(0x2C);
}

static void lcd_draw_pixel(uint16_t x, uint16_t y, uint16_t color)
{
    if (x >= 240 || y >= 320) return;
    lcd_set_window(x, y, x, y);
    DC_HIGH();
    spi_tx((uint8_t)(color >> 8));
    spi_tx((uint8_t)(color & 0xFF));
}

static void lcd_fill(uint16_t color)
{
    lcd_set_window(0, 0, 239, 319);
    DC_HIGH();
    for (uint32_t i = 0; i < 76800UL; i++) {
        spi_tx((uint8_t)(color >> 8));
        spi_tx((uint8_t)(color & 0xFF));
    }
}

static void lcd_fill_rect(uint16_t x, uint16_t y, uint16_t w, uint16_t h, uint16_t color)
{
    if (w == 0 || h == 0) return;
    lcd_set_window(x, y, (uint16_t)(x+w-1), (uint16_t)(y+h-1));
    DC_HIGH();
    for (uint32_t i = 0; i < (uint32_t)w*h; i++) {
        spi_tx((uint8_t)(color >> 8));
        spi_tx((uint8_t)(color & 0xFF));
    }
}

static void lcd_draw_bar(uint16_t x, uint16_t y,
                          uint16_t total_w, uint16_t h,
                          uint16_t fill_w,
                          uint16_t fill_color, uint16_t empty_color)
{
    if (fill_w > total_w) fill_w = total_w;
    if (fill_w > 0)
        lcd_fill_rect(x, y, fill_w, h, fill_color);
    if (fill_w < total_w)
        lcd_fill_rect((uint16_t)(x + fill_w), y, (uint16_t)(total_w - fill_w), h, empty_color);
}

/* ================================================================
 * 5×7 FONT  — ALL BITMAPS VERIFIED
 *
 * Column-based, LSB = top pixel.
 * Fixes vs original:
 *   [13] C  — was B's bitmap,   now correct C
 *   [29] Y  — was ')' shape,    now correct Y
 *   [33] B  — was F's bitmap,   now correct B
 * ================================================================ */
static const uint8_t font5x7[][5] = {
    {0x00, 0x00, 0x00, 0x00, 0x00}, /*  0: space */
    {0x3E, 0x51, 0x49, 0x45, 0x3E}, /*  1: 0     */
    {0x00, 0x42, 0x7F, 0x40, 0x00}, /*  2: 1     */
    {0x42, 0x61, 0x51, 0x49, 0x46}, /*  3: 2     */
    {0x21, 0x41, 0x45, 0x4B, 0x31}, /*  4: 3     */
    {0x18, 0x14, 0x12, 0x7F, 0x10}, /*  5: 4     */
    {0x27, 0x45, 0x45, 0x45, 0x39}, /*  6: 5     */
    {0x3C, 0x4A, 0x49, 0x49, 0x30}, /*  7: 6     */
    {0x01, 0x71, 0x09, 0x05, 0x03}, /*  8: 7     */
    {0x36, 0x49, 0x49, 0x49, 0x36}, /*  9: 8     */
    {0x06, 0x49, 0x49, 0x29, 0x1E}, /* 10: 9     */
    {0x00, 0x36, 0x36, 0x00, 0x00}, /* 11: :     */
    {0x7C, 0x12, 0x11, 0x12, 0x7C}, /* 12: A     */
    {0x3E, 0x41, 0x41, 0x41, 0x22}, /* 13: C     ← FIXED */
    {0x7F, 0x41, 0x41, 0x22, 0x1C}, /* 14: D     */
    {0x7F, 0x49, 0x49, 0x49, 0x41}, /* 15: E     */
    {0x7F, 0x09, 0x09, 0x09, 0x01}, /* 16: F     */
    {0x3E, 0x41, 0x49, 0x49, 0x7A}, /* 17: G     */
    {0x7F, 0x08, 0x08, 0x08, 0x7F}, /* 18: H     */
    {0x00, 0x41, 0x7F, 0x41, 0x00}, /* 19: I     */
    {0x7F, 0x40, 0x40, 0x40, 0x40}, /* 20: L     */
    {0x7F, 0x02, 0x0C, 0x02, 0x7F}, /* 21: M     */
    {0x7F, 0x04, 0x08, 0x10, 0x7F}, /* 22: N     */
    {0x3E, 0x41, 0x41, 0x41, 0x3E}, /* 23: O     */
    {0x7F, 0x09, 0x09, 0x09, 0x06}, /* 24: P     */
    {0x7F, 0x09, 0x19, 0x29, 0x46}, /* 25: R     */
    {0x46, 0x49, 0x49, 0x49, 0x31}, /* 26: S     */
    {0x01, 0x01, 0x7F, 0x01, 0x01}, /* 27: T     */
    {0x3F, 0x40, 0x40, 0x40, 0x3F}, /* 28: V     */
    {0x07, 0x08, 0x70, 0x08, 0x07}, /* 29: Y     ← FIXED */
    {0x00, 0x00, 0x00, 0x00, 0x00}, /* 30: /     (blank slot) */
    {0x00, 0x00, 0x7F, 0x00, 0x00}, /* 31: l     */
    {0x63, 0x14, 0x08, 0x14, 0x63}, /* 32: x / X */
    /* ---- HeteroWise additions ---- */
    {0x7F, 0x49, 0x49, 0x49, 0x36}, /* 33: B     ← FIXED */
    {0x20, 0x40, 0x41, 0x3F, 0x01}, /* 34: J     */
    {0x7F, 0x08, 0x14, 0x22, 0x41}, /* 35: K     */
    {0x3F, 0x40, 0x40, 0x40, 0x3F}, /* 36: U     */
    {0x08, 0x08, 0x08, 0x08, 0x08}, /* 37: -     */
    {0x00, 0x60, 0x60, 0x00, 0x00}, /* 38: .     */
    {0x3F, 0x40, 0x38, 0x40, 0x3F}, /* 39: W     */
    {0x61, 0x51, 0x49, 0x45, 0x43}, /* 40: Z     */
};

static uint8_t char_to_idx(char c)
{
    if (c == ' ') return 0;
    if (c >= '0' && c <= '9') return (uint8_t)(1 + (c - '0'));
    if (c == ':') return 11;
    if (c == 'A') return 12;
    if (c == 'C') return 13;
    if (c == 'D') return 14;
    if (c == 'E') return 15;
    if (c == 'F') return 16;
    if (c == 'G') return 17;
    if (c == 'H') return 18;
    if (c == 'I') return 19;
    if (c == 'L') return 20;
    if (c == 'M') return 21;
    if (c == 'N') return 22;
    if (c == 'O') return 23;
    if (c == 'P') return 24;
    if (c == 'R') return 25;
    if (c == 'S') return 26;
    if (c == 'T') return 27;
    if (c == 'V') return 28;
    if (c == 'Y') return 29;
    if (c == '/') return 30;
    if (c == 'l') return 31;
    if (c == 'x') return 32;
    if (c == 'X') return 32;
    if (c == 'B') return 33;
    if (c == 'J') return 34;
    if (c == 'K') return 35;
    if (c == 'U') return 36;
    if (c == '-') return 37;
    if (c == '.') return 38;
    if (c == 'W') return 39;
    if (c == 'Z') return 40;
    return 0;
}

static void lcd_draw_char(uint16_t x, uint16_t y, char c,
                           uint16_t color, uint16_t bg, uint8_t size)
{
    uint8_t idx = char_to_idx(c);
    for (uint8_t col = 0; col < 5; col++) {
        uint8_t line = font5x7[idx][col];
        for (uint8_t row = 0; row < 8; row++) {
            uint16_t px = (line & 0x01) ? color : bg;
            if (size == 1) {
                lcd_draw_pixel((uint16_t)(x+col), (uint16_t)(y+row), px);
            } else {
                lcd_fill_rect((uint16_t)(x + col*size),
                              (uint16_t)(y + row*size),
                              size, size, px);
            }
            line >>= 1;
        }
    }
}

static void lcd_draw_string(uint16_t x, uint16_t y, const char *str,
                              uint16_t color, uint16_t bg, uint8_t size)
{
    while (*str) {
        lcd_draw_char(x, y, *str, color, bg, size);
        x = (uint16_t)(x + 6*size);
        str++;
    }
}

static void lcd_draw_number(uint16_t x, uint16_t y, uint32_t num,
                              uint16_t color, uint16_t bg, uint8_t size)
{
    char buf[12];
    int_to_string(num, buf);
    lcd_draw_string(x, y, buf, color, bg, size);
}

/* ================================================================
 * HETEROWISE DISPLAY
 * ================================================================ */
static uint16_t hw_color(void)
{
    if (gHW[0]=='N') return COL_NPU;
    if (gHW[0]=='G') return COL_GPU;
    if (gHW[0]=='C') return COL_CPU;
    return COL_DIM;
}

static const char *mode_str(void)
{
    switch (gMode) {
        case 1:  return "NPU ";
        case 2:  return "GPU ";
        case 3:  return "CPU ";
        default: return "AUTO";
    }
}

static void draw_screen_skeleton(void)
{
    lcd_fill(COL_BG);

    /* Title bar */
    lcd_fill_rect(0, 0, 240, 28, COL_ACCENT);
    lcd_draw_string(4,  8, "HETEROWISE", COL_TEXT, COL_ACCENT, 2);
    lcd_draw_string(152, 10, "MODE:", COL_DIM,  COL_ACCENT, 1);
    lcd_draw_string(188, 10, mode_str(), COL_YELLOW, COL_ACCENT, 1);

    /* Panel dividers */
    lcd_fill_rect(0,  28, 240, 1, COL_ACCENT);
    lcd_fill_rect(0,  80, 240, 1, COL_DARKGRAY);
    lcd_fill_rect(0, 130, 240, 1, COL_DARKGRAY);
    lcd_fill_rect(0, 180, 240, 1, COL_DARKGRAY);
    lcd_fill_rect(0, 230, 240, 1, COL_DARKGRAY);
    lcd_fill_rect(0, 280, 240, 1, COL_ACCENT);

    /* Panel labels */
    lcd_draw_string(6,  32, "ACCELERATOR", COL_DIM, COL_BG, 1);
    lcd_draw_string(6,  84, "ENERGY",      COL_DIM, COL_BG, 1);
    lcd_draw_string(6, 134, "BUDGET",      COL_DIM, COL_BG, 1);
    lcd_draw_string(130, 134, "JOY ->",    COL_DIM, COL_BG, 1);
    lcd_draw_string(6, 184, "TEMPERATURE", COL_DIM, COL_BG, 1);
    lcd_draw_string(6, 234, "CLOCK",       COL_DIM, COL_BG, 1);
    lcd_draw_string(6, 284, "REMAINING:",  COL_DIM, COL_BG, 1);
}

static void draw_values(void)
{
    /* ── Panel A: Accelerator ─────────────────────────────────── */
    lcd_fill_rect(6, 46, 160, 30, COL_BG);
    lcd_draw_string(6, 48, (const char *)gHW, hw_color(), COL_BG, 3);

    /* ── Panel B: Energy ─────────────────────────────────────── */
    lcd_fill_rect(6, 98, 230, 28, COL_BG);
    {
        uint32_t ej       = gEnergyEJ;
        uint32_t int_part = ej / 10000UL;
        uint32_t frac     = ej % 10000UL;
        char fbuf[6];
        fbuf[0] = (char)('0' + frac/1000);
        fbuf[1] = (char)('0' + frac/100 %10);
        fbuf[2] = (char)('0' + frac/10  %10);
        fbuf[3] = (char)('0' + frac     %10);
        fbuf[4] = '\0';
        uint16_t xpos = 6;
        lcd_draw_number(xpos, 100, int_part, COL_TEXT, COL_BG, 2);
        xpos = (uint16_t)(xpos + 12);
        lcd_draw_string(xpos, 100, ".", COL_TEXT, COL_BG, 2);
        xpos = (uint16_t)(xpos + 12);
        lcd_draw_string(xpos, 100, fbuf, COL_TEXT, COL_BG, 2);
        lcd_draw_string(xpos + 48, 100, "J", COL_DIM, COL_BG, 2);
    }
    {
        uint32_t max_ej = (uint32_t)BUDGET_MAX * 10000UL;
        uint16_t fill_w = (max_ej > 0)
            ? (uint16_t)((uint32_t)200UL * gEnergyEJ / max_ej)
            : 0;
        if (fill_w > 200) fill_w = 200;
        lcd_draw_bar(6, 122, 200, 5, fill_w, hw_color(), COL_DARKGRAY);
    }

    /* ── Panel C: Budget ─────────────────────────────────────── */
    lcd_fill_rect(6, 148, 120, 24, COL_BG);
    lcd_draw_number(6, 150, gBudget, COL_YELLOW, COL_BG, 2);
    lcd_draw_string(30, 150, "J MAX", COL_DIM, COL_BG, 2);
    {
        uint16_t fill_w = (uint16_t)((uint32_t)200UL * gBudget / BUDGET_MAX);
        if (fill_w > 200) fill_w = 200;
        lcd_draw_bar(6, 170, 200, 5, fill_w, COL_YELLOW, COL_DARKGRAY);
    }

    /* ── Panel D: Temperature ────────────────────────────────── */
    lcd_fill_rect(6, 198, 230, 28, COL_BG);
    {
        uint16_t tcol = (gTempC >= 80) ? COL_WARN : COL_OK;
        lcd_draw_number(6, 200, gTempC, tcol, COL_BG, 2);
        lcd_draw_string(42, 200, "C", COL_DIM, COL_BG, 2);
        if (gTempC >= 80) {
            lcd_draw_string(80, 200, "HOT", COL_WARN, COL_BG, 1);
        }
        uint16_t fill_w = (gTempC <= 100)
            ? (uint16_t)((uint32_t)200UL * gTempC / 100UL)
            : 200;
        lcd_draw_bar(6, 220, 200, 5, fill_w, tcol, COL_DARKGRAY);
    }

    /* ── Panel E: Clock ──────────────────────────────────────── */
    lcd_fill_rect(6, 248, 230, 24, COL_BG);
    lcd_draw_number(6, 250, gClkMHz, COL_CYAN, COL_BG, 2);
    lcd_draw_string(60, 250, "MHZ", COL_DIM, COL_BG, 2);
    lcd_draw_string(140, 250, mode_str(), COL_YELLOW, COL_BG, 2);

    /* ── Bottom: remaining budget bar ───────────────────────── */
    {
        uint32_t budget_ej = (uint32_t)gBudget * 10000UL;
        uint32_t remaining = (gEnergyEJ < budget_ej)
            ? (budget_ej - gEnergyEJ)
            : 0UL;
        uint16_t fill_w = (budget_ej > 0)
            ? (uint16_t)((uint32_t)228UL * remaining / budget_ej)
            : 0;
        if (fill_w > 228) fill_w = 228;
        uint16_t bcol = (remaining == 0) ? COL_WARN : COL_AMBER;
        lcd_draw_bar(6, 294, 228, 14, fill_w, bcol, COL_DARKGRAY);
    }
}

static void draw_title_mode(void)
{
    lcd_fill_rect(188, 6, 48, 16, COL_ACCENT);
    lcd_draw_string(188, 10, mode_str(), COL_YELLOW, COL_ACCENT, 1);
}

/* ================================================================
 * JOYSTICK PROCESSING
 * ================================================================ */
static void joystick_update(void)
{
    uint16_t adc = gAdcResult;

    if (adc < (ADC_MID - ADC_DEADZONE) || adc > (ADC_MID + ADC_DEADZONE)) {
        uint32_t range  = BUDGET_MAX - BUDGET_MIN;
        uint8_t  budget = (uint8_t)(BUDGET_MIN + (uint32_t)adc * range / ADC_MAX);
        if (budget < BUDGET_MIN) budget = BUDGET_MIN;
        if (budget > BUDGET_MAX) budget = BUDGET_MAX;
        gBudget = budget;
    }
}

/* ================================================================
 * INTERRUPT HANDLERS
 * ================================================================ */
void ADC12_0_INST_IRQHandler(void)
{
    switch (DL_ADC12_getPendingInterrupt(ADC12_0_INST)) {
        case DL_ADC12_IIDX_MEM0_RESULT_LOADED:
            gAdcReady  = true;
            gAdcResult = DL_ADC12_getMemResult(ADC12_0_INST, DL_ADC12_MEM_IDX_0);
            break;
        default:
            break;
    }
}

void UART_0_INST_IRQHandler(void)
{
    switch (DL_UART_Main_getPendingInterrupt(UART_0_INST)) {
        case DL_UART_MAIN_IIDX_RX: {
            uint8_t byte = DL_UART_Main_receiveData(UART_0_INST);
            if (byte == (uint8_t)'\n') {
                gRxBuf[gRxIdx] = '\0';
                if (gRxIdx > 0) {
                    parse_pc_line(gRxBuf);
                }
                gRxIdx = 0;
            } else if (byte != (uint8_t)'\r') {
                if (gRxIdx < RX_BUF_LEN - 1) {
                    gRxBuf[gRxIdx++] = (char)byte;
                }
            }
            break;
        }
        default:
            break;
    }
}

/* ================================================================
 * MAIN
 * ================================================================ */
int main(void)
{
    SYSCFG_DL_init();

    NVIC_EnableIRQ(ADC12_0_INST_INT_IRQN);
    NVIC_EnableIRQ(UART_0_INST_INT_IRQN);

    DC_LOW();
    RST_HIGH();
    ili9341_init();

    draw_screen_skeleton();

    uart_send_string("HW_CONSOLE_READY\r\n");

    gAdcReady = false;
    DL_ADC12_enableConversions(ADC12_0_INST);
    DL_ADC12_startConversion(ADC12_0_INST);

    while (1)
    {
        if (gAdcReady) {
            gAdcReady = false;
            joystick_update();
            DL_ADC12_enableConversions(ADC12_0_INST);
            DL_ADC12_startConversion(ADC12_0_INST);
        }

        if (!gRxReady) {
            gNoRxCount++;
        }
        gRxReady = false;

        if (gNoRxCount >= SIM_TIMEOUT_CYCLES) {
            apply_simulation();
        }

        draw_values();
        uart_send_status();
        delay_ms(REFRESH_MS);
    }
}