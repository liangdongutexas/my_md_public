/*
* Key codes for multiple platforms
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once


#include <stdint.h>


#if defined(__linux__) || defined(__FreeBSD__)
#define KEY_ESCAPE 0x9
#define KEY_F1 0x43
#define KEY_F2 0x44
#define KEY_F3 0x45
#define KEY_F4 0x46
#define KEY_W 0x19
#define KEY_A 0x26
#define KEY_S 0x27
#define KEY_D 0x28
#define KEY_P 0x21
#define KEY_SPACE 0x41
#define KEY_KPADD 0x56
#define KEY_KPSUB 0x52
#define KEY_B 0x38
#define KEY_F 0x29
#define KEY_L 0x2E
#define KEY_N 0x39
#define KEY_O 0x20
#define KEY_T 0x1C
#endif

struct MouseStatus
{
    bool left_pressed=false;
    bool right_pressed = false;
    bool middle_pressed = false;
    int32_t move_x=0;
    int32_t move_y=0;
};

struct KeyboardStatus
{
    bool key_escape_pressed=false;
    bool key_f1_pressed=false;
    bool key_f2_pressed=false;
    bool key_f3_pressed=false;
    bool key_f4_pressed=false;
    bool key_w_pressed=false;
    bool key_a_pressed=false;
    bool key_s_pressed=false;
    bool key_d_pressed=false;
    bool key_p_pressed=false;
    bool key_space_pressed=false;
    bool key_kpadd_pressed=false;
    bool key_kpsub_pressed=false;
    bool key_b_pressed=false;
    bool key_f_pressed=false;
    bool key_l_pressed=false;
    bool key_n_pressed=false;
    bool key_o_pressed=false;
    bool key_t_pressed=false;
};

enum KeyMouseEvent
{
    KEY_ESCAPE_PRESSED,
    KEY_ESCAPE_RELEASED,
    KEY_F1_PRESSED,
    KEY_F1_RELEASED,
    KEY_F2_PRESSED,
    KEY_F2_RELEASED,
    KEY_F3_PRESSED,
    KEY_F3_RELEASED,
    KEY_F4_PRESSED,
    KEY_F4_RELEASED,
    KEY_W_PRESSED,
    KEY_W_RELEASED,
    KEY_A_PRESSED,
    KEY_A_RELEASED,
    KEY_S_PRESSED,
    KEY_S_RELEASED,
    KEY_D_PRESSED,
    KEY_D_RELEASED,
    KEY_P_PRESSED,
    KEY_P_RELEASED,
    KEY_KPADD_PRESSED,
    KEY_KPADD_RELEASED,
    KEY_KPSUB_PRESSED,
    KEY_KPSUB_RELEASED,
    MOUSE_LEFT_BUTTON_PRESSED,
    MOUSE_LEFT_BUTTON_RELEASED,
    MOUSE_RIGHT_BUTTON_PRESSED,
    MOUSE_RIGHT_BUTTON_RELEASED,
    MOUSE_MIDDLE_BUTTON_PRESSED,
    MOUSE_MIDDLE_BUTTON_RELEASED
};