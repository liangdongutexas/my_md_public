#pragma once

#include <xcb/xcb.h>
#include <vector>
#include <string.h>
#include <string>
#include <cassert>


#include "key-codes-status.h"
#include "frame-bench.hpp"



class XcbUI
{
private:
    static XcbUI* xcbUI_T;

    xcb_screen_t *screen;

    xcb_intern_atom_reply_t* atom_wm_delete_window;

    XcbUI();

public:
    std::string title = "Vulkan Example";
	std::string name = "vulkanExample";
    //width and height of the display
    uint32_t width=1920;
	uint32_t height=1080;

    xcb_connection_t *connection;
    xcb_window_t window;

    FrameBench frameBench;
    MouseStatus mouse;
    KeyboardStatus keyboard;


    /** @brief Set to true if fullscreen mode has been requested via command line */
    bool fullscreen = false;
    bool quit = false;

    bool windowResized = false;


    // Public method to access the instance of the class
    inline static XcbUI* getXcbUI() {
        if (!xcbUI_T) {
            xcbUI_T = new XcbUI();
        }
        return xcbUI_T;
    }

    // Preventing the copying of singleton objects
    XcbUI(const XcbUI&) = delete;
    XcbUI& operator=(const XcbUI&) = delete;

    void initxcbConnection();

    void flush(){xcb_flush(connection);};

    KeyMouseEvent getEvent();
    void mapEvent(const xcb_generic_event_t* event, KeyMouseEvent& mappedEventy);

    xcb_window_t setupWindow(std::string windowTitle);
    void updateTitle (std::string windowTitle);

    void getXcbWindowSize(uint32_t *width, uint32_t *height);

    void destroy();
};


