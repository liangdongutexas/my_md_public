#include "xcbUI.h"


XcbUI* XcbUI::xcbUI_T = nullptr;
XcbUI::XcbUI(){};

static inline xcb_intern_atom_reply_t* intern_atom_helper(xcb_connection_t *conn, bool only_if_exists, const char *str)
{
	xcb_intern_atom_cookie_t cookie = xcb_intern_atom(conn, only_if_exists, strlen(str), str);
	return xcb_intern_atom_reply(conn, cookie, NULL);
}

void XcbUI::initxcbConnection()
{
	const xcb_setup_t *setup;
	xcb_screen_iterator_t iter;
	int scr;

	// xcb_connect always returns a non-NULL pointer to a xcb_connection_t,
	// even on failure. Callers need to use xcb_connection_has_error() to
	// check for failure. When finished, use xcb_disconnect() to close the
	// connection and free the structure.
	connection = xcb_connect(NULL, &scr);
	assert( connection );
	if( xcb_connection_has_error(connection) ) {
		printf("Could not find a compatible Vulkan ICD!\n");
		fflush(stdout);
		exit(1);
	}

	setup = xcb_get_setup(connection);
	iter = xcb_setup_roots_iterator(setup);
	while (scr-- > 0)
		xcb_screen_next(&iter);
	screen = iter.data;
}


// Set up a window using XCB and request event types
xcb_window_t XcbUI::setupWindow(std::string windowTitle)
{
	uint32_t value_mask, value_list[32];

	window = xcb_generate_id(connection);

	value_mask = XCB_CW_BACK_PIXEL | XCB_CW_EVENT_MASK;
	value_list[0] = screen->black_pixel;
	value_list[1] =
		XCB_EVENT_MASK_KEY_RELEASE |
		XCB_EVENT_MASK_KEY_PRESS |
		XCB_EVENT_MASK_EXPOSURE |
		XCB_EVENT_MASK_STRUCTURE_NOTIFY |
		XCB_EVENT_MASK_POINTER_MOTION |
		XCB_EVENT_MASK_BUTTON_PRESS |
		XCB_EVENT_MASK_BUTTON_RELEASE;

	if (fullscreen)
	{
		width  = screen->width_in_pixels;
		height = screen->height_in_pixels;
	}

	xcb_create_window(connection,
		XCB_COPY_FROM_PARENT,
		window, screen->root,
		0, 0, width, height, 0,
		XCB_WINDOW_CLASS_INPUT_OUTPUT,
		screen->root_visual,
		value_mask, value_list);


	/* Magic code that will send notification when window is destroyed */
	xcb_intern_atom_reply_t* reply = intern_atom_helper(connection, true, "WM_PROTOCOLS");
	atom_wm_delete_window = intern_atom_helper(connection, false, "WM_DELETE_WINDOW");

	xcb_change_property(connection, XCB_PROP_MODE_REPLACE,
		window, (*reply).atom, 4, 32, 1,
		&(*atom_wm_delete_window).atom);

	xcb_change_property(connection, XCB_PROP_MODE_REPLACE,
	window, XCB_ATOM_WM_NAME, XCB_ATOM_STRING, 8,
	windowTitle.size(), windowTitle.c_str());

	free(reply);

	/**
	 * Set the WM_CLASS property to display
	 * title in dash tooltip and application menu
	 * on GNOME and other desktop environments
	 */
	std::string wm_class;
	wm_class = wm_class.insert(0, name);
	wm_class = wm_class.insert(name.size(), 1, '\0');
	wm_class = wm_class.insert(name.size() + 1, title);
	wm_class = wm_class.insert(wm_class.size(), 1, '\0');
	xcb_change_property(connection, XCB_PROP_MODE_REPLACE, window, XCB_ATOM_WM_CLASS, XCB_ATOM_STRING, 8, wm_class.size() + 2, wm_class.c_str());

	if (fullscreen)
	{
		xcb_intern_atom_reply_t *atom_wm_state = intern_atom_helper(connection, false, "_NET_WM_STATE");
		xcb_intern_atom_reply_t *atom_wm_fullscreen = intern_atom_helper(connection, false, "_NET_WM_STATE_FULLSCREEN");
		xcb_change_property(connection,
				XCB_PROP_MODE_REPLACE,
				window, atom_wm_state->atom,
				XCB_ATOM_ATOM, 32, 1,
				&(atom_wm_fullscreen->atom));
		free(atom_wm_fullscreen);
		free(atom_wm_state);
	}

	xcb_map_window(connection, window);
	flush();

	return(window);
}

void XcbUI::getXcbWindowSize(uint32_t *width, uint32_t *height) {
    xcb_get_geometry_cookie_t geom_cookie = xcb_get_geometry(connection, window);
    xcb_get_geometry_reply_t *geom_reply = xcb_get_geometry_reply(connection, geom_cookie, NULL);

    if (geom_reply) {
        *width = geom_reply->width;
        *height = geom_reply->height;
        free(geom_reply);
    } else {
        *width = 0;
        *height = 0;
        fprintf(stderr, "Could not get window geometry\n");
    }
}


void XcbUI::updateTitle(std::string windowTitle)
{
	xcb_change_property(connection, XCB_PROP_MODE_REPLACE,
		window, XCB_ATOM_WM_NAME, XCB_ATOM_STRING, 8,
		static_cast<uint32_t>(windowTitle.size()), windowTitle.c_str());
}


KeyMouseEvent XcbUI::getEvent()
{
    std::vector<KeyMouseEvent> events;
    xcb_generic_event_t *event;

    while ((event = xcb_poll_for_event(connection)))
    {
        KeyMouseEvent mappedEvent;
        mapEvent(event, mappedEvent);
        events.push_back(mappedEvent);
        free(event);
    }
	if (events.size()>0){
		return events[0];
	}
}

void XcbUI::mapEvent(const xcb_generic_event_t* event, KeyMouseEvent& mappedEvent)
{
	switch (event->response_type & 0x7f)
	{
	case XCB_CLIENT_MESSAGE:
		if ((*(xcb_client_message_event_t*)event).data.data32[0] ==
			(*atom_wm_delete_window).atom) {
			quit = true;
		}
		break;
	case XCB_MOTION_NOTIFY:
	{
		xcb_motion_notify_event_t *motion = (xcb_motion_notify_event_t *)event;

		mouse.move_x=(int32_t)motion->event_x;
		mouse.move_y=(int32_t)motion->event_y;
		break;
	}
	break;
	case XCB_BUTTON_PRESS:
	{
		xcb_button_press_event_t *press = (xcb_button_press_event_t *)event;
		if (press->detail == XCB_BUTTON_INDEX_1)
			mouse.left_pressed = true;
			mappedEvent = MOUSE_LEFT_BUTTON_PRESSED;
		if (press->detail == XCB_BUTTON_INDEX_2)
			mouse.middle_pressed = true;
			mappedEvent = MOUSE_MIDDLE_BUTTON_PRESSED;
		if (press->detail == XCB_BUTTON_INDEX_3)
			mouse.right_pressed = true;
			mappedEvent = MOUSE_RIGHT_BUTTON_PRESSED;
	}
	break;
	case XCB_BUTTON_RELEASE:
	{
		xcb_button_press_event_t *press = (xcb_button_press_event_t *)event;
		if (press->detail == XCB_BUTTON_INDEX_1)
			mouse.left_pressed = false;
			mappedEvent = MOUSE_LEFT_BUTTON_RELEASED;
		if (press->detail == XCB_BUTTON_INDEX_2)
			mouse.middle_pressed = false;
			mappedEvent = MOUSE_MIDDLE_BUTTON_RELEASED;
		if (press->detail == XCB_BUTTON_INDEX_3)
			mouse.right_pressed = false;
			mappedEvent = MOUSE_RIGHT_BUTTON_RELEASED;
	}
	break;
	case XCB_KEY_PRESS:
	{
		const xcb_key_press_event_t *keyEvent = (const xcb_key_press_event_t *)event;
		switch (keyEvent->detail)
		{
			case KEY_W:
				keyboard.key_w_pressed = true;
				mappedEvent = KEY_W_PRESSED;
				break;
			case KEY_S:
				keyboard.key_s_pressed = true;
				mappedEvent = KEY_S_PRESSED;
				break;
			case KEY_A:
				keyboard.key_a_pressed = true;
				mappedEvent = KEY_A_PRESSED;
				break;
			case KEY_D:
				keyboard.key_d_pressed = true;
				mappedEvent = KEY_D_PRESSED;
				break;
			case KEY_P:
				keyboard.key_p_pressed = true;
				mappedEvent = KEY_P_PRESSED;
				break;
			case KEY_F1:
				keyboard.key_f1_pressed = true;
				mappedEvent = KEY_F1_PRESSED;
				break;
			case KEY_F2:
				keyboard.key_f2_pressed = true;
				mappedEvent = KEY_F2_PRESSED;
				break;
		}
	}
	break;
	case XCB_KEY_RELEASE:
	{
		const xcb_key_release_event_t *keyEvent = (const xcb_key_release_event_t *)event;
		switch (keyEvent->detail)
		{
			case KEY_W:
				keyboard.key_w_pressed = false;
				mappedEvent = KEY_W_RELEASED;
				break;
			case KEY_S:
				keyboard.key_s_pressed = false;
				mappedEvent = KEY_S_RELEASED;
				break;
			case KEY_A:
				keyboard.key_a_pressed = false;
				mappedEvent = KEY_A_RELEASED;
				break;
			case KEY_D:
				keyboard.key_d_pressed = false;
				mappedEvent = KEY_D_RELEASED;
				break;
			case KEY_F1:
				keyboard.key_f1_pressed = false;
				mappedEvent = KEY_F1_RELEASED;
				break;
			case KEY_F2:
				keyboard.key_f2_pressed = false;
				mappedEvent = KEY_F2_RELEASED;
				break;
			case KEY_ESCAPE:
				quit = true;
				break;

		}
	}
	break;
	case XCB_DESTROY_NOTIFY:
		quit = true;
		break;
	case XCB_CONFIGURE_NOTIFY:
	{
		const xcb_configure_notify_event_t *cfgEvent = (const xcb_configure_notify_event_t *)event;
		if (((cfgEvent->width != width) || (cfgEvent->height != height)))
		{
				width = cfgEvent->width;
				height = cfgEvent->height;
				if ((width > 0) && (height > 0))
				{
					windowResized = true;
				}
		}
	}
	break;
	default:
		break;
	}
}

void XcbUI::destroy()
{
	// Free the atom_wm_delete_window reply
    free(atom_wm_delete_window);

    // Destroy the window
    xcb_destroy_window(connection, window);

    // Disconnect from the XCB connection
    xcb_disconnect(connection);
};


