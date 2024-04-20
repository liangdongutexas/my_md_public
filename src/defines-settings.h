#pragma once

#define _DIRECT_DISPLAY
#define _VULKAN_COMPUTE


#include <vulkan/vulkan.h>
#include <string>
#include <vector>

#include "vulkan-tools.h"

#define VULKAN_APP_MAIN()                           \
VulkanBase *vulkanBase;								\
int main(const int argc, const char *argv[])		\
{													\
	vulkanBase = new App;                           \
    vulkanBase-> init();                            \
    vulkanBase-> prepare();					        \
	vulkanBase->renderLoop();					    \
	delete(vulkanBase);								\
	return 0;										\
}

#define DESTROY_VULKAN_OBJECT(destroyFunction, device, object)                  \
    if (object != VK_NULL_HANDLE) {                                             \
        VK_CHECK_RESULT(destroyFunction(device, object, nullptr));              \
        object = VK_NULL_HANDLE;                                                \
    }

#ifdef DEBUG
#define DEBUG_COUT std::cout
#define DEBUG_CERR std::cerr
#else
#define DEBUG_COUT 0 && std::cout
#define DEBUG_CERR 0 && std::cerr
#endif
