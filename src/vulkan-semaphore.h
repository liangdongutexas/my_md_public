#pragma once
#include <vulkan/vulkan.h>
#include "vulkan-logical-device.h"

namespace vks
{
    class VulkanSemaphore
    {
    private:
        VulkanLogicalDevice* logicalDevice = VulkanLogicalDevice::getVulkanLogicalDevice();
        bool isExternal = false;
        bool isInternal = false;
    public:
        VkSemaphore semaphore{};

        void createSemaphore(bool signal = false);

        static VkExternalSemaphoreHandleTypeFlagBits getDefaultSemaphoreHandleType();

        void* createExternalSemaphore();

        void* getSemaphoreHandle();

        void destroy();
    };
}