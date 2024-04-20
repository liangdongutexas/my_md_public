#include "vulkan-semaphore.h"

namespace vks
{
    void VulkanSemaphore::destroy()
    {
        if (semaphore != VK_NULL_HANDLE)
        {
            vkDestroySemaphore(logicalDevice->device, semaphore, nullptr);
        }
    };

    void VulkanSemaphore::createSemaphore(bool signal)
    {
        VkSemaphoreCreateInfo semaphoreCreateInfo {};
		semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if (semaphore == VK_NULL_HANDLE){
		    VK_CHECK_RESULT(vkCreateSemaphore(logicalDevice->device, &semaphoreCreateInfo, nullptr, &semaphore));
        }
        else{
            DEBUG_COUT<<"please destroy semaphore before recreate"<<std::endl;
        }

        if (signal)
        {
            // Signal the semaphore
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &semaphore;
            VK_CHECK_RESULT(vkQueueSubmit(logicalDevice->graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE));
            VK_CHECK_RESULT(vkQueueWaitIdle(logicalDevice->graphicsQueue));
        }

    };

    VkExternalSemaphoreHandleTypeFlagBits VulkanSemaphore::getDefaultSemaphoreHandleType()
    {
    #ifdef _WIN64
        return IsWindows8OrGreater()
                    ? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
                    : VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
    #else
        return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    #endif /* _WIN64 */
    }

    void* VulkanSemaphore::createExternalSemaphore()
    {
        VkSemaphoreCreateInfo semaphoreInfo = {};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkExportSemaphoreCreateInfoKHR exportSemaphoreCreateInfo = {};
        exportSemaphoreCreateInfo.sType =
            VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;

    #ifdef _VK_TIMELINE_SEMAPHORE
        VkSemaphoreTypeCreateInfo timelineCreateInfo;
        timelineCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        timelineCreateInfo.pNext = NULL;
        timelineCreateInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        timelineCreateInfo.initialValue = 0;
        exportSemaphoreCreateInfo.pNext = &timelineCreateInfo;
    #else
        exportSemaphoreCreateInfo.pNext = NULL;
    #endif /* _VK_TIMELINE_SEMAPHORE */
        exportSemaphoreCreateInfo.handleTypes = getDefaultSemaphoreHandleType();
        semaphoreInfo.pNext = &exportSemaphoreCreateInfo;

        if (semaphore == VK_NULL_HANDLE){
		    VK_CHECK_RESULT(vkCreateSemaphore(logicalDevice->device, &semaphoreInfo, nullptr, &semaphore));
        }
        else{
            DEBUG_COUT<<"please destroy semaphore before recreate"<<std::endl;
        }

        return getSemaphoreHandle();
    }


    void* VulkanSemaphore::getSemaphoreHandle()
    {
    #ifdef _WIN64
        HANDLE handle;

        VkSemaphoreGetWin32HandleInfoKHR semaphoreGetWin32HandleInfoKHR = {};
        semaphoreGetWin32HandleInfoKHR.sType =
            VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
        semaphoreGetWin32HandleInfoKHR.pNext = NULL;
        semaphoreGetWin32HandleInfoKHR.semaphore = semaphore;
        semaphoreGetWin32HandleInfoKHR.handleType = handleType;

        PFN_vkGetSemaphoreWin32HandleKHR fpGetSemaphoreWin32HandleKHR;
        fpGetSemaphoreWin32HandleKHR =
            (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(
                m_device, "vkGetSemaphoreWin32HandleKHR");
        if (!fpGetSemaphoreWin32HandleKHR) {
            throw std::runtime_error("Failed to retrieve vkGetMemoryWin32HandleKHR!");
        }
        if (fpGetSemaphoreWin32HandleKHR(m_device, &semaphoreGetWin32HandleInfoKHR,
                                        &handle) != VK_SUCCESS) {
            throw std::runtime_error("Failed to retrieve handle for buffer!");
        }

        return (void *)handle;
    #else
        int fd;

        VkSemaphoreGetFdInfoKHR semaphoreGetFdInfoKHR = {};
        semaphoreGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
        semaphoreGetFdInfoKHR.pNext = NULL;
        semaphoreGetFdInfoKHR.semaphore = semaphore;
        semaphoreGetFdInfoKHR.handleType = getDefaultSemaphoreHandleType();

        PFN_vkGetSemaphoreFdKHR fpGetSemaphoreFdKHR;
        fpGetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(logicalDevice->device, "vkGetSemaphoreFdKHR");
        if (!fpGetSemaphoreFdKHR) {
            DEBUG_CERR<<"Failed to retrieve vkGetMemoryWin32HandleKHR!"<<std::endl;
        }
        if (fpGetSemaphoreFdKHR(logicalDevice->device, &semaphoreGetFdInfoKHR, &fd) != VK_SUCCESS) {
            throw std::runtime_error("Failed to retrieve handle for buffer!");
        }

        return (void *)(uintptr_t)fd;
    #endif /* _WIN64 */
    }


}