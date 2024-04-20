#pragma once

#include <vulkan/vulkan.h>
#include <optional>

#include "vulkan-tools.h"
#include "defines-settings.h"



namespace vks
{
    class VulkanCommandBuffer;

    class VulkanLogicalDevice{
    private:
        // Private Static Instance of the class
        static VulkanLogicalDevice* device_T;

        VulkanLogicalDevice();
    public:
        // Public method to access the instance of the class
        inline static VulkanLogicalDevice* getVulkanLogicalDevice() {
            if (!device_T) {
                device_T = new VulkanLogicalDevice();
            }
            return device_T;
        }

        VkDevice device;
        //all the queue families that will logical device possess
        std::optional<uint32_t> graphicsFamilyIndex;
        VkQueue graphicsQueue;
        VkCommandPool graphicsCommandPool;
        std::optional<uint32_t> transferFamilyIndex;
        VkQueue transferQueue;
        VkCommandPool transferCommandPool;
        std::optional<uint32_t> computeFamilyIndex;
        VkQueue computeQueue;
        VkCommandPool computeCommandPool;
        std::optional<uint32_t> presentFamilyIndex;
        VkQueue presentQueue;


        void createQueueCommandPool();
        // Preventing the copying of singleton objects
        VulkanLogicalDevice(const VulkanLogicalDevice&) = delete;
        VulkanLogicalDevice& operator=(const VulkanLogicalDevice&) = delete;

        VulkanCommandBuffer getCommandBuffer(VkQueueFlagBits queueFlag, VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);
        VulkanCommandBuffer beginSingleTimeCommands(VkQueueFlagBits queueFlag);

        void endSingleTimeCommands(VulkanCommandBuffer& vulkanCommandBuffer);

        void destroy();
    };


    //Command Buffer wrapper for single time command
	class VulkanCommandBuffer{
    private:
        VulkanLogicalDevice* logicalDevice = VulkanLogicalDevice::getVulkanLogicalDevice();
	public:
		uint32_t queueFamilyIndex;
		VkQueue queue;
		VkCommandPool commandPool;

		VkCommandBuffer commandBuffer;				// Command buffer storing the dispatch commands and barriers

        inline void reset(){
            if (commandBuffer != VK_NULL_HANDLE){
                VK_CHECK_RESULT(vkResetCommandBuffer(commandBuffer, 0));
            }
            else{
                std::cerr<<"cannot reset a command buffer which is NULL"<<std::endl;
            }

        }
        void destroy();
	};
}