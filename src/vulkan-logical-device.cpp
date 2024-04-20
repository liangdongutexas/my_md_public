#include "vulkan-logical-device.h"

namespace vks
{
    VulkanLogicalDevice* VulkanLogicalDevice::device_T = nullptr;

    VulkanLogicalDevice::VulkanLogicalDevice(){};


    void VulkanCommandBuffer::destroy(){
        if (commandBuffer != VK_NULL_HANDLE){
            vkFreeCommandBuffers(logicalDevice->device, commandPool, 1, &commandBuffer);
        }
        else{
            DEBUG_CERR<<"trying to free a command buffer that is null"<<std::endl;
        }

    }

        //function defines
    void VulkanLogicalDevice::destroy()
    {
        if (device != VK_NULL_HANDLE){
            // Ensure all operations on the logicalDevice.device have been finished before destroying
            vkDeviceWaitIdle(device);

            // Destroy command pools
            if (graphicsCommandPool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device, graphicsCommandPool, nullptr);
            }
            if (transferCommandPool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device, transferCommandPool, nullptr);
            }
            if (computeCommandPool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device, computeCommandPool, nullptr);
            }

            // Resetting optional indices is not strictly necessary as they do not allocate resources,
            // but it's good practice to clear or reset class state during destruction.
            graphicsFamilyIndex.reset();
            transferFamilyIndex.reset();
            computeFamilyIndex.reset();
            presentFamilyIndex.reset();

            // Destroying the logical logicalDevice.device also implicitly frees all queues, so explicit queue destruction is not needed.
            vkDestroyDevice(device, nullptr);
            device = VK_NULL_HANDLE;
            delete device_T;
        }
    };

    /**
     * @brief typedef struct VkCommandPoolCreateInfo {
                VkStructureType             sType;
                const void*                 pNext;
                VkCommandPoolCreateFlags    flags;
                uint32_t                    queueFamilyIndex;
            } VkCommandPoolCreateInfo;
     *
     */
    void VulkanLogicalDevice::createQueueCommandPool()
    {
        if (device!=VK_NULL_HANDLE){
            //get all the queues and create the corresponding command pool
            std::vector<std::optional<uint32_t>> familyIndices =  {graphicsFamilyIndex, computeFamilyIndex, transferFamilyIndex};
            std::vector<VkQueue*>                 queueus      =  {&graphicsQueue, &computeQueue, &transferQueue};
            std::vector<VkCommandPool*> 		 commandPools  =  {&graphicsCommandPool, &computeCommandPool, &transferCommandPool};

            // create command pool
            for (uint32_t i=0; i<familyIndices.size(); ++i){
                if (familyIndices[i].has_value()){
                    vkGetDeviceQueue(device, familyIndices[i].value(), 0, queueus[i]);

                    VkCommandPoolCreateInfo cmdPoolInfo = {};
                    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
                    cmdPoolInfo.queueFamilyIndex = familyIndices[i].value();
                    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
                    VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, commandPools[i]));
                }
            }

            vkGetDeviceQueue(device, presentFamilyIndex.value(), 0, &presentQueue);

            DEBUG_COUT<<graphicsFamilyIndex.value()<<computeFamilyIndex.value()<<transferFamilyIndex.value()<<presentFamilyIndex.value()<<std::endl;
        }
    };

    vks::VulkanCommandBuffer VulkanLogicalDevice::getCommandBuffer(VkQueueFlagBits queueFlag, VkCommandBufferLevel level)
    {
        //command buffer wrapper to be returned
		VulkanCommandBuffer vulkanCommandBuffer;

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

		if ((VK_QUEUE_TRANSFER_BIT & queueFlag) == VK_QUEUE_TRANSFER_BIT){
			allocInfo.commandPool = transferCommandPool;
			vulkanCommandBuffer.commandPool = transferCommandPool;
			vulkanCommandBuffer.queue = transferQueue;
		}
		else if ((VK_QUEUE_COMPUTE_BIT & queueFlag) == VK_QUEUE_COMPUTE_BIT){
			allocInfo.commandPool = computeCommandPool;
			vulkanCommandBuffer.commandPool = computeCommandPool;
			vulkanCommandBuffer.queue = computeQueue;
		}
		else if ((VK_QUEUE_GRAPHICS_BIT & queueFlag) == VK_QUEUE_GRAPHICS_BIT){
			allocInfo.commandPool = graphicsCommandPool;
			vulkanCommandBuffer.commandPool = graphicsCommandPool;
			vulkanCommandBuffer.queue = graphicsQueue;
		}
		else {throw std::runtime_error("no command pool supports the requested single time command functionality");}

        allocInfo.commandBufferCount = 1;

        VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &allocInfo, &vulkanCommandBuffer.commandBuffer));

        return vulkanCommandBuffer;
    };

    /**
     * @brief
     *
     * @param queueFlag QueueFlag inputed by the user to indicate what kinds of one time buffer the user intended to use
     * @return VkCommandBuffer
     */
    VulkanCommandBuffer VulkanLogicalDevice::beginSingleTimeCommands(VkQueueFlagBits queueFlag)
    {

        VulkanCommandBuffer commandBuffer = getCommandBuffer(queueFlag);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer.commandBuffer, &beginInfo));

        return commandBuffer;
    }


    void VulkanLogicalDevice::endSingleTimeCommands(vks::VulkanCommandBuffer& vulkanCommandBuffer) {
        vkEndCommandBuffer(vulkanCommandBuffer.commandBuffer);

        VkSubmitInfo submitInfo = vks::initializers::submitInfo();
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &vulkanCommandBuffer.commandBuffer;
        // Create fence to ensure that the command buffer has finished executing
        VkFenceCreateInfo fenceInfo = vks::initializers::fenceCreateInfo(VK_FLAGS_NONE);
        VkFence fence;
        VK_CHECK_RESULT(vkCreateFence(device, &fenceInfo, nullptr, &fence));
        // Submit to the queue
        VK_CHECK_RESULT(vkQueueSubmit(vulkanCommandBuffer.queue, 1, &submitInfo, fence));
        // Wait for the fence to signal that command buffer has finished executing
        VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT));
        vkDestroyFence(device, fence, nullptr);

        vulkanCommandBuffer.destroy();
    }
}