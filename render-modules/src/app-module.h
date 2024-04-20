#pragma once

#include <vulkan/vulkan.h>
#include <optional>
#include "vulkan-logical-device.h"
#include "vulkan-physical-device.h"
#include "vulkan-buffer.h"
#include "key-codes-status.h"

class AppModule
{
protected:
    vks::VulkanPhysicalDevice* physicalDevice = vks::VulkanPhysicalDevice::getVulkanPhysicalDevice();
    vks::VulkanLogicalDevice* logicalDevice = vks::VulkanLogicalDevice::getVulkanLogicalDevice();
    VkRenderPass renderPass;
    uint32_t subpass;

    // Declare function pointers for the conditional rendering functions
    PFN_vkCmdBeginConditionalRenderingEXT fpVkCmdBeginConditionalRenderingEXT = nullptr;
    PFN_vkCmdEndConditionalRenderingEXT fpVkCmdEndConditionalRenderingEXT = nullptr;
    VkConditionalRenderingBeginInfoEXT conditionalRenderingBeginInfo = vks::initializers::conditionalBeginInfo();

    //whether the app module is visible in the window off screen computing
    vks::Buffer visibilityBuffer;
    uint32_t visibility = 1;

public:
    virtual void init();
    virtual void deinit();
    virtual void prepare();
    virtual void unprepare();

    //whether or not the command buffer associated with the pipeline or vertex buffer need to be recorded
    bool memoryResourcesReady = false;
    bool pipelineReady = false;
    bool needRecording = true;

    void updateRenderPass(VkRenderPass renderPass_E, uint32_t subpass_E);
    void setVisibility(uint32_t vis);

    virtual void updateMemoryResources();
    //include handling window resize, other xcb event or due to dynamical data events
    virtual void handleEvent(KeyMouseEvent KMEvent) = 0;
};