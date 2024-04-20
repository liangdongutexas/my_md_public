#include "app-module.h"


void AppModule::init()
{
    visibilityBuffer.allocateMemory(VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, sizeof(uint32_t));
    setVisibility(1);

    conditionalRenderingBeginInfo.buffer = visibilityBuffer.buffer;
    fpVkCmdBeginConditionalRenderingEXT = reinterpret_cast<PFN_vkCmdBeginConditionalRenderingEXT>(vkGetDeviceProcAddr(logicalDevice->device, "vkCmdBeginConditionalRenderingEXT"));
    fpVkCmdEndConditionalRenderingEXT = reinterpret_cast<PFN_vkCmdEndConditionalRenderingEXT>(vkGetDeviceProcAddr(logicalDevice->device, "vkCmdEndConditionalRenderingEXT"));
};

void AppModule::prepare(){};
void AppModule::unprepare(){};

 void AppModule::deinit()
 {
    fpVkCmdEndConditionalRenderingEXT = nullptr;
    fpVkCmdBeginConditionalRenderingEXT = nullptr;
    conditionalRenderingBeginInfo = vks::initializers::conditionalBeginInfo();

    visibilityBuffer.destroy();
 };

void AppModule::updateMemoryResources(){};

void AppModule::setVisibility(uint32_t vis)
{
    visibility = vis;
    visibilityBuffer.map();
    uint32_t* mappedRef = (uint32_t*)(visibilityBuffer.mapped);
    *mappedRef = visibility;
    visibilityBuffer.unmap();
};

void AppModule::updateRenderPass(VkRenderPass renderPass_E, uint32_t subpass_E)
{
    renderPass=renderPass_E;
    subpass=subpass_E;
};

