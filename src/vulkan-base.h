#pragma once


#include "defines-settings.h"

#include <chrono>

#include "vulkan-texture.h"
#include "vulkan-physical-device.h"
#include "vulkan-logical-device.h"
#include "vulkan-debug.h"
#include "xcbUI.h"
#include "vulkan-swapchain.h"



class VulkanBase
{
private:
    VkInstance instance;
    //a physical device wrapper
    vks::VulkanPhysicalDevice* physicalDevice = vks::VulkanPhysicalDevice::getVulkanPhysicalDevice();

    //functions utilized by initVulkan
    void createInstanceDebug();
    bool pickPhysicalDevice();
    bool isDeviceSuitable(VkPhysicalDevice device);

    void createLogicalDevice();
    void setupDisplay();

protected:
    bool vsync = false;

    bool requiresStencil{ false };

    std::string AppName;


    //a logical device wrapper
    vks::VulkanLogicalDevice* logicalDevice = vks::VulkanLogicalDevice::getVulkanLogicalDevice();

	// Depth buffer format (selected during initDeviceBasedObjects())
    vks::Texture depthStencil;

    bool basePrepared = false;

    XcbUI* xcbUI = XcbUI::getXcbUI();

    //applications that based on logical device and its queues
    vks::VulkanSwapChain swapchain;
    // List of available frame buffers (same as number of swap chain images)
	std::vector<VkFramebuffer> frameBuffers;

    // Global render pass for frame buffer writes
	VkRenderPass renderPass{ VK_NULL_HANDLE };
    VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();

    //rank preference of combinations of queue Index, 0 is reserved for unacceptable, used by isDeviceSuitable() function
    virtual uint32_t rankIndexCombination(std::optional<uint32_t> graphicsIndex, std::optional<uint32_t> transferIndex, std::optional<uint32_t> computeIndex, std::optional<uint32_t> presentIndex) = 0;
    virtual std::vector<const char*> getInstanceLayers() = 0;
    virtual std::vector<const char*> getInstanceExtensions() = 0;
    virtual std::vector<const char*> getDeviceExtensions() = 0;

    //functions called by prepare()
    void setupSwapChain();
    virtual void setupDepthStencil();



public:
    VkClearColorValue defaultClearColor = { { 0.025f, 0.025f, 0.025f, 1.0f } };
    std::string title = "VulkanApp";


    virtual void init();
    virtual void deinit();
    virtual void prepare();
    virtual void unprepare();
    virtual void renderLoop() = 0;
};


