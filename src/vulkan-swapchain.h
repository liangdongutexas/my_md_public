/*
* Class wrapping access to the swap chain
*
* A swap chain is a collection of framebuffers used for rendering and presentation to the windowing system
*
* Copyright (C) 2016-2023 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once



#include <vulkan/vulkan.h>
#include <xcb/xcb.h>
#include <stdlib.h>
#include <string>
#include <assert.h>
#include <stdio.h>
#include <vector>
#include <optional>
#include <set>



#include "vulkan-tools.h"
#include "vulkan-logical-device.h"
#include "vulkan-physical-device.h"

namespace vks
{
	struct SwapChainDetails {
		VkSurfaceCapabilitiesKHR capabilities = {};
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> presentModes;

		bool isComplete() {return !formats.empty() && !presentModes.empty();};
	};


	class VulkanSwapChain
	{
	private:
		VulkanPhysicalDevice* physicalDevice = VulkanPhysicalDevice::getVulkanPhysicalDevice();
		VulkanLogicalDevice* logicalDevice = VulkanLogicalDevice::getVulkanLogicalDevice();
		SwapChainDetails details;

		void printSwapchainCreateInfo(const VkSwapchainCreateInfoKHR& ci);
		VkResult checkVkPresentInfoKHR(const VkPresentInfoKHR& presentInfo);

	public:
		VkInstance instance;
		VkSurfaceKHR surface;

		//index of the image buffer acquired from swapchain
		uint32_t currentImage;

		VkSwapchainKHR swapChain = VK_NULL_HANDLE;
		uint32_t imageCount = 0;

		VkFormat colorFormat;
		VkColorSpaceKHR colorSpace;
		std::vector<VkImage> images;
		std::vector<VkImageView> imageViews;

		VkSemaphore presentComplete;


		void initSurface(xcb_connection_t* connection, xcb_window_t& window);
		void setPhysicalDevice();

		inline bool isDetailsComplete() {return details.isComplete();};
		std::set<std::optional<uint32_t>> getPresentQueueIndices();

		//Create the swapchain and get its images with given width and height
		void create(uint32_t* width, uint32_t* height, bool vsync = false);
		VkResult acquireNextImage();
		VkResult queuePresent(VkSemaphore waitSemaphore = VK_NULL_HANDLE);
		void destroy();
	};
};