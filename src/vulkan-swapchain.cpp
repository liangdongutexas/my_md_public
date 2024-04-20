/*
* Class wrapping access to the swap chain
*
* A swap chain is a collection of framebuffers used for rendering and presentation to the windowing system
*
* Copyright (C) 2016-2023 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkan-swapchain.h"
/**
* Set instance, physical and logical device to use for the swapchain and get all required function pointers
*
* @param instance Vulkan instance to use
* @param physicalDevice Physical device used to query properties and formats relevant to the swapchain
* @param device Logical representation of the device to create the swapchain for
*
*/
namespace vks
{
	void VulkanSwapChain::initSurface(xcb_connection_t* connection, xcb_window_t& window)
	{
		VkResult err = VK_SUCCESS;

		// Check if instance is valid
		if (instance == VK_NULL_HANDLE) {
			vks::tools::exitFatal("Vulkan instance is not valid!", VK_ERROR_INITIALIZATION_FAILED);
		}

		// Check if connection is valid
		if (connection == nullptr) {
			vks::tools::exitFatal("XCB connection is not valid!", VK_ERROR_INITIALIZATION_FAILED);
		}

		// Check if window is valid
		if (window == XCB_NONE) {
			vks::tools::exitFatal("XCB window is not valid!", VK_ERROR_INITIALIZATION_FAILED);
		}

		if (surface != VK_NULL_HANDLE)
		{
			vkDestroySurfaceKHR(instance, surface, nullptr);
		}

		VkXcbSurfaceCreateInfoKHR surfaceCreateInfo = {};
		surfaceCreateInfo.sType = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR;
		surfaceCreateInfo.connection = connection;
		surfaceCreateInfo.window = window;
		err = vkCreateXcbSurfaceKHR(instance, &surfaceCreateInfo, nullptr, &surface);

		if (err != VK_SUCCESS) {
			vks::tools::exitFatal("Could not create surface!", err);
		}
	}

	/**
	 * @brief set physical device to swapchain
	 *
	 * @param physicalDevice
	 */
	void VulkanSwapChain::setPhysicalDevice()
	{
		//get physical device capabilities
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice->device, surface, &details.capabilities);

		//get physical device surface formats
		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice->device, surface, &formatCount, nullptr);
		if (formatCount != 0) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice->device, surface, &formatCount, details.formats.data());
		}

		//get physical device present mode
		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice->device, surface, &presentModeCount, nullptr);
		if (presentModeCount != 0) {
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice->device, surface, &presentModeCount, details.presentModes.data());
		}
	};

	std::set<std::optional<uint32_t>> VulkanSwapChain::getPresentQueueIndices()
	{
		std::set<std::optional<uint32_t>> result={std::nullopt};

		if (physicalDevice->device != VK_NULL_HANDLE) {
			// Queue family properties, used for setting up requested queues upon device creation
			uint32_t queueFamilyCount;
			vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice->device, &queueFamilyCount, nullptr);
			assert(queueFamilyCount > 0);
			std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
			vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice->device, &queueFamilyCount, queueFamilyProperties.data());


			for (uint32_t i = 0; i < queueFamilyCount; i++){
				VkBool32 presentSupport;
				vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice->device, i, surface, &presentSupport);
				if (presentSupport) {
					result.insert((std::optional<uint32_t>)i);
				}
			}
		}
		return result;
	}


	/**
	 * typedef struct VkSwapchainCreateInfoKHR {
		VkStructureType                  sType;
		const void*                      pNext;
		VkSwapchainCreateFlagsKHR        flags;
		VkSurfaceKHR                     surface;
		uint32_t                         minImageCount;
		VkFormat                         imageFormat;
		VkColorSpaceKHR                  imageColorSpace;
		VkExtent2D                       imageExtent;
		uint32_t                         imageArrayLayers;
		VkImageUsageFlags                imageUsage;
		VkSharingMode                    imageSharingMode;
		uint32_t                         queueFamilyIndexCount;
		const uint32_t*                  pQueueFamilyIndices;
		VkSurfaceTransformFlagBitsKHR    preTransform;
		VkCompositeAlphaFlagBitsKHR      compositeAlpha;
		VkPresentModeKHR                 presentMode;
		VkBool32                         clipped;
		VkSwapchainKHR                   oldSwapchain;
	} VkSwapchainCreateInfoKHR;
	*
	* Create the swapchain and get its images with given width and height
	*
	* @param width Pointer to the width of the swapchain (may be adjusted to fit the requirements of the swapchain)
	* @param height Pointer to the height of the swapchain (may be adjusted to fit the requirements of the swapchain)
	* @param vsync (Optional) Can be used to force vsync-ed rendering (by using VK_PRESENT_MODE_FIFO_KHR as presentation mode)
	*/
	void VulkanSwapChain::create(uint32_t *width, uint32_t *height, bool vsync)
	{
		// Store the current swap chain handle so we can use it later on to ease up recreation
		VkSwapchainKHR oldSwapchain = swapChain;

		VkExtent2D swapchainExtent = {};
		// If width (and height) equals the special value 0xFFFFFFFF, the size of the surface will be set by the swapchain
		if (details.capabilities.currentExtent.width == (uint32_t)-1)
		{
			// If the surface size is undefined, the size is set to the size of the images requested.
			swapchainExtent.width = std::clamp(*width, details.capabilities.minImageExtent.width, details.capabilities.maxImageExtent.width);
			swapchainExtent.height = std::clamp(*height, details.capabilities.minImageExtent.height, details.capabilities.maxImageExtent.height);
		}
		else
		{
			// If the surface size is defined, the swap chain size must match
			swapchainExtent = details.capabilities.currentExtent;
			*width = details.capabilities.currentExtent.width;
			*height = details.capabilities.currentExtent.height;
		}


		// Select a present mode for the swapchain
		// The VK_PRESENT_MODE_FIFO_KHR mode must always be present as per spec
		// This mode waits for the vertical blank ("v-sync")
		VkPresentModeKHR swapchainPresentMode = VK_PRESENT_MODE_FIFO_KHR;
		// If v-sync is not requested, try to find a mailbox mode
		// It's the lowest latency non-tearing present mode available
		if (!vsync)
		{
			for (size_t i = 0; i < details.presentModes.size(); i++)
			{
				if (details.presentModes[i] == VK_PRESENT_MODE_MAILBOX_KHR)
				{
					swapchainPresentMode = VK_PRESENT_MODE_MAILBOX_KHR;
					break;
				}
				if (details.presentModes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR)
				{
					swapchainPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
				}
			}
		}
		else{
			bool presentModeSupported = false;
			for (const auto& availablePresentMode : details.presentModes) {
				if (swapchainPresentMode == availablePresentMode) {
					presentModeSupported = true;
					break;
				}
			}
			if (!presentModeSupported) {
				std::cerr << "Preferred present mode is not supported by the surface." << std::endl;
			}
		}


		// Determine the number of images
		uint32_t desiredNumberOfSwapchainImages = details.capabilities.minImageCount + 1;

		if ((details.capabilities.maxImageCount > 0) && (desiredNumberOfSwapchainImages > details.capabilities.maxImageCount))
		{
			desiredNumberOfSwapchainImages = details.capabilities.maxImageCount;
		}



		// Find the transformation of the surface
		VkSurfaceTransformFlagsKHR preTransform;
		if (details.capabilities.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
		{
			// We prefer a non-rotated transform
			preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
		}
		else
		{
			preTransform = details.capabilities.currentTransform;
		}


		// Find a supported composite alpha format (not all devices support alpha opaque)
		VkCompositeAlphaFlagBitsKHR compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		// Simply select the first composite alpha format available
		std::vector<VkCompositeAlphaFlagBitsKHR> compositeAlphaFlags = {
			VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
			VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
			VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
			VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
		};
		bool compositeAlphaSupported = false;
		for (auto& compositeAlphaFlag : compositeAlphaFlags) {
			if (details.capabilities.supportedCompositeAlpha & compositeAlphaFlag) {
				compositeAlpha = compositeAlphaFlag;
				compositeAlphaSupported = true;
				break;
			};
		}
		if (!compositeAlphaSupported) {
			std::cerr << "Preferred compositeAlphaFlags is not supported by the surface." << std::endl;
		}


		// We want to get a format that best suits our needs, so we try to get one from a set of preferred formats
		// Initialize the format to the first one returned by the implementation in case we can't find one of the preffered formats
		VkSurfaceFormatKHR selectedFormat = details.formats[0];
		std::vector<VkFormat> preferredImageFormats = {
			VK_FORMAT_B8G8R8A8_UNORM,
			VK_FORMAT_R8G8B8A8_UNORM,
			VK_FORMAT_A8B8G8R8_UNORM_PACK32
		};
		bool formatSupported = false;
		for (auto& availableFormat : details.formats) {
			if (std::find(preferredImageFormats.begin(), preferredImageFormats.end(), availableFormat.format) != preferredImageFormats.end()) {
				selectedFormat = availableFormat;
				formatSupported = true;
				break;
			}
		}
		if (!formatSupported) {
			std::cerr << "Preferred image format is not supported by the surface." << std::endl;
		}
		colorFormat = selectedFormat.format;
		colorSpace = selectedFormat.colorSpace;



		VkSwapchainCreateInfoKHR swapchainCI = {};
		swapchainCI.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		swapchainCI.surface = surface;
		swapchainCI.minImageCount = desiredNumberOfSwapchainImages;
		swapchainCI.imageFormat = colorFormat;
		swapchainCI.imageColorSpace = colorSpace;
		swapchainCI.imageExtent = { swapchainExtent.width, swapchainExtent.height };
		swapchainCI.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		swapchainCI.preTransform = (VkSurfaceTransformFlagBitsKHR)preTransform;
		swapchainCI.imageArrayLayers = 1;

		if (logicalDevice->graphicsFamilyIndex.has_value() && logicalDevice->presentFamilyIndex.has_value() && (logicalDevice->graphicsFamilyIndex != logicalDevice->presentFamilyIndex)) {
            swapchainCI.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            swapchainCI.queueFamilyIndexCount = 2;
			uint32_t queueFamilyIndices[] = {logicalDevice->graphicsFamilyIndex.value(), logicalDevice->presentFamilyIndex.value()};
            swapchainCI.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            swapchainCI.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			swapchainCI.queueFamilyIndexCount = 0;
        }

		swapchainCI.presentMode = swapchainPresentMode;
		// Setting oldSwapChain to the saved handle of the previous swapchain aids in resource reuse and makes sure that we can still present already acquired images
		swapchainCI.oldSwapchain = oldSwapchain;
		// Setting clipped to VK_TRUE allows the implementation to discard rendering outside of the surface area
		swapchainCI.clipped = VK_TRUE;
		swapchainCI.compositeAlpha = compositeAlpha;

		// Enable transfer source on swap chain images if supported
		if (details.capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) {
			swapchainCI.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		}

		// Enable transfer destination on swap chain images if supported
		if (details.capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) {
			swapchainCI.imageUsage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		}

		if(logicalDevice->device != VK_NULL_HANDLE){
			/*
			printSwapchainCreateInfo(swapchainCI);
			*/
			VK_CHECK_RESULT(vkCreateSwapchainKHR(logicalDevice->device, &swapchainCI, nullptr, &swapChain));
		}
		else{
			std::cerr<<"logical device is absent when creating swapchain"<<std::endl;
		}



		// If an existing swap chain is re-created, destroy the old swap chain
		// This also cleans up all the presentable images
		if (oldSwapchain != VK_NULL_HANDLE)
		{
			for (uint32_t i = 0; i < imageCount; i++)
			{
				vkDestroyImageView(logicalDevice->device, imageViews[i], nullptr);
			}
			vkDestroySwapchainKHR(logicalDevice->device, oldSwapchain, nullptr);
		}
		VK_CHECK_RESULT(vkGetSwapchainImagesKHR(logicalDevice->device, swapChain, &imageCount, NULL));

		// Get the swap chain images
		images.resize(imageCount);
		VK_CHECK_RESULT(vkGetSwapchainImagesKHR(logicalDevice->device, swapChain, &imageCount, images.data()));

		// Get the swap chain buffers containing the image and imageview
		imageViews.resize(imageCount);
		for (uint32_t i = 0; i < imageCount; i++)
		{
			VkImageViewCreateInfo colorAttachmentView = {};
			colorAttachmentView.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			colorAttachmentView.pNext = NULL;
			colorAttachmentView.format = colorFormat;
			colorAttachmentView.components = {
				VK_COMPONENT_SWIZZLE_IDENTITY,
				VK_COMPONENT_SWIZZLE_IDENTITY,
				VK_COMPONENT_SWIZZLE_IDENTITY,
				VK_COMPONENT_SWIZZLE_IDENTITY
			};
			colorAttachmentView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			colorAttachmentView.subresourceRange.baseMipLevel = 0;
			colorAttachmentView.subresourceRange.levelCount = 1;
			colorAttachmentView.subresourceRange.baseArrayLayer = 0;
			colorAttachmentView.subresourceRange.layerCount = 1;
			colorAttachmentView.viewType = VK_IMAGE_VIEW_TYPE_2D;
			colorAttachmentView.flags = 0;

			colorAttachmentView.image = images[i];

			VK_CHECK_RESULT(vkCreateImageView(logicalDevice->device, &colorAttachmentView, nullptr, &imageViews[i]));
		}

		//create present complete semaphore
		VkSemaphoreCreateInfo semaphoreCreateInfo {};
		semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		VK_CHECK_RESULT(vkCreateSemaphore(logicalDevice->device, &semaphoreCreateInfo, nullptr, &presentComplete));
	}


	void VulkanSwapChain::printSwapchainCreateInfo(const VkSwapchainCreateInfoKHR& ci)
	{
		DEBUG_COUT << "VkSwapchainCreateInfoKHR:" << std::endl;
		DEBUG_COUT << "  sType: " << ci.sType << std::endl;
		DEBUG_COUT << "  pNext: " << ci.pNext << std::endl;
		DEBUG_COUT << "  flags: " << ci.flags << std::endl;
		DEBUG_COUT << "  surface: " << ci.surface << std::endl;
		DEBUG_COUT << "  minImageCount: " << ci.minImageCount << std::endl;
		DEBUG_COUT << "  imageFormat: " << ci.imageFormat << std::endl;
		DEBUG_COUT << "  imageColorSpace: " << ci.imageColorSpace << std::endl;
		DEBUG_COUT << "  imageExtent: " << ci.imageExtent.width << " x " << ci.imageExtent.height << std::endl;
		DEBUG_COUT << "  imageArrayLayers: " << ci.imageArrayLayers << std::endl;
		DEBUG_COUT << "  imageUsage: " << ci.imageUsage << std::endl;
		DEBUG_COUT << "  imageSharingMode: " << ci.imageSharingMode << std::endl;
		DEBUG_COUT << "  queueFamilyIndexCount: " << ci.queueFamilyIndexCount << std::endl;
		if (ci.queueFamilyIndexCount > 0 && ci.pQueueFamilyIndices != nullptr) {
			DEBUG_COUT << "  Queue Family Indices: ";
			for (uint32_t i = 0; i < ci.queueFamilyIndexCount; ++i) {
				DEBUG_COUT << ci.pQueueFamilyIndices[i] << " ";
			}
			DEBUG_COUT << std::endl;
		}
		DEBUG_COUT << "  preTransform: " << ci.preTransform << std::endl;
		DEBUG_COUT << "  compositeAlpha: " << ci.compositeAlpha << std::endl;
		DEBUG_COUT << "  presentMode: " << ci.presentMode << std::endl;
		DEBUG_COUT << "  clipped: " << ci.clipped << std::endl;
		DEBUG_COUT << "  oldSwapchain: " << ci.oldSwapchain << std::endl;
	}

	/**
	* Acquires the next image in the swap chain
	*
	* @param presentCompleteSemaphore (Optional) Semaphore that is signaled when the image is ready for use
	* @param imageIndex Pointer to the image index that will be increased if the next image could be acquired
	*
	* @note The function will always wait until the next image has been acquired by setting timeout to UINT64_MAX
	*
	* @return VkResult of the image acquisition
	*/
	VkResult VulkanSwapChain::acquireNextImage()
	{
		// By setting timeout to UINT64_MAX we will always wait until the next image has been acquired or an actual error is thrown
		// With that we don't have to handle VK_NOT_READY
		return vkAcquireNextImageKHR(logicalDevice->device, swapChain, UINT64_MAX, presentComplete, (VkFence)nullptr, &currentImage);
	}

	/**
	* Queue an image for presentation
	*
	* @param queue Presentation queue for presenting the image
	* @param imageIndex Index of the swapchain image to queue for presentation
	* @param waitSemaphore (Optional) Semaphore that is waited on before the image is presented (only used if != VK_NULL_HANDLE)
	*
	* @return VkResult of the queue presentation
	*/
	VkResult VulkanSwapChain::queuePresent(VkSemaphore waitSemaphore)
	{
		VkPresentInfoKHR presentInfo = {};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.pNext = NULL;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &swapChain;
		presentInfo.pImageIndices = &currentImage;
		// Check if a wait semaphore has been specified to wait for before presenting the image
		if (waitSemaphore != VK_NULL_HANDLE)
		{
			presentInfo.pWaitSemaphores = &waitSemaphore;
			presentInfo.waitSemaphoreCount = 1;
		}
		if (logicalDevice->presentQueue != VK_NULL_HANDLE){
			return vkQueuePresentKHR(logicalDevice->presentQueue, &presentInfo);
		}
		else{
			std::cerr<<"present queue is null"<<std::endl;
		}

	}


	// Utility function to check if the VkPresentInfoKHR is valid
	VkResult VulkanSwapChain::checkVkPresentInfoKHR(const VkPresentInfoKHR& presentInfo) {
		// Check if the structure type is correct
		if (presentInfo.sType != VK_STRUCTURE_TYPE_PRESENT_INFO_KHR) {
			std::cerr << "Invalid sType for VkPresentInfoKHR!" << std::endl;
			return VK_ERROR_INITIALIZATION_FAILED;
		}

		// Check if the pointers are not null where necessary
		if (presentInfo.swapchainCount == 0 || presentInfo.pSwapchains == nullptr || presentInfo.pImageIndices == nullptr) {
			std::cerr << "Swapchain count is zero, or swapchain/image indices pointers are null!" << std::endl;
			return VK_ERROR_INITIALIZATION_FAILED;
		}

		// Validate swapchain handles and image indices
		std::vector<VkSwapchainKHR> validSwapChains={swapChain};

		for (uint32_t i = 0; i < presentInfo.swapchainCount; ++i) {
			// Check if the swapchain handle is valid
			bool isValidSwapChain = false;
			for (const auto& sc : validSwapChains) {
				if (sc == presentInfo.pSwapchains[i]) {
					isValidSwapChain = true;
					break;
				}
			}

			if (!isValidSwapChain) {
				std::cerr << "Invalid swapchain handle provided!" << std::endl;
				return VK_ERROR_INITIALIZATION_FAILED;
			}

			// Check if the image index is within the allowed range
			if (presentInfo.pImageIndices[i] >= imageCount) {
				std::cerr << "Image index out of bounds!" << std::endl;
				return VK_ERROR_INITIALIZATION_FAILED;
			}
		}

		// Validate wait semaphores if they are provided
		if (presentInfo.waitSemaphoreCount > 0 && presentInfo.pWaitSemaphores == nullptr) {
			std::cerr << "Wait semaphore count is non-zero but wait semaphores pointer is null!" << std::endl;
			return VK_ERROR_INITIALIZATION_FAILED;
		}

		return VK_SUCCESS;
	}



	/**
	* Destroy and free Vulkan resources used for the swapchain
	*/
	void VulkanSwapChain::destroy()
	{

		details.formats.clear();
		details.presentModes.clear();

		if (swapChain != VK_NULL_HANDLE)
		{
			for (uint32_t i = 0; i < images.size(); i++)
			{
				vkDestroyImageView(logicalDevice->device, imageViews[i], nullptr);
				imageViews.clear();
			}

			vkDestroySwapchainKHR(logicalDevice->device, swapChain, nullptr);
			swapChain = VK_NULL_HANDLE;
		}

		if (surface != VK_NULL_HANDLE)
		{
			if (instance == VK_NULL_HANDLE){
				std::cerr<<"instance is null, cannot detroy surface"<<std::endl;
			}
			else{
				vkDestroySurfaceKHR(instance, surface, nullptr);
				surface = VK_NULL_HANDLE;
				instance = VK_NULL_HANDLE;
			}
		}

		if (presentComplete != VK_NULL_HANDLE){
			vkDestroySemaphore(logicalDevice->device, presentComplete, 0);
		}
	}
}
