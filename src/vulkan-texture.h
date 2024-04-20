/*
* Vulkan texture loader
*
* Copyright(C) by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license(MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include <fstream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include <ktx.h>
#include <ktxvulkan.h>

#include "vulkan-logical-device.h"
#include "vulkan-physical-device.h"



namespace vks
{
	class Texture
	{
	private:
		VulkanPhysicalDevice* physicalDevice = VulkanPhysicalDevice::getVulkanPhysicalDevice();
		VulkanLogicalDevice* logicalDevice = VulkanLogicalDevice::getVulkanLogicalDevice();
	protected:
		VkImage image;
		VkDeviceMemory imageMemory;
		void* mappedMemory;
	public:
		VkImageView 		  view;
		VkSampler             sampler;
		VkDescriptorImageInfo descriptor;

		VkImageLayout         imageLayout;

		VkFormat 			  format;
		uint32_t              width, height;
		uint32_t              mipLevels;
		uint32_t              layerCount;
		uint32_t			  numFaces;

		void 	allocateImageMemory(VkImageCreateInfo imageInfo, VkMemoryPropertyFlags properties, void* data, std::vector<VkBufferImageCopy> bufferCopyRegions);
		void 	allocateImageMemory(VkImageCreateInfo imageInfo, VkMemoryPropertyFlags properties);
		void	createView(VkImageViewCreateInfo viewInfo);
		void 	createSampler(VkSamplerCreateInfo samplerInfo);
		void 	updateImageLayout(VkImageLayout imageLayout);

		VkResult flush(VkDeviceSize size, VkDeviceSize offset=0);
		void    updateDescriptor();
		void    destroy();
		ktxResult loadKTXFile(std::string filename, ktxTexture **target);

		void loadFromFile(
			std::string        filename,
			VkFormat           format,
			VkFilter           filter          = VK_FILTER_LINEAR,
			VkImageUsageFlags  imageUsageFlags = VK_IMAGE_USAGE_SAMPLED_BIT,
			VkImageLayout      imageLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			bool               forceLinear     = false);
		//layers and mip levels are assumed to be 1
		void fromBuffer(
			void *             buffer,
			VkFormat           format,
			uint32_t           texWidth,
			uint32_t           texHeight,
			VkFilter           filter          = VK_FILTER_LINEAR,
			VkImageUsageFlags  imageUsageFlags = VK_IMAGE_USAGE_SAMPLED_BIT,
			VkImageLayout      imageLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	};
}        // namespace vks
