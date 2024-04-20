/*
 * Vulkan device class
 *
 * Encapsulates a physical Vulkan device and its logical representation
 *
 * Copyright (C) 2016-2023 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#pragma once

#include <vulkan/vulkan.h>
#include <algorithm>
#include <assert.h>
#include <exception>
#include <string>
#include <optional>
#include <set>

namespace vks
{
	class VulkanPhysicalDevice
	{
	private:
		static VulkanPhysicalDevice* physicalDevice_T;

		VulkanPhysicalDevice();
	public:
		inline static VulkanPhysicalDevice* getVulkanPhysicalDevice(){
			if (!physicalDevice_T){
				physicalDevice_T = new VulkanPhysicalDevice();
			}

			return physicalDevice_T;
		}

		/** @brief Physical device representation */
		VkPhysicalDevice device;

		/** @brief List of extensions supported by the device */
		std::vector<std::string> supportedExtensions;

		/** @brief Queue family properties of the physical device */
		std::vector<VkQueueFamilyProperties> queueFamilyProperties;

		/** @brief Properties of the physical device including limits that the application can check against */
		VkPhysicalDeviceProperties properties;

		/** @brief Features of the physical device that an application can use to check if a feature is supported */
		VkPhysicalDeviceFeatures features;

		/** @brief Memory types and heaps of the physical device */
		VkPhysicalDeviceMemoryProperties memoryProperties;

		VkPhysicalDevice operator() () {return device;};

		void setPhysicalDevice(VkPhysicalDevice physicalDevice);

		std::set<std::optional<uint32_t>> getQueueFamilyIndices(VkQueueFlags queueFlags) const;
		uint32_t getMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties) const;
		VkFormat	getSupportedDepthFormat(bool checkSamplingSupport);
		VkFormatProperties getFormatProperties(VkFormat format);

	};
}
