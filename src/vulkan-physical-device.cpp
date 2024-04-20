#include "vulkan-physical-device.h"
#include <stdexcept>

namespace vks
{
	VulkanPhysicalDevice* VulkanPhysicalDevice::physicalDevice_T = nullptr;
	VulkanPhysicalDevice::VulkanPhysicalDevice(){};

	void VulkanPhysicalDevice::setPhysicalDevice(VkPhysicalDevice physicalDevice)
	{
		assert(physicalDevice);
		this->device = physicalDevice;

		// Store Properties features, limits and properties of the physical device for later use
		// Device properties also contain limits and sparse properties
		vkGetPhysicalDeviceProperties(physicalDevice, &properties);
		// Features should be checked by the examples before using them
		vkGetPhysicalDeviceFeatures(physicalDevice, &features);
		// Memory properties are used regularly for creating all kinds of buffers
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

		// Queue family properties, used for setting up requested queues upon device creation
		uint32_t queueFamilyCount;
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
		assert(queueFamilyCount > 0);
		queueFamilyProperties.resize(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilyProperties.data());

		// Get list of supported extensions
		uint32_t extCount = 0;
		vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, nullptr);
		if (extCount > 0)
		{
			std::vector<VkExtensionProperties> extensions(extCount);
			if (vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, &extensions.front()) == VK_SUCCESS)
			{
				for (auto ext : extensions)
				{
					supportedExtensions.push_back(std::string(ext.extensionName));
				}
			}
		}
	}

	VkFormatProperties VulkanPhysicalDevice::getFormatProperties(VkFormat format)
	{
		VkFormatProperties formatProperties;
		vkGetPhysicalDeviceFormatProperties(device, format, &formatProperties);\
		return formatProperties;
	};

	/**
	 * @brief return a vector of all indices that cover the queueFlags
	 */
	std::set<std::optional<uint32_t>> VulkanPhysicalDevice::getQueueFamilyIndices(VkQueueFlags queueFlags) const
	{
		std::set<std::optional<uint32_t>> result={std::nullopt};
		// For other queue types or if no separate compute queue is present, return the first one to support the requested flags
		for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++)
		{
			if ((queueFamilyProperties[i].queueFlags & queueFlags) == queueFlags)
			{
				result.insert((std::optional<uint32_t>)i);
			}
		}

		return result;
	}


	/**
	* Get the index of a memory type that has all the requested property bits set
	*
	* @param typeBits Bit mask with bits set for each memory type supported by the resource to request for (from VkMemoryRequirements)
	* @param properties Bit mask of properties for the memory type to request
	* @param (Optional) memTypeFound Pointer to a bool that is set to true if a matching memory type has been found
	*
	* @return Index of the requested memory type
	*
	* @throw Throws an exception if memTypeFound is null and no memory type could be found that supports the requested properties
	*/
	uint32_t VulkanPhysicalDevice::getMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties) const
	{
		for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
            if ((typeBits & (1 << i)) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

		throw std::runtime_error("Could not find a matching memory type");

	}

	/**
	* Select the best-fit depth format for this device from a list of possible depth (and stencil) formats
	*
	* @param checkSamplingSupport Check if the format can be sampled from (e.g. for shader reads)
	*
	* @return The depth format that best fits for the current device
	*
	* @throw Throws an exception if no depth format fits the requirements
	*/
	VkFormat VulkanPhysicalDevice::getSupportedDepthFormat(bool checkSamplingSupport)
	{
		// All depth formats may be optional, so we need to find a suitable depth format to use
		std::vector<VkFormat> depthFormats = { VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D16_UNORM_S8_UINT, VK_FORMAT_D16_UNORM };
		for (auto& format : depthFormats)
		{
			VkFormatProperties formatProperties;
			vkGetPhysicalDeviceFormatProperties(device, format, &formatProperties);
			// Format must support depth stencil attachment for optimal tiling
			if (formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
			{
				if (checkSamplingSupport) {
					if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
						continue;
					}
				}
				return format;
			}
		}
		throw std::runtime_error("Could not find a matching depth format");
	}
}