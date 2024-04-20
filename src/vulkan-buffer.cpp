/*
* Vulkan buffer class
*
* Encapsulates a Vulkan buffer
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkan-buffer.h"



namespace vks
{
	/**
	* @brief typedef struct VkBufferCreateInfo {
				VkStructureType        sType;
				const void*            pNext;
				VkBufferCreateFlags    flags;
				VkDeviceSize           size;
				VkBufferUsageFlags     usage;
				VkSharingMode          sharingMode;
				uint32_t               queueFamilyIndexCount;
				const uint32_t*        pQueueFamilyIndices;
			} VkBufferCreateInfo;
	*/


	void Buffer::allocateMemory(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkDeviceSize size)
	{
		if (size>0){
			destroy();
		}

		this->usageFlags = usageFlags;
		this->memoryPropertyFlags = memoryPropertyFlags;
		this->size = size;

		// Create the buffer handle
		VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usageFlags;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		VK_CHECK_RESULT(vkCreateBuffer(logicalDevice->device, &bufferInfo, nullptr, &buffer));

		// Create the memory backing up the buffer handle
		VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(logicalDevice->device, buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = physicalDevice->getMemoryType(memRequirements.memoryTypeBits, memoryPropertyFlags);


		// If the buffer has VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT set we also need to enable the appropriate flag during allocation
		VkMemoryAllocateFlagsInfoKHR allocFlagsInfo{};
		if (usageFlags & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
			allocFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_KHR;
			allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
			allocInfo.pNext = &allocFlagsInfo;
		}

		VK_CHECK_RESULT(vkAllocateMemory(logicalDevice->device, &allocInfo, nullptr, &memory));

		alignment = memRequirements.alignment;

		// Initialize a default descriptor that covers the whole buffer size
		setupDescriptor();

		// Attach the memory to the buffer object
		VK_CHECK_RESULT(bind());
	}

	void Buffer::allocateMemory(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkDeviceSize size, void *data, std::optional<uint32_t> dstQueueFamilyIndex)
	{

		// If a pointer to the buffer data has been passed, map the buffer and copy over the data
		if (data != nullptr)
		{
			if ((memoryPropertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0){
				vks::Buffer stagingBuffer;
				stagingBuffer.allocateMemory(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, size);
				allocateMemory(usageFlags | VK_BUFFER_USAGE_TRANSFER_DST_BIT, memoryPropertyFlags, size);

				stagingBuffer.map();
				memcpy(stagingBuffer.mapped, data, size);
				stagingBuffer.unmap();

				// Copy from staging buffer to storage buffer
				VulkanCommandBuffer vksCopyCmd = logicalDevice->beginSingleTimeCommands(VK_QUEUE_TRANSFER_BIT);
				VkBufferCopy copyRegion = {};
				copyRegion.size = size;
				vkCmdCopyBuffer(vksCopyCmd.commandBuffer, stagingBuffer.buffer, buffer, 1, &copyRegion);
				// Define the buffer memory barrier
				if (dstQueueFamilyIndex.has_value() &&  logicalDevice->transferFamilyIndex.value()!= dstQueueFamilyIndex.value()){
					VkBufferMemoryBarrier bufferMemoryBarrier = {
						.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
						.pNext = nullptr,
						.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,  // After data has been written via transfer
						.dstAccessMask = 0,   // Before being written by compute shader
						.srcQueueFamilyIndex = logicalDevice->transferFamilyIndex.value(),  // Assuming you have this index
						.dstQueueFamilyIndex = dstQueueFamilyIndex.value(),   // Assuming you have this index
						.buffer = buffer,
						.offset = 0,
						.size = VK_WHOLE_SIZE
					};

					vkCmdPipelineBarrier(
						vksCopyCmd.commandBuffer,
						VK_PIPELINE_STAGE_TRANSFER_BIT,  // After transfer write
						VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
						0,  // No special flags
						0, nullptr,  // No global memory barriers
						1, &bufferMemoryBarrier,  // Buffer transition
						0, nullptr  // No image barriers
					);
				}

				logicalDevice->endSingleTimeCommands(vksCopyCmd);
    			stagingBuffer.destroy();
			}
			else {
				allocateMemory(usageFlags, memoryPropertyFlags, size);
				VK_CHECK_RESULT(map());
				memcpy(mapped, data, size);
				if ((memoryPropertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0){flush();}
				unmap();
			}
		}
	}

	VkExternalMemoryHandleTypeFlagBits Buffer::getDefaultMemHandleType()
	{
	#ifdef _WIN64
		return IsWindows8Point1OrGreater()
					? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
					: VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
	#else
		return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
	#endif /* _WIN64 */
	};

	void* Buffer::createExternalBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkDeviceSize size)
	{
		VkBufferCreateInfo bufferInfo = {};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usageFlags;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			VkExternalMemoryBufferCreateInfo externalMemoryBufferInfo = {};
			externalMemoryBufferInfo.sType =
				VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
			externalMemoryBufferInfo.handleTypes = getDefaultMemHandleType();
		bufferInfo.pNext = &externalMemoryBufferInfo;
		VK_CHECK_RESULT(vkCreateBuffer(logicalDevice->device, &bufferInfo, nullptr, &buffer));

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(logicalDevice->device, buffer, &memRequirements);
			VkExportMemoryAllocateInfoKHR vulkanExportMemoryAllocateInfoKHR = {};
			vulkanExportMemoryAllocateInfoKHR.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
			vulkanExportMemoryAllocateInfoKHR.pNext = NULL;
			vulkanExportMemoryAllocateInfoKHR.handleTypes = getDefaultMemHandleType();
		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.pNext = &vulkanExportMemoryAllocateInfoKHR;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = physicalDevice->getMemoryType(memRequirements.memoryTypeBits, memoryPropertyFlags);

		VK_CHECK_RESULT(vkAllocateMemory(logicalDevice->device, &allocInfo, nullptr, &memory))

		bind();
		return getMemHandle();
	};


	void* Buffer::getMemHandle()
	{
	#ifdef _WIN64
		HANDLE handle = 0;

		VkMemoryGetWin32HandleInfoKHR vkMemoryGetWin32HandleInfoKHR = {};
		vkMemoryGetWin32HandleInfoKHR.sType =
			VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
		vkMemoryGetWin32HandleInfoKHR.pNext = NULL;
		vkMemoryGetWin32HandleInfoKHR.memory = memory;
		vkMemoryGetWin32HandleInfoKHR.handleType = handleType;

		PFN_vkGetMemoryWin32HandleKHR fpGetMemoryWin32HandleKHR;
		fpGetMemoryWin32HandleKHR =
			(PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(
				m_device, "vkGetMemoryWin32HandleKHR");
		if (!fpGetMemoryWin32HandleKHR) {
			throw std::runtime_error("Failed to retrieve vkGetMemoryWin32HandleKHR!");
		}
		if (fpGetMemoryWin32HandleKHR(m_device, &vkMemoryGetWin32HandleInfoKHR,
										&handle) != VK_SUCCESS) {
			throw std::runtime_error("Failed to retrieve handle for buffer!");
		}
		return (void *)handle;
	#else
		int fd = -1;

		VkMemoryGetFdInfoKHR vkMemoryGetFdInfoKHR = {};
		vkMemoryGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
		vkMemoryGetFdInfoKHR.pNext = NULL;
		vkMemoryGetFdInfoKHR.memory = memory;
		vkMemoryGetFdInfoKHR.handleType = getDefaultMemHandleType();

		PFN_vkGetMemoryFdKHR fpGetMemoryFdKHR;
		fpGetMemoryFdKHR =
			(PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(logicalDevice->device, "vkGetMemoryFdKHR");
		if (!fpGetMemoryFdKHR) {
			DEBUG_CERR<<"Failed to retrieve vkGetMemoryWin32HandleKHR!"<<std::endl;
		}
		if (fpGetMemoryFdKHR(logicalDevice->device, &vkMemoryGetFdInfoKHR, &fd) != VK_SUCCESS) {
			DEBUG_CERR<<"Failed to retrieve handle for buffer!"<<std::endl;
		}
		return (void *)(uintptr_t)fd;
		#endif /* _WIN64 */
	};


	void Buffer::importExternalBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkDeviceSize size, void* handle)
	{

	};



	/**
	* Map a memory range of this buffer. If successful, mapped points to the specified buffer range.
	*
	* @param size (Optional) Size of the memory range to map. Pass VK_WHOLE_SIZE to map the complete buffer range.
	* @param offset (Optional) Byte offset from beginning
	*
	* @return VkResult of the buffer mapping call
	*/
	VkResult Buffer::map(VkDeviceSize size, VkDeviceSize offset)
	{
		return vkMapMemory(logicalDevice->device, memory, offset, size, 0, &mapped);
	}

	/**
	* Unmap a mapped memory range
	*
	* @note Does not return a result as vkUnmapMemory can't fail
	*/
	void Buffer::unmap()
	{
		if (mapped)
		{
			vkUnmapMemory(logicalDevice->device, memory);
			mapped = nullptr;
		}
	}

	/**
	* Attach the allocated memory block to the buffer
	*
	* @param offset (Optional) Byte offset (from the beginning) for the memory region to bind
	*
	* @return VkResult of the bindBufferMemory call
	*/
	VkResult Buffer::bind(VkDeviceSize offset)
	{
		return vkBindBufferMemory(logicalDevice->device, buffer, memory, offset);
	}

	/**
	* Setup the default descriptor for this buffer
	* @brief typedef struct VkDescriptorBufferInfo {
				VkBuffer        buffer;
				VkDeviceSize    offset;
				VkDeviceSize    range;
			} VkDescriptorBufferInfo;
	* @param size (Optional) Size of the memory range of the descriptor
	* @param offset (Optional) Byte offset from beginning
	*
	*/
	void Buffer::setupDescriptor(VkDeviceSize size, VkDeviceSize offset)
	{
		descriptor.offset = offset;
		descriptor.buffer = buffer;
		descriptor.range = size;
	}


	/**
	* Copies the specified data to the mapped buffer
	*
	* @param data Pointer to the data to copy
	* @param size Size of the data to copy in machine units
	*
	*/
	void Buffer::copyTo(void* data, VkDeviceSize size, VkDeviceSize offset)
	{
		if (!mapped) {
            VK_CHECK_RESULT(map());
        }
        if (mapped) {
            memcpy(static_cast<char*>(mapped)+offset, data, size);
        }
        unmap();
	}

	/**
	* Flush a memory range of the buffer to make it visible to the device
	*
	* @note Only required for non-coherent memory
	*
	* @param size (Optional) Size of the memory range to flush. Pass VK_WHOLE_SIZE to flush the complete buffer range.
	* @param offset (Optional) Byte offset from beginning
	*
	* @return VkResult of the flush call
	*/
	VkResult Buffer::flush(VkDeviceSize size, VkDeviceSize offset)
	{
		VkMappedMemoryRange mappedRange = {};
		mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
		mappedRange.memory = memory;
		mappedRange.offset = offset;
		mappedRange.size = size;
		return vkFlushMappedMemoryRanges(logicalDevice->device, 1, &mappedRange);
	}

	/**
	* Invalidate a memory range of the buffer to make it visible to the host
	*
	* @note Only required for non-coherent memory
	*
	* @param size (Optional) Size of the memory range to invalidate. Pass VK_WHOLE_SIZE to invalidate the complete buffer range.
	* @param offset (Optional) Byte offset from beginning
	*
	* @return VkResult of the invalidate call
	*/
	VkResult Buffer::invalidate(VkDeviceSize size, VkDeviceSize offset)
	{
		VkMappedMemoryRange mappedRange = {};
		mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
		mappedRange.memory = memory;
		mappedRange.offset = offset;
		mappedRange.size = size;
		return vkInvalidateMappedMemoryRanges(logicalDevice->device, 1, &mappedRange);
	}

	void Buffer::destroy()
	{
		unmap();
        if (buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(logicalDevice->device, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
        }
        if (memory != VK_NULL_HANDLE) {
            vkFreeMemory(logicalDevice->device, memory, nullptr);
            memory = VK_NULL_HANDLE;
        }
	};

};
