/*
* Vulkan texture loader
*
* Copyright(C) by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license(MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkan-texture.h"
#include "vulkan-buffer.h"
#include "vulkan-tools.h"

namespace vks
{
	/**
	 * @brief typedef struct VkImageMemoryBarrier {
			VkStructureType            sType;
			const void*                pNext;
			VkAccessFlags              srcAccessMask;
			VkAccessFlags              dstAccessMask;
			VkImageLayout              oldLayout;
			VkImageLayout              newLayout;
			uint32_t                   srcQueueFamilyIndex;
			uint32_t                   dstQueueFamilyIndex;
			VkImage                    image;
			VkImageSubresourceRange    subresourceRange;
		} VkImageMemoryBarrier;
	 */
	void Texture::allocateImageMemory(VkImageCreateInfo imageInfo, VkMemoryPropertyFlags properties, void* data, std::vector<VkBufferImageCopy> bufferCopyRegions)
	{
		allocateImageMemory(imageInfo, properties);

		// If a pointer to the buffer data has been passed, map the buffer and copy over the data
		if (data != nullptr)
		{
			VkMemoryRequirements memRequirements;
			vkGetImageMemoryRequirements(logicalDevice->device, image, &memRequirements);
			if ((properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0){
				// Staging buffers for font data upload it is in general larger than needed in case copied data is only a sub resource of the image
				vks::Buffer stagingBuffer;
				stagingBuffer.allocateMemory(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memRequirements.size, data, logicalDevice->transferFamilyIndex.value());

				VulkanCommandBuffer copyCmd	= logicalDevice->beginSingleTimeCommands(VK_QUEUE_TRANSFER_BIT);

				VkImageAspectFlags aspects=vks::tools::getImageAspectFlags(imageInfo.format);
				VkImageSubresourceRange  subresourceRange= {aspects, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};

				// Prepare for transfer
				tools::setImageLayout(
					copyCmd.commandBuffer,
					image,
					VK_IMAGE_LAYOUT_UNDEFINED,
					VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
					subresourceRange,
					VK_PIPELINE_STAGE_HOST_BIT,
					VK_PIPELINE_STAGE_TRANSFER_BIT);

				// Copy mip levels from staging buffer
				assert(bufferCopyRegions.size()>0);
				vkCmdCopyBufferToImage(
					copyCmd.commandBuffer,
					stagingBuffer.buffer,
					image,
					VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
					static_cast<uint32_t>(bufferCopyRegions.size()),
					bufferCopyRegions.data()
				);

				// Prepare for future use
				if ((imageInfo.initialLayout == VK_IMAGE_LAYOUT_UNDEFINED) || (imageInfo.initialLayout == VK_IMAGE_LAYOUT_PREINITIALIZED)){
					DEBUG_CERR<<"initial Layout is VK_IMAGE_LAYOUT_UNDEFINED or VK_IMAGE_LAYOUT_PREINITIALIZED "<<"cannot transition back to initial layout"<<std::endl;
				}
				else{
					tools::setImageLayout(
					copyCmd.commandBuffer,
					image,
					VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
					imageInfo.initialLayout,
					subresourceRange,
					VK_PIPELINE_STAGE_TRANSFER_BIT);
					logicalDevice->endSingleTimeCommands(copyCmd);
					stagingBuffer.destroy();
				}
			}

			else {
				VK_CHECK_RESULT(vkMapMemory(logicalDevice->device, imageMemory, 0, memRequirements.size, 0, &mappedMemory));
				memcpy(mappedMemory, data, memRequirements.size);
				if ((properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0){flush(memRequirements.size);}
				vkUnmapMemory(logicalDevice->device, imageMemory);
			}
		}
	};

	/**
	 * @brief typedef struct VkImageCreateInfo {
				VkStructureType          sType;
				const void*              pNext;
				VkImageCreateFlags       flags;
				VkImageType              imageType;
				VkFormat                 format;
				VkExtent3D               extent;
				uint32_t                 mipLevels;
				uint32_t                 arrayLayers;
				VkSampleCountFlagBits    samples;
				VkImageTiling            tiling;
				VkImageUsageFlags        usage;
				VkSharingMode            sharingMode;
				uint32_t                 queueFamilyIndexCount;
				const uint32_t*          pQueueFamilyIndices;
				VkImageLayout            initialLayout;
			} VkImageCreateInfo;
	 *
	 */
	void Texture::allocateImageMemory(VkImageCreateInfo imageInfo, VkMemoryPropertyFlags properties) {

		VkImageCreateInfo imageInfoCopy = imageInfo;

		imageInfoCopy.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		VK_CHECK_RESULT(vkCreateImage(logicalDevice->device, &imageInfoCopy, nullptr, &image));

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(logicalDevice->device, image, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = physicalDevice->getMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(logicalDevice->device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate image memory!");
		}

		vkBindImageMemory(logicalDevice->device, image, imageMemory, 0);
	};

	/**
	 * @brief typedef struct VkImageViewCreateInfo {
				VkStructureType            sType;
				const void*                pNext;
				VkImageViewCreateFlags     flags;
				VkImage                    image;
				VkImageViewType            viewType;
				VkFormat                   format;
				VkComponentMapping         components;
				VkImageSubresourceRange    subresourceRange;
			} VkImageViewCreateInfo;

	* typedef struct VkImageSubresourceRange {
			VkImageAspectFlags    aspectMask;
			uint32_t              baseMipLevel;
			uint32_t              levelCount;
			uint32_t              baseArrayLayer;
			uint32_t              layerCount;
		} VkImageSubresourceRange;
	 */
	void Texture::createView(VkImageViewCreateInfo viewInfo) {
		viewInfo.image = image;
		VK_CHECK_RESULT(vkCreateImageView(logicalDevice->device, &viewInfo, nullptr, &view));
		updateDescriptor();
    };

	/**
	 * @brief typedef struct VkSamplerCreateInfo {
				VkStructureType         sType;
				const void*             pNext;
				VkSamplerCreateFlags    flags;
				VkFilter                magFilter;
				VkFilter                minFilter;
				VkSamplerMipmapMode     mipmapMode;
				VkSamplerAddressMode    addressModeU;
				VkSamplerAddressMode    addressModeV;
				VkSamplerAddressMode    addressModeW;
				float                   mipLodBias;
				VkBool32                anisotropyEnable;
				float                   maxAnisotropy;
				VkBool32                compareEnable;
				VkCompareOp             compareOp;
				float                   minLod;
				float                   maxLod;
				VkBorderColor           borderColor;
				VkBool32                unnormalizedCoordinates;
			} VkSamplerCreateInfo;
	 *
	 */
	void Texture::createSampler(VkSamplerCreateInfo samplerInfo) {
		VK_CHECK_RESULT(vkCreateSampler(logicalDevice->device, &samplerInfo, nullptr, &sampler));
		updateDescriptor();
	};

	void Texture::updateImageLayout(VkImageLayout imageLayout)
	{
		this->imageLayout = imageLayout;
		updateDescriptor();
	};

	VkResult Texture::flush(VkDeviceSize size, VkDeviceSize offset)
	{
		VkMappedMemoryRange mappedRange = {};
		mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
		mappedRange.memory = imageMemory;
		mappedRange.offset = offset;
		mappedRange.size = size;
		return vkFlushMappedMemoryRanges(logicalDevice->device, 1, &mappedRange);
	};

	void Texture::destroy()
	{
		if (view!=VK_NULL_HANDLE){vkDestroyImageView(logicalDevice->device, view, nullptr);view=VK_NULL_HANDLE;}
		if(image!=VK_NULL_HANDLE){vkDestroyImage(logicalDevice->device, image, nullptr);image=VK_NULL_HANDLE;}
		if(imageMemory!=VK_NULL_HANDLE){vkFreeMemory(logicalDevice->device, imageMemory, nullptr);imageMemory=VK_NULL_HANDLE;}
		if (sampler!=VK_NULL_HANDLE){vkDestroySampler(logicalDevice->device, sampler, nullptr);sampler=VK_NULL_HANDLE;}
	};


	/**
	 * @brief typedef struct VkDescriptorImageInfo {
					VkSampler        sampler;
					VkImageView      imageView;
					VkImageLayout    imageLayout;
				} VkDescriptorImageInfo;
	 * A image in general contains multiple layers, mip levels, and aspects (such as color, depth, and stencil components)
	 *
	 */
	void Texture::updateDescriptor()
	{
		descriptor.sampler = sampler;
		descriptor.imageView = view;
		descriptor.imageLayout = imageLayout;
	}


	ktxResult Texture::loadKTXFile(std::string filename, ktxTexture **target)
	{
		ktxResult result = KTX_SUCCESS;

		if (!vks::tools::fileExists(filename)) {
			vks::tools::exitFatal("Could not load texture from " + filename + "\n\nMake sure the assets submodule has been checked out and is up-to-date.", -1);
		}
		result = ktxTexture_CreateFromNamedFile(filename.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, target);
		return result;
	}


	/**
	* Load a 2D texture including all mip levels
	* @brief typedef struct VkBufferImageCopy {
				VkDeviceSize                bufferOffset;
				uint32_t                    bufferRowLength;
				uint32_t                    bufferImageHeight;
				VkImageSubresourceLayers    imageSubresource;
				VkOffset3D                  imageOffset;
				VkExtent3D                  imageExtent;
			} VkBufferImageCopy;
	*
	* @param filename File to load (supports .ktx)
	* @param format Vulkan format of the image data stored in the file
	* @param device Vulkan device to create the texture on
	* @param copyQueue Queue used for the texture staging copy commands (must support transfer)
	* @param (Optional) imageUsageFlags Usage flags for the texture's image (defaults to VK_IMAGE_USAGE_SAMPLED_BIT)
	* @param (Optional) imageLayout Usage layout for the texture (defaults VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
	* @param (Optional) forceLinear Force linear tiling (not advised, defaults to false)
	*
	*/
	void Texture::loadFromFile(std::string filename, VkFormat format, VkFilter filter, VkImageUsageFlags imageUsageFlags, VkImageLayout imageLayout, bool forceLinear)
	{
		ktxTexture* ktxTexture;
		ktxResult result = loadKTXFile(filename, &ktxTexture);
		assert(result == KTX_SUCCESS);

		width = ktxTexture->baseWidth;
		height = ktxTexture->baseHeight;
		mipLevels = ktxTexture->numLevels;
		layerCount = ktxTexture->numLayers;
		numFaces =ktxTexture->numFaces;

		ktx_uint8_t *ktxTextureData = ktxTexture_GetData(ktxTexture);
		ktx_size_t ktxTextureSize = ktxTexture_GetSize(ktxTexture);


		// Only use linear tiling if requested (and supported by the device)
		// Support for linear tiling is mostly limited, so prefer to use
		// optimal tiling instead
		// On most implementations linear tiling will only support a very
		// limited amount of formats and features (mip maps, cubemaps, arrays, etc.)
		this->imageLayout = imageLayout;

		if (!forceLinear)
		{
			// Create optimal tiled target image
			VkImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
			imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
			imageCreateInfo.format = format;
			imageCreateInfo.mipLevels = mipLevels;
			imageCreateInfo.arrayLayers = 1;
			imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
			imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
			imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			imageCreateInfo.initialLayout = imageLayout;
			imageCreateInfo.extent = { width, height, 1 };
			// Ensure that the TRANSFER_DST bit is set for staging
			imageCreateInfo.usage = imageUsageFlags | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
			imageCreateInfo.flags = (numFaces==6)? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;

			// Setup buffer copy regions for each mip level
			std::vector<VkBufferImageCopy> bufferCopyRegions;
			for (uint32_t layer = 0; layer < layerCount; layer++)
			{
				for (uint32_t level = 0; level < mipLevels; level++)
				{
					ktx_size_t offset;
					KTX_error_code result = ktxTexture_GetImageOffset(ktxTexture, level, layer, 0, &offset);
					assert(result == KTX_SUCCESS);

					VkBufferImageCopy bufferCopyRegion = {};
					bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
					bufferCopyRegion.imageSubresource.mipLevel = level;
					bufferCopyRegion.imageSubresource.baseArrayLayer = layer;
					bufferCopyRegion.imageSubresource.layerCount = 1;
					bufferCopyRegion.imageExtent.width = std::max(1u,ktxTexture->baseWidth >> level);
					bufferCopyRegion.imageExtent.height = std::max(1u,ktxTexture->baseHeight >> level);
					bufferCopyRegion.imageExtent.depth = 1;
					bufferCopyRegion.bufferOffset = offset;

					bufferCopyRegions.push_back(bufferCopyRegion);
				}
			}
			allocateImageMemory(imageCreateInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, (void*)ktxTextureData, bufferCopyRegions);
		}
		else
		{
			// Prefer using optimal tiling, as linear tiling
			// may support only a small set of features
			// depending on implementation (e.g. no mip maps, only one layer, etc.)

			// Get device properties for the requested texture format
			VkFormatProperties formatProperties = physicalDevice->getFormatProperties(format);
			// Check if this support is supported for linear tiling
			assert(formatProperties.linearTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT);
			assert(mipLevels == 1 & layerCount==1);

			VkImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
			imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
			imageCreateInfo.format = format;
			imageCreateInfo.extent = { width, height, 1 };
			imageCreateInfo.mipLevels = 1;
			imageCreateInfo.arrayLayers = 1;
			imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
			imageCreateInfo.tiling = VK_IMAGE_TILING_LINEAR;
			imageCreateInfo.usage = imageUsageFlags;
			imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			imageCreateInfo.initialLayout = imageLayout;

			std::vector<VkBufferImageCopy> bufferCopyRegions = {};
			VkBufferImageCopy bufferCopyRegion;
			bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			bufferCopyRegion.imageSubresource.mipLevel = 0;
			bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
			bufferCopyRegion.imageSubresource.layerCount = 1;
			bufferCopyRegion.imageExtent.width = width;
			bufferCopyRegion.imageExtent.height = height;
			bufferCopyRegion.imageExtent.depth = 1;
			bufferCopyRegion.bufferOffset = 0;
			bufferCopyRegions.push_back(bufferCopyRegion);

			allocateImageMemory(imageCreateInfo, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, (void*)ktxTextureData, bufferCopyRegions);
		}

		ktxTexture_Destroy(ktxTexture);

		// Create image view
		// Textures are not directly accessed by the shaders and
		// are abstracted by image views containing additional
		// information and sub resource ranges
		VkImageViewCreateInfo viewCreateInfo = {};
		viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		if (numFaces == 6){
			viewCreateInfo.viewType = (layerCount==6)? VK_IMAGE_VIEW_TYPE_CUBE: VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
		}
		else {
			viewCreateInfo.viewType = (layerCount==1)? VK_IMAGE_VIEW_TYPE_2D: VK_IMAGE_VIEW_TYPE_2D_ARRAY;
		}
		viewCreateInfo.format = format;
		viewCreateInfo.flags = 0;
		viewCreateInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		// Linear tiling usually won't support mip maps
		// Only set mip map count if optimal tiling is used
		viewCreateInfo.subresourceRange.levelCount = mipLevels;
		viewCreateInfo.subresourceRange.layerCount = layerCount;
		viewCreateInfo.image = image;
		createView(viewCreateInfo);


		// Create a default sampler
		VkSamplerCreateInfo samplerCreateInfo = {};
		samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerCreateInfo.magFilter = filter;
		samplerCreateInfo.minFilter = filter;
		samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerCreateInfo.addressModeU = (layerCount == 1)? VK_SAMPLER_ADDRESS_MODE_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerCreateInfo.addressModeV = samplerCreateInfo.addressModeU;
		samplerCreateInfo.addressModeW = samplerCreateInfo.addressModeU;
		samplerCreateInfo.mipLodBias = 0.0f;
		samplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;
		samplerCreateInfo.minLod = 0.0f;
		// Max level-of-detail should match mip level count
		samplerCreateInfo.maxLod = (forceLinear) ? 0.0f: (float)mipLevels;
		// Only enable anisotropic filtering if enabled on the device
		samplerCreateInfo.maxAnisotropy = physicalDevice->features.samplerAnisotropy ? physicalDevice->properties.limits.maxSamplerAnisotropy : 1.0f;
		samplerCreateInfo.anisotropyEnable = physicalDevice->features.samplerAnisotropy;
		samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
		createSampler(samplerCreateInfo);

		// Update descriptor image info member that can be used for setting up descriptor sets
		updateDescriptor();
	}

	/**
	* Creates a 2D texture from a buffer
	*
	* @param buffer Buffer containing texture data to upload
	* @param bufferSize Size of the buffer in machine units
	* @param width Width of the texture to create
	* @param height Height of the texture to create
	* @param format Vulkan format of the image data stored in the file
	* @param device Vulkan device to create the texture on
	* @param copyQueue Queue used for the texture staging copy commands (must support transfer)
	* @param (Optional) filter Texture filtering for the sampler (defaults to VK_FILTER_LINEAR)
	* @param (Optional) imageUsageFlags Usage flags for the texture's image (defaults to VK_IMAGE_USAGE_SAMPLED_BIT)
	* @param (Optional) imageLayout Usage layout for the texture (defaults VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
	*/
	void Texture::fromBuffer(void* buffer, VkFormat format, uint32_t texWidth, uint32_t texHeight, VkFilter filter, VkImageUsageFlags imageUsageFlags, VkImageLayout imageLayout)
	{
		assert(buffer);
		width = texWidth;
		height = texHeight;
		mipLevels = 1;

		// Create optimal tiled target image
		VkImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
		imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
		imageCreateInfo.format = format;
		imageCreateInfo.mipLevels = mipLevels;
		imageCreateInfo.arrayLayers = 1;
		imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageCreateInfo.initialLayout = imageLayout;
		imageCreateInfo.extent = { width, height, 1 };
		// Ensure that the TRANSFER_DST bit is set for staging
		imageCreateInfo.usage = imageUsageFlags | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

		std::vector<VkBufferImageCopy> bufferCopyRegions = {};
		VkBufferImageCopy bufferCopyRegion;
		bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		bufferCopyRegion.imageSubresource.mipLevel = 0;
		bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
		bufferCopyRegion.imageSubresource.layerCount = 1;
		bufferCopyRegion.imageExtent.width = width;
		bufferCopyRegion.imageExtent.height = height;
		bufferCopyRegion.imageExtent.depth = 1;
		bufferCopyRegion.bufferOffset = 0;
		bufferCopyRegions.push_back(bufferCopyRegion);

		allocateImageMemory(imageCreateInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, bufferCopyRegions);

		// Create image view
		VkImageViewCreateInfo viewCreateInfo = {};
		viewCreateInfo.flags = 0;
		viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewCreateInfo.pNext = NULL;
		viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewCreateInfo.format = format;
		viewCreateInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		viewCreateInfo.subresourceRange.levelCount = 1;
		viewCreateInfo.image = image;
		createView(viewCreateInfo);

		// Create sampler
		VkSamplerCreateInfo samplerCreateInfo = {};
		samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerCreateInfo.magFilter = filter;
		samplerCreateInfo.minFilter = filter;
		samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerCreateInfo.mipLodBias = 0.0f;
		samplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;
		samplerCreateInfo.minLod = 0.0f;
		samplerCreateInfo.maxLod = 0.0f;
		samplerCreateInfo.maxAnisotropy = 1.0f;
		createSampler(samplerCreateInfo);

		// Update descriptor image info member that can be used for setting up descriptor sets
		updateDescriptor();
	}

}
