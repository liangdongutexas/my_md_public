#include "vulkan-pipeline.h"

namespace vks
{
    /**
	 * @brief typedef struct VkDescriptorSetLayoutCreateInfo {					typedef struct VkDescriptorSetLayoutBinding {
				VkStructureType                        sType;								uint32_t              binding;
				const void*                            pNext;								VkDescriptorType      descriptorType;
				VkDescriptorSetLayoutCreateFlags       flags;								uint32_t              descriptorCount;
				uint32_t                               bindingCount;						VkShaderStageFlags    stageFlags;
				const VkDescriptorSetLayoutBinding*    pBindings;							const VkSampler*      pImmutableSamplers;
			} VkDescriptorSetLayoutCreateInfo;												} VkDescriptorSetLayoutBinding;
	*
	* The sum of VkDescriptorSetLayoutBinding.descriptorCount for all bindings in a VkDescriptorSetLayout determines
	* the total number of individual descriptors within a descriptor set created with that layout.
	* The actual creation or instantiation of descriptors (allocating memory, setting buffer or image resources) happens
	* when you bind resources to the descriptor set with calls like vkUpdateDescriptorSets
	*
	*  typedef struct VkDescriptorPoolCreateInfo {					typedef struct VkDescriptorPoolSize {
		VkStructureType                sType;											VkDescriptorType    type;
		const void*                    pNext;											uint32_t            descriptorCount;
		VkDescriptorPoolCreateFlags    flags;											} VkDescriptorPoolSize;
		uint32_t                       maxSets;
		uint32_t                       poolSizeCount;
		const VkDescriptorPoolSize*    pPoolSizes;
		} VkDescriptorPoolCreateInfo;
	*
	* If you plan to allocate multiple descriptor sets from the pool, then VkDescriptorPoolSize.descriptorCount for each type
	* must be equal to or larger than the total number of descriptors of that type needed for all those sets combined.
	*
	typedef struct VkDescriptorSetAllocateInfo {					typedef struct VkWriteDescriptorSet {
				VkStructureType                 sType;							VkStructureType                  sType;
				const void*                     pNext;							const void*                      pNext;
				VkDescriptorPool                descriptorPool;					VkDescriptorSet                  dstSet;
				uint32_t                        descriptorSetCount;				uint32_t                         dstBinding;
				const VkDescriptorSetLayout*    pSetLayouts;					uint32_t                         dstArrayElement;
			} VkDescriptorSetAllocateInfo;										uint32_t                         descriptorCount;
																				VkDescriptorType                 descriptorType;
																				const VkDescriptorImageInfo*     pImageInfo;
																				const VkDescriptorBufferInfo*    pBufferInfo;
																				const VkBufferView*              pTexelBufferView;
																			} VkWriteDescriptorSet;
	*
	*/
	void PipeLine::allocateDescriptorSet(const BufferBindings& bufferBindings, const ImageBindings& imageBindings)
	{
		std::vector<VkDescriptorPoolSize> poolSizes;
		for (uint32_t i = 0; i < bufferBindings.bindingCount; ++i) {
			VkDescriptorPoolSize poolSize = {bufferBindings.bindings[i].descriptorType, bufferBindings.bindings[i].descriptorCount};
			poolSizes.push_back(poolSize);
		}

		for (uint32_t i = 0; i < imageBindings.bindingCount; ++i) {
			VkDescriptorPoolSize poolSize = {imageBindings.bindings[i].descriptorType, imageBindings.bindings[i].descriptorCount};
			poolSizes.push_back(poolSize);
		}

		// Create the pool
		VkDescriptorPoolCreateInfo descriptorPoolInfo{};
		descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		descriptorPoolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		descriptorPoolInfo.pPoolSizes = poolSizes.data();
		descriptorPoolInfo.maxSets = 1;
		VK_CHECK_RESULT(vkCreateDescriptorPool(logicalDevice->device, &descriptorPoolInfo, nullptr, &descriptorPool));

		// Create descriptor set layout
		std::vector<VkDescriptorSetLayoutBinding> totalBindings = bufferBindings.bindings;
		totalBindings.insert(totalBindings.end(), imageBindings.bindings.begin(), imageBindings.bindings.end());
		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo{};
		descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		descriptorSetLayoutCreateInfo.pBindings = totalBindings.data();
		descriptorSetLayoutCreateInfo.bindingCount = totalBindings.size();
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(logicalDevice->device, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayout));

		// Create descriptor set
		VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{};
		descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		descriptorSetAllocateInfo.descriptorPool = descriptorPool;
		descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;
		descriptorSetAllocateInfo.descriptorSetCount = 1;
		VK_CHECK_RESULT(vkAllocateDescriptorSets(logicalDevice->device, &descriptorSetAllocateInfo, &descriptorSet));

		std::vector<VkWriteDescriptorSet> writeDescriptorSets;
		for (uint32_t i = 0; i < bufferBindings.bindingCount; ++i) {
			VkWriteDescriptorSet writeDescriptorSet{};
			writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writeDescriptorSet.dstSet = descriptorSet;
			writeDescriptorSet.descriptorType = bufferBindings.bindings[i].descriptorType;
			writeDescriptorSet.dstBinding = bufferBindings.bindings[i].binding;
			writeDescriptorSet.descriptorCount = bufferBindings.bindings[i].descriptorCount;
			writeDescriptorSet.pBufferInfo = bufferBindings.bufferInfoss[i].data();
			writeDescriptorSets.push_back(writeDescriptorSet);
		}

		for (uint32_t i = 0; i < imageBindings.bindingCount; ++i) {
			VkWriteDescriptorSet writeDescriptorSet{};
			writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writeDescriptorSet.dstSet = descriptorSet;
			writeDescriptorSet.descriptorType = imageBindings.bindings[i].descriptorType;
			writeDescriptorSet.dstBinding = imageBindings.bindings[i].binding;
			writeDescriptorSet.descriptorCount = imageBindings.bindings[i].descriptorCount;
			writeDescriptorSet.pImageInfo = imageBindings.imageInfoss[i].data();
			writeDescriptorSets.push_back(writeDescriptorSet);
		}
		vkUpdateDescriptorSets(logicalDevice->device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, NULL);
	}

	void PipeLine::allocateDescriptorSet(const BufferBindings& bufferBindings)
	{
		ImageBindings imageBindings = ImageBindings();
		allocateDescriptorSet(bufferBindings, imageBindings);
	};

	void PipeLine::allocateDescriptorSet(const ImageBindings& imageBindings)
	{
		BufferBindings bufferBindings = BufferBindings();
		allocateDescriptorSet(bufferBindings, imageBindings);
	};

	/**
	 * @brief typedef struct VkPipelineLayoutCreateInfo {
				VkStructureType                 sType;
				const void*                     pNext;
				VkPipelineLayoutCreateFlags     flags;
				uint32_t                        setLayoutCount;
				const VkDescriptorSetLayout*    pSetLayouts;
				uint32_t                        pushConstantRangeCount;
				const VkPushConstantRange*      pPushConstantRanges;
			} VkPipelineLayoutCreateInfo;
			typedef struct VkComputePipelineCreateInfo {
				VkStructureType                    sType;
				const void*                        pNext;
				VkPipelineCreateFlags              flags;
				VkPipelineShaderStageCreateInfo    stage;
				VkPipelineLayout                   layout;
				VkPipeline                         basePipelineHandle;
				int32_t                            basePipelineIndex;
			} VkComputePipelineCreateInfo;
	 */
	void PipeLine::createPipeline(VkComputePipelineCreateInfo pipelineCreateInfo, VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo)
	{
		// Create compute pipeline
		// Compute pipelines are created separate from graphics pipelines even if they use the same queue (family index)
		pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutCreateInfo.setLayoutCount = 1;
		pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
		VK_CHECK_RESULT(vkCreatePipelineLayout(logicalDevice->device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

		pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		pipelineCreateInfo.layout = pipelineLayout;
		pipelineCreateInfo.stage = shaderStages[0];
		VK_CHECK_RESULT(vkCreateComputePipelines(logicalDevice->device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipeline));
	};

	void PipeLine::createComputePipeline()
	{
		VkComputePipelineCreateInfo pipelineCreateInfo{};
		pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		createPipeline(pipelineCreateInfo);
	};

	/**
	 * @brief typedef struct VkGraphicsPipelineCreateInfo {
				VkStructureType                                  sType;
				const void*                                      pNext;
				VkPipelineCreateFlags                            flags;
				uint32_t                                         stageCount;
				const VkPipelineShaderStageCreateInfo*           pStages;
				const VkPipelineVertexInputStateCreateInfo*      pVertexInputState;
				const VkPipelineInputAssemblyStateCreateInfo*    pInputAssemblyState;
				const VkPipelineTessellationStateCreateInfo*     pTessellationState;
				const VkPipelineViewportStateCreateInfo*         pViewportState;
				const VkPipelineRasterizationStateCreateInfo*    pRasterizationState;
				const VkPipelineMultisampleStateCreateInfo*      pMultisampleState;
				const VkPipelineDepthStencilStateCreateInfo*     pDepthStencilState;
				const VkPipelineColorBlendStateCreateInfo*       pColorBlendState;
				const VkPipelineDynamicStateCreateInfo*          pDynamicState;
				VkPipelineLayout                                 layout;
				VkRenderPass                                     renderPass;
				uint32_t                                         subpass;
				VkPipeline                                       basePipelineHandle;
				int32_t                                          basePipelineIndex;
			} VkGraphicsPipelineCreateInfo;
	 *
	 */
	void PipeLine::createPipeline(VkGraphicsPipelineCreateInfo pipelineCreateInfo, VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo)
	{
		// Create compute pipeline
		// Compute pipelines are created separate from graphics pipelines even if they use the same queue (family index)
		pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutCreateInfo.setLayoutCount = 1;
		pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
		VK_CHECK_RESULT(vkCreatePipelineLayout(logicalDevice->device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

		pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineCreateInfo.layout = pipelineLayout;
		pipelineCreateInfo.stageCount =shaderStages.size();
		pipelineCreateInfo.pStages = shaderStages.data();

		VK_CHECK_RESULT(vkCreateGraphicsPipelines(logicalDevice->device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipeline));
	};


	// Function to add a semaphore to the map
	void PipeLine::addSemaphore(const std::string& name, bool signal) {
		VulkanSemaphore semaphore;
		semaphore.createSemaphore(signal);
		semaphoreMap[name] = semaphore;
	}

	VkSemaphore& PipeLine::getSemaphore(const std::string& name)
	{
		return semaphoreMap[name].semaphore;
	};


	/**
	 * @brief
	 *  typedef struct VkPipelineShaderStageCreateInfo {
			VkStructureType                     sType;
			const void*                         pNext;
			VkPipelineShaderStageCreateFlags    flags;
			VkShaderStageFlagBits               stage;
			VkShaderModule                      module;
			const char*                         pName;
			const VkSpecializationInfo*         pSpecializationInfo;
		} VkPipelineShaderStageCreateInfo;
	* @param fileName
	* @param stage
	* @return VkPipelineShaderStageCreateInfo
	*/
	bool PipeLine::loadShader(std::string fileName, VkShaderStageFlagBits stage)
	{
		VkPipelineShaderStageCreateInfo shaderStage = {};
		shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStage.stage = stage;
		//create the module from the filename source code which has to be spir-v format
		shaderStage.module = vks::tools::loadShader(fileName.c_str(), logicalDevice->device);
		shaderStage.pName = "main";
		assert(shaderStage.module != VK_NULL_HANDLE);
		shaderModules.push_back(shaderStage.module);
		shaderStages.push_back(shaderStage);

		return true;
	}
	void PipeLine::destroy()
	{
		// Ensure that the device is idle before destroying resources
		vkDeviceWaitIdle(logicalDevice->device);

		// Destroy semaphores
		for (auto& semaphoreEntry : semaphoreMap) {
			semaphoreEntry.second.destroy();
		}
		semaphoreMap.clear();

		// Destroy shader modules
		for (VkShaderModule shaderModule : shaderModules) {
			vkDestroyShaderModule(logicalDevice->device, shaderModule, nullptr);
		}
		shaderModules.clear();
		shaderStages.clear();

		// Destroy the pipeline cache
		if (pipelineCache != VK_NULL_HANDLE) {
			vkDestroyPipelineCache(logicalDevice->device, pipelineCache, nullptr);
			pipelineCache = VK_NULL_HANDLE;
		}

		// Destroy the pipeline
		if (pipeline != VK_NULL_HANDLE) {
			vkDestroyPipeline(logicalDevice->device, pipeline, nullptr);
			pipeline = VK_NULL_HANDLE;
		}

		// Destroy the pipeline layout
		if (pipelineLayout != VK_NULL_HANDLE) {
			vkDestroyPipelineLayout(logicalDevice->device, pipelineLayout, nullptr);
			pipelineLayout = VK_NULL_HANDLE;
		}

		// Destroy the descriptor set layout
		if (descriptorSetLayout != VK_NULL_HANDLE) {
			vkDestroyDescriptorSetLayout(logicalDevice->device, descriptorSetLayout, nullptr);
			descriptorSetLayout = VK_NULL_HANDLE;
		}

		// Normally, you don't need to explicitly destroy descriptor sets as they are cleaned up when the descriptor pool is destroyed.
		// But if needed, they should be freed before destroying the descriptor pool.

		// Destroy the descriptor pool
		if (descriptorPool != VK_NULL_HANDLE) {
			vkDestroyDescriptorPool(logicalDevice->device, descriptorPool, nullptr);
			descriptorPool = VK_NULL_HANDLE;
		}
	};
}