#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <map>
#include <string>

#include "vulkan-tools.h"
#include "vulkan-logical-device.h"
#include "vulkan-semaphore.h"


namespace vks
{
	struct BufferBindings {
		std::vector<VkDescriptorSetLayoutBinding> bindings;
		std::vector<std::vector<VkDescriptorBufferInfo>> bufferInfoss;
		const uint32_t bindingCount;

		inline BufferBindings(const VkDeviceSize bc = 0): bindingCount(bc) {bindings.resize(bindingCount); bufferInfoss.resize(bindingCount);};
		inline bool resizeBufferInfoss(){
			bool bindingsComplete = true;
			for (uint32_t i=0; i<bindingCount; i++){
				bindingsComplete =  (bindingsComplete && (bindings[i].descriptorCount > 0));
			}

			if (bindingsComplete){
				for (uint32_t i=0; i<bindingCount; i++){
					bufferInfoss[i].resize(bindings[i].descriptorCount);
				}
			}
			else {
				return false;
			}

			return true;
		}
	};

	struct ImageBindings {
		std::vector<VkDescriptorSetLayoutBinding> bindings;
		std::vector<std::vector<VkDescriptorImageInfo>> imageInfoss;
		const uint32_t bindingCount;

		inline ImageBindings(const VkDeviceSize bc = 0): bindingCount(bc) {bindings.resize(bindingCount); imageInfoss.resize(bindingCount);};

		inline bool resizeImageInfoss(){
			bool bindingsComplete = true;
			for (uint32_t i=0; i<bindingCount; i++){
				bindingsComplete =  (bindingsComplete && (bindings[i].descriptorCount > 0));
			}

			if (bindingsComplete){
				for (uint32_t i=0; i<bindingCount; i++){
					imageInfoss[i].resize(bindings[i].descriptorCount);
				}
			}
			else {
				return false;
			}

			return true;
		}
	};

	class PipeLine{
	private:
		VulkanLogicalDevice* logicalDevice = VulkanLogicalDevice::getVulkanLogicalDevice();
	public:
		VkDescriptorSetLayout descriptorSetLayout;	// shader binding layout
		VkDescriptorPool descriptorPool;            // Descriptor pool for compute module
		VkDescriptorSet descriptorSet;				// shader bindings

		VkPipelineLayout pipelineLayout;			// Layout of the compute pipeline
		VkPipeline pipeline;						// pipeline for updating particle positions
		VkPipelineCache pipelineCache;              // pipeline cache

		std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
		std::vector<VkShaderModule> shaderModules;	// store all the shader modules for the pipeline

		std::map<std::string, VulkanSemaphore> semaphoreMap;  // Execution dependency between compute & graphic submission

		inline ~PipeLine(){destroy();};

		void allocateDescriptorSet(const BufferBindings& bufferBindings, const ImageBindings& imageBindings);
		void allocateDescriptorSet(const BufferBindings& bufferBindings);
		void allocateDescriptorSet(const ImageBindings& imageBindings);

		void createPipeline(VkComputePipelineCreateInfo pipelineCreateInfo, VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = VkPipelineLayoutCreateInfo{});
		void createComputePipeline();

		void createPipeline(VkGraphicsPipelineCreateInfo pipelineCreateInfo, VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = VkPipelineLayoutCreateInfo{});

		void createPipeline(VkRayTracingPipelineCreateInfoKHR pipelineCreateInfo, VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = VkPipelineLayoutCreateInfo{});

		// Function to add a semaphore to the map
		void addSemaphore(const std::string& name, bool signal = false);
		VkSemaphore& getSemaphore(const std::string& name);

		bool loadShader(std::string fileName, VkShaderStageFlagBits stage);

		void destroy();
	};



}