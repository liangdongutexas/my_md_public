#include "app.h"

/**
 * @brief 0 is reserved for not suitable
 */
uint32_t App::rankIndexCombination(std::optional<uint32_t> graphicsIndex, std::optional<uint32_t> transferIndex, std::optional<uint32_t> computeIndex, std::optional<uint32_t> presentIndex)
{
	//starting from least wanted cases
	//least wanted combination when either graphics or present is not given bcause they are all required
	if (!graphicsIndex.has_value() || !presentIndex.has_value() || !computeIndex.has_value() || !transferIndex.has_value()){return 0;}
	if (graphicsIndex.value() != presentIndex.value()) {return 0;}

	//prefer compute queue different from graphics queue
	if (computeIndex ==graphicsIndex) {return 1;}
	else {
		return 2;
	}
}

std::vector<const char*> App::getInstanceLayers()
{
	std::vector<const char*> layersRequired{};
#ifdef DEBUG
	layersRequired.push_back("VK_LAYER_KHRONOS_validation");
#endif
	return layersRequired;
};

std::vector<const char*> App::getInstanceExtensions()
{
	std::vector<const char*> instanceExtensionsRequired = {};
	instanceExtensionsRequired.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
	instanceExtensionsRequired.push_back("VK_KHR_xcb_surface");
#ifdef DEBUG
	instanceExtensionsRequired.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

#ifdef CUDA_COMPUTE
	instanceExtensionsRequired.push_back("VK_KHR_external_memory_capabilities");
	instanceExtensionsRequired.push_back("VK_KHR_external_semaphore_capabilities");
#endif

	return instanceExtensionsRequired;
};

std::vector<const char*> App::getDeviceExtensions()
{
	std::vector<const char*> deviceExtensionsRequired = {};
	deviceExtensionsRequired.push_back("VK_KHR_swapchain");
	deviceExtensionsRequired.push_back("VK_EXT_conditional_rendering");

#ifdef CUDA_COMPUTE
	deviceExtensionsRequired.push_back("VK_KHR_external_memory");
	deviceExtensionsRequired.push_back("VK_KHR_external_semaphore");
	#if defined(__linux__)
		deviceExtensionsRequired.push_back("VK_KHR_external_memory_fd");
		deviceExtensionsRequired.push_back("VK_KHR_external_semaphore_fd");
	#elif defined(_WIN32)
		deviceExtensionsRequired.push_back("VK_KHR_external_memory_win32");
d		eviceExtensionsRequired.push_back("VK_KHR_external_semaphore_win32");
	#endif
#endif

	return deviceExtensionsRequired;
};

App::App()
{
	AppName = "VulkanParticles";
};

App::~App()
{
	VK_CHECK_RESULT(vkDeviceWaitIdle(logicalDevice->device));
	unprepare();
	deinit();
};

void App::init()
{
	VulkanBase::init();
	GUI.init();
	world.init();
};

void App::prepare()
{
	VulkanBase::prepare();
	if (!AppPrepared)
	{
		setupRenderPass();
		setupFrameBuffer();
		GUI.prepare();
		world.prepare();
		setupCommandBuffers();
		AppPrepared = true;
	}

};


void App::unprepare()
{
	if (AppPrepared)
	{
		VK_CHECK_RESULT(vkDeviceWaitIdle(logicalDevice->device));

		for (auto commandBuffer:graphicsCommandBuffers){
			commandBuffer.destroy();
		}
		graphicsCommandBuffers.clear();
		computeCommandBuffer.destroy();

		world.unprepare();
		GUI.unprepare();

		// Destroy frame buffers
		DEBUG_COUT << "framebuffer size: " << frameBuffers.size() << std::endl;
		for (auto framebuffer : frameBuffers) {
			DEBUG_COUT << "Destroying framebuffer: " << framebuffer << std::endl;
			vkDestroyFramebuffer(logicalDevice->device, framebuffer, nullptr);
		}
		frameBuffers.clear(); // Clear the vector after destroying its contents


		if (renderPass) {
			DEBUG_COUT << "Destroying render pass: " << renderPass << std::endl;
			vkDestroyRenderPass(logicalDevice->device, renderPass, nullptr);
		}

		AppPrepared = false;
	}
	VulkanBase::unprepare();
};

void App::deinit()
{
	world.deinit();
	GUI.deinit();
	VulkanBase::deinit();
}

/**
 * @brief typedef struct VkAttachmentDescription {					typedef struct VkRenderPassCreateInfo {
			VkAttachmentDescriptionFlags    flags;						VkStructureType                   sType;
			VkFormat                        format;						const void*                       pNext;
			VkSampleCountFlagBits           samples;					VkRenderPassCreateFlags           flags;
			VkAttachmentLoadOp              loadOp;						uint32_t                          attachmentCount;
			VkAttachmentStoreOp             storeOp;					const VkAttachmentDescription*    pAttachments;
			VkAttachmentLoadOp              stencilLoadOp;				uint32_t                          subpassCount;
			VkAttachmentStoreOp             stencilStoreOp;				const VkSubpassDescription*       pSubpasses;
			VkImageLayout                   initialLayout;				uint32_t                          dependencyCount;
			VkImageLayout                   finalLayout;				const VkSubpassDependency*        pDependencies;
		} VkAttachmentDescription;										} VkRenderPassCreateInfo;
*
* 			typedef struct VkSubpassDependency {
			uint32_t                srcSubpass;
			uint32_t                dstSubpass;
			VkPipelineStageFlags    srcStageMask;
			VkPipelineStageFlags    dstStageMask;
			VkAccessFlags           srcAccessMask;
			VkAccessFlags           dstAccessMask;
			VkDependencyFlags       dependencyFlags;
		} VkSubpassDependency;

		typedef struct VkSubpassDescription {
			VkSubpassDescriptionFlags       flags;
			VkPipelineBindPoint             pipelineBindPoint;
			uint32_t                        inputAttachmentCount;
			const VkAttachmentReference*    pInputAttachments;
			uint32_t                        colorAttachmentCount;
			const VkAttachmentReference*    pColorAttachments;
			const VkAttachmentReference*    pResolveAttachments;
			const VkAttachmentReference*    pDepthStencilAttachment;
			uint32_t                        preserveAttachmentCount;
			const uint32_t*                 pPreserveAttachments;
		} VkSubpassDescription;
 */
void App::setupRenderPass()
{
	if (renderPass != VK_NULL_HANDLE){
		vkDestroyRenderPass(logicalDevice->device, renderPass, nullptr);
		renderPass = VK_NULL_HANDLE;
	}

	std::array<VkAttachmentDescription, 2> attachments = {};
	// Color attachment
	attachments[0].format = swapchain.colorFormat;
	attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
	// Depth attachment
	attachments[1].format = depthStencil.format;
	attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;



	VkAttachmentReference colorReference = {};
	colorReference.attachment = 0;
	colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depthReference = {};
	depthReference.attachment = 1;
	depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpassDescription = {};
	subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpassDescription.colorAttachmentCount = 1;
	subpassDescription.pColorAttachments = &colorReference;
	subpassDescription.pDepthStencilAttachment = &depthReference;
	subpassDescription.inputAttachmentCount = 0;
	subpassDescription.pInputAttachments = nullptr;
	subpassDescription.preserveAttachmentCount = 0;
	subpassDescription.pPreserveAttachments = nullptr;
	subpassDescription.pResolveAttachments = nullptr;

	// Subpass dependencies for layout transitions
	std::array<VkSubpassDependency, 2> dependencies;

	dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[0].dstSubpass = 0;
	dependencies[0].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	dependencies[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
	dependencies[0].dependencyFlags = 0;

	dependencies[1].srcSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[1].dstSubpass = 0;
	dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[1].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[1].srcAccessMask = 0;
	dependencies[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
	dependencies[1].dependencyFlags = 0;


	VkRenderPassCreateInfo renderPassInfo = {};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
	renderPassInfo.pAttachments = attachments.data();
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpassDescription;
	renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
	renderPassInfo.pDependencies = dependencies.data();

	VK_CHECK_RESULT(vkCreateRenderPass(logicalDevice->device, &renderPassInfo, nullptr, &renderPass));

	VkClearValue clearValues[2];
    clearValues[0].color = defaultClearColor;
    clearValues[1].depthStencil = { 1.0f, 0 };
	renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = xcbUI->width;
    renderPassBeginInfo.renderArea.extent.height = xcbUI->height;
    renderPassBeginInfo.clearValueCount = 2;
    renderPassBeginInfo.pClearValues = clearValues;

	world.updateRenderPass(renderPass, 0);
	GUI.updateRenderPass(renderPass, 0);

}

/**
 * @brief typedef struct VkFramebufferCreateInfo {
			VkStructureType             sType;
			const void*                 pNext;
			VkFramebufferCreateFlags    flags;
			VkRenderPass                renderPass;
			uint32_t                    attachmentCount;
			const VkImageView*          pAttachments;
			uint32_t                    width;
			uint32_t                    height;
			uint32_t                    layers;
		} VkFramebufferCreateInfo;
 *
 */
void App::setupFrameBuffer()
{
	// Destroy frame buffers first
    for (auto framebuffer : frameBuffers) {
        vkDestroyFramebuffer(logicalDevice->device, framebuffer, nullptr);
    }
    frameBuffers.clear(); // Clear the vector after destroying its contents

	VkImageView attachments[2];
	// Depth/Stencil attachment is the same for all frame buffers
	attachments[1] = depthStencil.view;

	VkFramebufferCreateInfo frameBufferCreateInfo = {};
	frameBufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	frameBufferCreateInfo.pNext = NULL;
	frameBufferCreateInfo.renderPass = renderPass;
	frameBufferCreateInfo.attachmentCount = 2;
	frameBufferCreateInfo.pAttachments = attachments;
	frameBufferCreateInfo.width = xcbUI->width;
	frameBufferCreateInfo.height = xcbUI->height;
	frameBufferCreateInfo.layers = 1;

	// Create frame buffers for every swap chain image

	frameBuffers.resize(swapchain.imageCount);
	for (uint32_t i = 0; i < frameBuffers.size(); i++)
	{
		attachments[0] = swapchain.imageViews[i];
		VK_CHECK_RESULT(vkCreateFramebuffer(logicalDevice->device, &frameBufferCreateInfo, nullptr, &frameBuffers[i]));
	}
	/*
	DEBUG_COUT<<"frameBuffers size is "<<frameBuffers.size()<<std::endl;
	*/
}
void App::setupCommandBuffers()
{
	graphicsCommandBuffers.resize(frameBuffers.size());
	for (uint32_t i = 0; i < frameBuffers.size(); ++i)
	{
		graphicsCommandBuffers[i] = logicalDevice->getCommandBuffer(VK_QUEUE_GRAPHICS_BIT, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
	};

	computeCommandBuffer = logicalDevice->getCommandBuffer(VK_QUEUE_COMPUTE_BIT, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
};



void App::reRecordCommandBuffers()
{
	for (uint32_t i = 0; i < graphicsCommandBuffers.size(); ++i)
	{
		graphicsCommandBuffers[i].reset();
	};
	computeCommandBuffer.reset();

	recordCommandBuffers();
};

/**
 * @brief typedef struct VkRenderPassBeginInfo {
            VkStructureType        sType;
            const void*            pNext;
            VkRenderPass           renderPass;
            VkFramebuffer          framebuffer;
            VkRect2D               renderArea;
            uint32_t               clearValueCount;
            const VkClearValue*    pClearValues;
        } VkRenderPassBeginInfo;
 *
 */
void App::recordCommandBuffers()
{
	VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

	//record graphics command buffers
	VkClearValue clearValues[2];
    clearValues[0].color = defaultClearColor;
    clearValues[1].depthStencil = { 1.0f, 0 };

	VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
	renderPassBeginInfo.renderPass = renderPass;
	renderPassBeginInfo.renderArea.offset.x = 0;
	renderPassBeginInfo.renderArea.offset.y = 0;
	renderPassBeginInfo.renderArea.extent.width = xcbUI->width;
	renderPassBeginInfo.renderArea.extent.height = xcbUI->height;
	renderPassBeginInfo.clearValueCount = 2;
	renderPassBeginInfo.pClearValues = clearValues;

	for (uint32_t i = 0; i < frameBuffers.size(); ++i)
	{
		renderPassBeginInfo.framebuffer = frameBuffers[i];

		VK_CHECK_RESULT(vkBeginCommandBuffer(graphicsCommandBuffers[i].commandBuffer, &cmdBufInfo));
		{
			world.acquireBarrier(graphicsCommandBuffers[i].commandBuffer);
			// Draw the particle system using the update vertex buffer
			vkCmdBeginRenderPass(graphicsCommandBuffers[i].commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

				world.recordGraphicsCommandBuffer(graphicsCommandBuffers[i].commandBuffer);
				GUI.recordGraphicsCommandBuffer(graphicsCommandBuffers[i].commandBuffer);

			vkCmdEndRenderPass(graphicsCommandBuffers[i].commandBuffer);
			world.releaseBarrier(graphicsCommandBuffers[i].commandBuffer);
		}
		VK_CHECK_RESULT(vkEndCommandBuffer(graphicsCommandBuffers[i].commandBuffer));
	}

	//record compute command buffer
	VK_CHECK_RESULT(vkBeginCommandBuffer(computeCommandBuffer.commandBuffer, &cmdBufInfo));
	{
		world.recordComputeCommandBuffer(computeCommandBuffer.commandBuffer);
	}
	VK_CHECK_RESULT(vkEndCommandBuffer(computeCommandBuffer.commandBuffer));
};



void App::handleEvent(KeyMouseEvent KMEvent)
{
	if (xcbUI->windowResized){
		DEBUG_COUT<<"window resized!!!!!!!!!!!!!"<<std::endl;
		if (!basePrepared || !AppPrepared){
			return;
		}
		else{
			basePrepared = false;
			AppPrepared = false;

			// Ensure all operations on the logicalDevice->device have been finished before destroying resources
			vkDeviceWaitIdle(logicalDevice->device);
			unprepare();
			prepare();

			xcbUI->windowResized = false;
		}
	}
	else{
		world.handleEvent(KMEvent);
		GUI.handleEvent(KMEvent);
	}
};

/**
 * @brief typedef struct VkSubmitInfo {
			VkStructureType                sType; // Specifies the type of the structure.
			const void*                    pNext; // Pointer to extension-specific structure.
			uint32_t                       waitSemaphoreCount; // Number of semaphores to wait on before execution.
			const VkSemaphore*             pWaitSemaphores; // Array of semaphores to wait on.
			const VkPipelineStageFlags*    pWaitDstStageMask; // Array of pipeline stages at which each corresponding semaphore wait will occur.
			uint32_t                       commandBufferCount; // Number of command buffers to execute.
			const VkCommandBuffer*         pCommandBuffers; // Array of command buffers to execute.
			uint32_t                       signalSemaphoreCount; // Number of semaphores to be signaled once the command buffers have completed execution.
			const VkSemaphore*             pSignalSemaphores; // Array of semaphores to signal.
		} VkSubmitInfo;

 *
 */
void App::computeDrawSingleFrame()
{
	// Wait for rendering finished
    VkPipelineStageFlags computeWaitStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
    computeSubmitInfo.commandBufferCount = 1;
    computeSubmitInfo.pCommandBuffers = &computeCommandBuffer.commandBuffer;
    computeSubmitInfo.waitSemaphoreCount = 1;
    computeSubmitInfo.pWaitSemaphores = &world.graphicsPipeLine.getSemaphore("computeBegin");
    computeSubmitInfo.pWaitDstStageMask = &computeWaitStageMask;
    computeSubmitInfo.signalSemaphoreCount = 1;
    computeSubmitInfo.pSignalSemaphores = &world.computePipeLine.getSemaphore("renderingBegin");
    VK_CHECK_RESULT(vkQueueSubmit(logicalDevice->computeQueue, 1, &computeSubmitInfo, VK_NULL_HANDLE));
	VK_CHECK_RESULT(vkQueueWaitIdle(logicalDevice->computeQueue));


    VkPipelineStageFlags graphicsWaitStageMasks[] = { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    VkSemaphore graphicsWaitSemaphores[] = { world.computePipeLine.getSemaphore("renderingBegin"), swapchain.presentComplete };
    VkSemaphore graphicsSignalSemaphores[] = { world.graphicsPipeLine.getSemaphore("computeBegin"), world.graphicsPipeLine.getSemaphore("presentBegin") };

	// identify swapchain incompatible with the surface (OUT_OF_DATE) event
	if (swapchain.acquireNextImage() == VK_ERROR_OUT_OF_DATE_KHR) {
		xcbUI->windowResized = true;
	}
	// Submit graphics commands
    VkSubmitInfo graphicsSubmitInfo = vks::initializers::submitInfo();
    graphicsSubmitInfo.commandBufferCount = 1;
    graphicsSubmitInfo.pCommandBuffers = &graphicsCommandBuffers[swapchain.currentImage].commandBuffer;
	DEBUG_COUT<<"current image from swapchain is "<<swapchain.currentImage<<std::endl;
    graphicsSubmitInfo.waitSemaphoreCount = 2;
    graphicsSubmitInfo.pWaitSemaphores = graphicsWaitSemaphores;
    graphicsSubmitInfo.pWaitDstStageMask = graphicsWaitStageMasks;
    graphicsSubmitInfo.signalSemaphoreCount = 2;
    graphicsSubmitInfo.pSignalSemaphores = graphicsSignalSemaphores;
	DEBUG_COUT<<"begin submit graphics command buffer"<<std::endl;
    VK_CHECK_RESULT(vkQueueSubmit(logicalDevice->graphicsQueue, 1, &graphicsSubmitInfo, VK_NULL_HANDLE));
	VK_CHECK_RESULT(vkQueueWaitIdle(logicalDevice->graphicsQueue));

	DEBUG_COUT<<"begin present to swapchain"<<std::endl;
    VkResult result = swapchain.queuePresent(world.graphicsPipeLine.getSemaphore("presentBegin"));
	// Recreate the swapchain if it's no longer compatible with the surface (OUT_OF_DATE) or no longer optimal for presentation (SUBOPTIMAL)
	if ((result == VK_ERROR_OUT_OF_DATE_KHR) || (result == VK_SUBOPTIMAL_KHR)) {
		xcbUI->windowResized = true;
		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			return;
		}
	}
	else {
		VK_CHECK_RESULT(result);
	}
	VK_CHECK_RESULT(vkQueueWaitIdle(logicalDevice->presentQueue));
}

void App::renderLoop()
{
	xcbUI->flush();
	int i=0;
	while(!xcbUI->quit){


		xcbUI->frameBench.beginFrame();
			world.updateMemoryResources();
			GUI.updateMemoryResources();

			if(world.computeNeedRecording || world.graphicsNeedRecording || GUI.needRecording){
				DEBUG_COUT<<"begin d%th recording"<<i<<std::endl;
				reRecordCommandBuffers();
				++i;
			}
			//draw the image representation of the virtual world
			computeDrawSingleFrame();

		xcbUI->frameBench.endFrame();

		//get event caught by xcb
		KeyMouseEvent KMEvent = xcbUI->getEvent();
		//update the virtual world based on the mouse and keyboard status
		handleEvent(KMEvent);
	}

	// Flush logicalDevice->device to make sure all resources can be freed
	if (logicalDevice->device != VK_NULL_HANDLE) {
		vkDeviceWaitIdle(logicalDevice->device);
	}
}


