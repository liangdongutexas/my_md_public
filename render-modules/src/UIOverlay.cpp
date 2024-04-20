
/*
* UI overlay class using ImGui
*
* Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/
#include "UIOverlay.h"

void UIOverlay::init()
{
	AppModule::init();
	// Init ImGui
	ImGui::CreateContext();
	// Color scheme
	ImGuiStyle& style = ImGui::GetStyle();
	style.Colors[ImGuiCol_TitleBg] = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
	style.Colors[ImGuiCol_TitleBgActive] = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
	style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(1.0f, 0.0f, 0.0f, 0.1f);
	style.Colors[ImGuiCol_MenuBarBg] = ImVec4(1.0f, 0.0f, 0.0f, 0.4f);
	style.Colors[ImGuiCol_Header] = ImVec4(0.8f, 0.0f, 0.0f, 0.4f);
	style.Colors[ImGuiCol_HeaderActive] = ImVec4(1.0f, 0.0f, 0.0f, 0.4f);
	style.Colors[ImGuiCol_HeaderHovered] = ImVec4(1.0f, 0.0f, 0.0f, 0.4f);
	style.Colors[ImGuiCol_FrameBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.8f);
	style.Colors[ImGuiCol_CheckMark] = ImVec4(1.0f, 0.0f, 0.0f, 0.8f);
	style.Colors[ImGuiCol_SliderGrab] = ImVec4(1.0f, 0.0f, 0.0f, 0.4f);
	style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(1.0f, 0.0f, 0.0f, 0.8f);
	style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(1.0f, 1.0f, 1.0f, 0.1f);
	style.Colors[ImGuiCol_FrameBgActive] = ImVec4(1.0f, 1.0f, 1.0f, 0.2f);
	style.Colors[ImGuiCol_Button] = ImVec4(1.0f, 0.0f, 0.0f, 0.4f);
	style.Colors[ImGuiCol_ButtonHovered] = ImVec4(1.0f, 0.0f, 0.0f, 0.6f);
	style.Colors[ImGuiCol_ButtonActive] = ImVec4(1.0f, 0.0f, 0.0f, 0.8f);
	style.Alpha = 1.0f;
	// Dimensions
	ImGuiIO& io = ImGui::GetIO();
	io.FontGlobalScale = scale;
	createFontTexture();
	memoryResourcesReady = true;
};

void UIOverlay::prepare()
{
	prepareGraphicsPipeline();
};

void UIOverlay::unprepare()
{
	graphicsPipeline.destroy();
	needRecording = true;
};

void UIOverlay::deinit()
{
	vertexBuffer.destroy();
	indexBuffer.destroy();
	fontTexture.destroy();

	if (ImGui::GetCurrentContext()) {
		ImGui::DestroyContext();
	}
	AppModule::deinit();
}

/** Prepare all vulkan resources required to render the UI overlay */
void UIOverlay::createFontTexture()
{
	ImGuiIO& io = ImGui::GetIO();

	// Create font texture
	unsigned char* fontData;
	int texWidth, texHeight;

	const std::string filename = getAssetBasePath() + "/Roboto-Medium.ttf";
	io.Fonts->AddFontFromFileTTF(filename.c_str(), 16.0f * scale);
	io.Fonts->Build();

	io.Fonts->GetTexDataAsRGBA32(&fontData, &texWidth, &texHeight);


	//SRS - Set ImGui style scale factor to handle retina and other HiDPI displays (same as font scaling above)
	ImGuiStyle& style = ImGui::GetStyle();
	style.ScaleAllSizes(scale);

	// Create target image for copy
	VkImageCreateInfo imageInfo = vks::initializers::imageCreateInfo();
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
	imageInfo.extent.width = texWidth;
	imageInfo.extent.height = texHeight;
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	std::vector<VkBufferImageCopy> bufferCopyRegions = {};
	VkBufferImageCopy bufferCopyRegion;
	bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	bufferCopyRegion.imageSubresource.mipLevel = 0;
	bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
	bufferCopyRegion.imageSubresource.layerCount = 1;
	bufferCopyRegion.imageExtent.width = texWidth;
	bufferCopyRegion.imageExtent.height = texHeight;
	bufferCopyRegion.imageExtent.depth = 1;
	bufferCopyRegion.imageOffset = {0,0,0};
	bufferCopyRegion.bufferOffset = 0;
	bufferCopyRegion.bufferRowLength = 0;
	bufferCopyRegions.push_back(bufferCopyRegion);

	fontTexture.allocateImageMemory(imageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, fontData, bufferCopyRegions);

	VkImageViewCreateInfo viewInfo = vks::initializers::imageViewCreateInfo();
	viewInfo.flags = 0;
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
	viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	fontTexture.createView(viewInfo);

	// Font texture Sampler
	VkSamplerCreateInfo samplerInfo = vks::initializers::samplerCreateInfo();
	samplerInfo.magFilter = VK_FILTER_LINEAR;
	samplerInfo.minFilter = VK_FILTER_LINEAR;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
	fontTexture.createSampler(samplerInfo);

	fontTexture.updateImageLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

/** Prepare a separate pipeline for the UI overlay rendering decoupled from the main application */
void UIOverlay::prepareGraphicsPipeline()
{
    if (renderPass != VK_NULL_HANDLE){
        graphicsPipeline.destroy();
    }
    else{
        throw std::runtime_error("renderPass is not created, cannot create graphics pipeline");
    }


	vks::ImageBindings imageBindings(1);
	imageBindings.bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	imageBindings.bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	imageBindings.bindings[0].binding = 0;
	imageBindings.bindings[0].descriptorCount = 1;
	if (imageBindings.resizeImageInfoss()){
		imageBindings.imageInfoss[0][0] = fontTexture.descriptor;
		graphicsPipeline.allocateDescriptorSet(imageBindings);
	}


	if (graphicsPipeline.loadShader(getShaderBasePath() + "/glsl/base/uioverlay.vert.spv", VK_SHADER_STAGE_VERTEX_BIT) &&
	   graphicsPipeline.loadShader(getShaderBasePath() + "/glsl/base/uioverlay.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT))
	{
		// Pipeline layout
		// Push constants for UI rendering parameters
		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo;
		VkPushConstantRange pushConstantRange = vks::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(PushConstBlock), 0);
		pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
		pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
		pipelineLayoutCreateInfo.pNext = NULL;
		pipelineLayoutCreateInfo.flags = 0;


		// Setup graphics pipeline for UI rendering
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);

		VkPipelineRasterizationStateCreateInfo rasterizationState =
			vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);

		// Enable blending
		VkPipelineColorBlendAttachmentState blendAttachmentState{};
		blendAttachmentState.blendEnable = VK_TRUE;
		blendAttachmentState.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		blendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
		blendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		blendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;

		VkPipelineColorBlendStateCreateInfo colorBlendState =
			vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);

		VkPipelineDepthStencilStateCreateInfo depthStencilState =
			vks::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, VK_COMPARE_OP_ALWAYS);

		VkPipelineViewportStateCreateInfo viewportState =
			vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);

		VkPipelineMultisampleStateCreateInfo multisampleState =
			vks::initializers::pipelineMultisampleStateCreateInfo(rasterizationSamples);

		std::vector<VkDynamicState> dynamicStateEnables = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};
		VkPipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);

		VkGraphicsPipelineCreateInfo pipelineCreateInfo{};
		pipelineCreateInfo.renderPass = renderPass;
		pipelineCreateInfo.subpass = subpass;
		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;

/*
	#if defined(VK_KHR_dynamic_rendering)
		// SRS - if we are using dynamic rendering (i.e. renderPass null), must define color, depth and stencil attachments at pipeline create time
		VkPipelineRenderingCreateInfo pipelineRenderingCreateInfo = {};
		if (renderPass == VK_NULL_HANDLE) {
			pipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
			pipelineRenderingCreateInfo.colorAttachmentCount = 1;
			pipelineRenderingCreateInfo.pColorAttachmentFormats = &colorFormat;
			pipelineRenderingCreateInfo.depthAttachmentFormat = depthFormat;
			pipelineRenderingCreateInfo.stencilAttachmentFormat = depthFormat;
			pipelineCreateInfo.pNext = &pipelineRenderingCreateInfo;
		}
	#endif
*/
		// Vertex bindings an attributes based on ImGui vertex definition
		std::vector<VkVertexInputBindingDescription> vertexInputBindings = {
			vks::initializers::vertexInputBindingDescription(0, sizeof(ImDrawVert), VK_VERTEX_INPUT_RATE_VERTEX),
		};
		std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
			vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(ImDrawVert, pos)),	// Location 0: Position
			vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32_SFLOAT, offsetof(ImDrawVert, uv)),	// Location 1: UV
			vks::initializers::vertexInputAttributeDescription(0, 2, VK_FORMAT_R8G8B8A8_UNORM, offsetof(ImDrawVert, col)),	// Location 0: Color
		};
		VkPipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
		vertexInputState.pVertexBindingDescriptions = vertexInputBindings.data();
		vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
		vertexInputState.pVertexAttributeDescriptions = vertexInputAttributes.data();

		pipelineCreateInfo.pVertexInputState = &vertexInputState;


		graphicsPipeline.createPipeline(pipelineCreateInfo, pipelineLayoutCreateInfo);
	}

}

void UIOverlay::updateMemoryResources()
{
	generateUI();
};

void UIOverlay::generateUI()
{
	if (!visibility) return;

	ImGuiIO& io = ImGui::GetIO();
	io.DisplaySize = ImVec2((float)xcbUI->width, (float)xcbUI->height);
	io.DeltaTime = xcbUI->frameBench.frameTimer;

	if (ImGui::GetCurrentContext() == nullptr) {
        DEBUG_COUT << "ImGui context is not created" << std::endl;
    }

	ImFontAtlas* fontAtlas = io.Fonts;
	if (fontAtlas->IsBuilt()) {
		// Font atlas is built
	} else {
		DEBUG_COUT << "Font atlas is not built" << std::endl;
	}

	ImGuiStyle& style = ImGui::GetStyle();
	if (style.WindowRounding >= 0.0f && style.Colors[ImGuiCol_WindowBg].w > 0.0f) {
		// ImGui style settings are configured
	} else {
		DEBUG_COUT << "ImGui style settings are not configured properly" << std::endl;
	}

    // Check if the ImGui IO settings are properly initialized
    if (io.DisplaySize.x <= 0 || io.DisplaySize.y <= 0 || io.DeltaTime <= 0.0f) {
        DEBUG_COUT << "ImGui IO settings are not initialized properly" << std::endl;
    }
	if (firstTimeRender){
		firstTimeRender = false;
		generateUI();
	}

	ImGui::NewFrame();
		DEBUG_COUT<<"Generate ImGUI content "<<std::endl;
		ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
		ImGui::SetNextWindowPos(ImVec2(10 * scale, 10 * scale));
		ImGui::SetNextWindowSize(ImVec2(10 * scale, 10 * scale), ImGuiSetCond_FirstUseEver);
		if (ImGui::Begin("Vulkan Example", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove)) {
			DEBUG_COUT << "Window is active" << std::endl;

			// Continue with your UI code
			DEBUG_COUT << "Title: '" << xcbUI->title << "'" << std::endl;
			ImGui::TextUnformatted(xcbUI->title.c_str());
			DEBUG_COUT << "Device Name: '" << physicalDevice->properties.deviceName << "'" << std::endl;
			ImGui::TextUnformatted(physicalDevice->properties.deviceName);
			DEBUG_COUT << "Last FPS detected is '" << xcbUI->frameBench.lastFPS << "'" << std::endl;
			ImGui::Text("%.2f ms/frame (%.1d fps)", (1000.0f / xcbUI->frameBench.lastFPS), xcbUI->frameBench.lastFPS);

			ImGui::Button("Test Button");
		} else {
			DEBUG_COUT << "Window is not active" << std::endl;
		}

		ImGui::PushItemWidth(110.0f *scale);
		ImGui::PopItemWidth();
		ImGui::End();
		ImGui::PopStyleVar();

		DEBUG_COUT << "Ending ImGui content generation" << std::endl;
	ImGui::Render();

	needRecording = (updateVertexBuffer() || updated);
	updated = false;
};
/** Update vertex and index buffer containing the imGui elements when required */

bool UIOverlay::updateVertexBuffer()
{
	ImDrawData* imDrawData = ImGui::GetDrawData();
	bool updateCmdBuffers = false;

	if (imDrawData && imDrawData->CmdListsCount > 0) {
    	DEBUG_COUT << "Draw lists are available" << std::endl;
	}
	else {
		DEBUG_COUT << "No draw lists or commands" << std::endl;
		return false;
	}

	// Note: Alignment is done inside buffer creation
	VkDeviceSize vertexBufferSize = imDrawData->TotalVtxCount * sizeof(ImDrawVert);
	VkDeviceSize indexBufferSize = imDrawData->TotalIdxCount * sizeof(ImDrawIdx);
	DEBUG_COUT<<"vertexbufferSize is "<<vertexBufferSize<<" indexBufferSize is "<<indexBufferSize<<std::endl;
	// Update buffers only if vertex or index count has been changed compared to current buffer size
	if ((vertexBufferSize == 0) || (indexBufferSize == 0)) {
		return false;
	}

	DEBUG_COUT<<"vertex buffer is "<<vertexBuffer.buffer<<"index buffer is "<<indexBuffer.buffer<<std::endl;
	// Vertex buffer
	if ((vertexBuffer.buffer == VK_NULL_HANDLE) || (vertexCount != imDrawData->TotalVtxCount)) {
		vertexBuffer.destroy();
		DEBUG_COUT<<"Allocate vertex buffer "<<std::endl;
		vertexBuffer.allocateMemory(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, vertexBufferSize);
		vertexCount = imDrawData->TotalVtxCount;
		updateCmdBuffers = true;
	}

	// Index buffer
	if ((indexBuffer.buffer == VK_NULL_HANDLE) || (indexCount < imDrawData->TotalIdxCount)) {
		indexBuffer.destroy();
		DEBUG_COUT<<"Allocate index buffer "<<std::endl;
		indexBuffer.allocateMemory(VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, indexBufferSize);
		indexCount = imDrawData->TotalIdxCount;
		updateCmdBuffers = true;
	}

	// Upload data
	vertexBuffer.map();
	ImDrawVert* vtxDst = (ImDrawVert*)vertexBuffer.mapped;
	indexBuffer.map();
	ImDrawIdx* idxDst = (ImDrawIdx*)indexBuffer.mapped;

	for (int n = 0; n < imDrawData->CmdListsCount; n++) {
		const ImDrawList* cmd_list = imDrawData->CmdLists[n];
		memcpy(vtxDst, cmd_list->VtxBuffer.Data, cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
		memcpy(idxDst, cmd_list->IdxBuffer.Data, cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
		vtxDst += cmd_list->VtxBuffer.Size;
		idxDst += cmd_list->IdxBuffer.Size;
	}

	// Flush to make writes visible to GPU
	vertexBuffer.flush();
	vertexBuffer.unmap();
	indexBuffer.flush();
	indexBuffer.unmap();

	DEBUG_COUT<<"updated!! vertex buffer is "<<vertexBuffer.buffer<<"index buffer is "<<indexBuffer.buffer<<std::endl;
	return updateCmdBuffers;
}

/**
 * @brief it is assumed that the commandBuffer is in recording state
 */
void UIOverlay::recordGraphicsCommandBuffer(const VkCommandBuffer commandBuffer)
{
	needRecording = false;
	ImDrawData* imDrawData = ImGui::GetDrawData();
	int32_t vertexOffset = 0;
	int32_t indexOffset = 0;
	ImGuiIO& io = ImGui::GetIO();

    if (fpVkCmdBeginConditionalRenderingEXT != nullptr && fpVkCmdEndConditionalRenderingEXT != nullptr){
		fpVkCmdBeginConditionalRenderingEXT(commandBuffer, &conditionalRenderingBeginInfo);
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline.pipeline);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline.pipelineLayout, 0, 1, &graphicsPipeline.descriptorSet, 0, NULL);

			pushConstBlock.scale = glm::vec2(2.0f / io.DisplaySize.x, 2.0f / io.DisplaySize.y);
			pushConstBlock.translate = glm::vec2(-1.0f);
			vkCmdPushConstants(commandBuffer, graphicsPipeline.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstBlock), &pushConstBlock);

			VkDeviceSize offsets[1] = { 0 };
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer.buffer, offsets);
			vkCmdBindIndexBuffer(commandBuffer, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);

			for (int32_t i = 0; i < imDrawData->CmdListsCount; i++)
			{
				const ImDrawList* cmd_list = imDrawData->CmdLists[i];
				for (int32_t j = 0; j < cmd_list->CmdBuffer.Size; j++)
				{
					const ImDrawCmd* pcmd = &cmd_list->CmdBuffer[j];
					VkRect2D scissorRect;
					scissorRect.offset.x = std::max((int32_t)(pcmd->ClipRect.x), 0);
					scissorRect.offset.y = std::max((int32_t)(pcmd->ClipRect.y), 0);
					scissorRect.extent.width = (uint32_t)(pcmd->ClipRect.z - pcmd->ClipRect.x);
					scissorRect.extent.height = (uint32_t)(pcmd->ClipRect.w - pcmd->ClipRect.y);
					vkCmdSetScissor(commandBuffer, 0, 1, &scissorRect);
					vkCmdDrawIndexed(commandBuffer, pcmd->ElemCount, 1, indexOffset, vertexOffset, 0);
					indexOffset += pcmd->ElemCount;
				}
				vertexOffset += cmd_list->VtxBuffer.Size;
			}
		fpVkCmdEndConditionalRenderingEXT(commandBuffer);
	}
	else{
        DEBUG_CERR<<"cannot start conditional recording"<<std::endl;
    }
}

void UIOverlay::handleEvent(KeyMouseEvent KMEvent)
{
	if(KMEvent == KEY_F1_RELEASED){
		visibility = (visibility+1)%2;
		setVisibility(visibility);
	}
};

bool UIOverlay::header(const char *caption)
{
	return ImGui::CollapsingHeader(caption, ImGuiTreeNodeFlags_DefaultOpen);
}

bool UIOverlay::checkBox(const char *caption, bool *value)
{
	bool res = ImGui::Checkbox(caption, value);
	if (res) { updated = true; };
	return res;
}

bool UIOverlay::checkBox(const char *caption, int32_t *value)
{
	bool val = (*value == 1);
	bool res = ImGui::Checkbox(caption, &val);
	*value = val;
	if (res) { updated = true; };
	return res;
}

bool UIOverlay::radioButton(const char* caption, bool value)
{
	bool res = ImGui::RadioButton(caption, value);
	if (res) { updated = true; };
	return res;
}

bool UIOverlay::inputFloat(const char *caption, float *value, float step, uint32_t precision)
{
	bool res = ImGui::InputFloat(caption, value, step, step * 10.0f, precision);
	if (res) { updated = true; };
	return res;
}

bool UIOverlay::sliderFloat(const char* caption, float* value, float min, float max)
{
	bool res = ImGui::SliderFloat(caption, value, min, max);
	if (res) { updated = true; };
	return res;
}

bool UIOverlay::sliderInt(const char* caption, int32_t* value, int32_t min, int32_t max)
{
	bool res = ImGui::SliderInt(caption, value, min, max);
	if (res) { updated = true; };
	return res;
}

bool UIOverlay::comboBox(const char *caption, int32_t *itemindex, std::vector<std::string> items)
{
	if (items.empty()) {
		return false;
	}
	std::vector<const char*> charitems;
	charitems.reserve(items.size());
	for (size_t i = 0; i < items.size(); i++) {
		charitems.push_back(items[i].c_str());
	}
	uint32_t itemCount = static_cast<uint32_t>(charitems.size());
	bool res = ImGui::Combo(caption, itemindex, &charitems[0], itemCount, itemCount);
	if (res) { updated = true; };
	return res;
}

bool UIOverlay::button(const char *caption)
{
	bool res = ImGui::Button(caption);
	if (res) { updated = true; };
	return res;
}

bool UIOverlay::colorPicker(const char* caption, float* color) {
	bool res = ImGui::ColorEdit4(caption, color, ImGuiColorEditFlags_NoInputs);
	if (res) { updated = true; };
	return res;
}

void UIOverlay::text(const char *formatstr, ...)
{
	va_list args;
	va_start(args, formatstr);
	ImGui::TextV(formatstr, args);
	va_end(args);
}

