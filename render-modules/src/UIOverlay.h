/*
* UI overlay class using ImGui
*
* Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <sstream>
#include <iomanip>


#include <vulkan/vulkan.h>
#include "app-module.h"
#include "vulkan-tools.h"
#include "vulkan-debug.h"
#include "vulkan-buffer.h"
#include "vulkan-pipeline.h"
#include "vulkan-texture.h"
#include "imgui.h"
#include "xcbUI.h"



class UIOverlay: public AppModule
{
private:
	XcbUI* xcbUI = XcbUI::getXcbUI();

	vks::Buffer vertexBuffer;
	vks::Buffer indexBuffer;
	vks::Texture fontTexture;

	struct PushConstBlock {
		glm::vec2 scale;
		glm::vec2 translate;
	} pushConstBlock;

	bool firstTimeRender = true;
	int32_t vertexCount = 0;
	int32_t indexCount = 0;

	VkSampleCountFlagBits rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	//need to figure out where to load shaders
	vks::PipeLine graphicsPipeline;

	bool updated = false;
	float scale = 1.0f;

	//to be called by prepare() method
	void createFontTexture();

	void generateUI();
	bool updateVertexBuffer();


	bool header(const char* caption);
	bool checkBox(const char* caption, bool* value);
	bool checkBox(const char* caption, int32_t* value);
	bool radioButton(const char* caption, bool value);
	bool inputFloat(const char* caption, float* value, float step, uint32_t precision);
	bool sliderFloat(const char* caption, float* value, float min, float max);
	bool sliderInt(const char* caption, int32_t* value, int32_t min, int32_t max);
	bool comboBox(const char* caption, int32_t* itemindex, std::vector<std::string> items);
	bool button(const char* caption);
	bool colorPicker(const char* caption, float* color);
	void text(const char* formatstr, ...);

public:
	void init() override;
    void deinit() override;
    void prepare() override;
    void unprepare() override;

	void prepareGraphicsPipeline();
	void updateMemoryResources() override;

	void recordGraphicsCommandBuffer(VkCommandBuffer graphicsCommandBuffer);

	void handleEvent(KeyMouseEvent KMEvent) override;
};

