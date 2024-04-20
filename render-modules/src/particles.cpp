#include "particles.h"


void PhysicsWorld::init()
{
    AppModule::init();
    prepareMemoryBuffers();
    loadAssets();
    memoryResourcesReady = true;
};

void PhysicsWorld::prepare()
{
    prepareComputePipeLine();
    prepareGraphicsPipeLine();
};

void PhysicsWorld::unprepare()
{
    graphicsPipeLine.destroy();
    computePipeLine.destroy();

    needRecording = true;
    computeNeedRecording = true;
    graphicsNeedRecording = true;
}

void PhysicsWorld::deinit()
{
    textures.particle.destroy();
    textures.gradient.destroy();
    particlesBuffer.destroy();
    uniformBuffer.destroy();
    AppModule::deinit();
}

// Setup and fill the compute shader storage buffers containing the particles
void PhysicsWorld::prepareMemoryBuffers()
{

    std::default_random_engine rndEngine(0);
    std::uniform_real_distribution<float> rndDist(-1.0f, 1.0f);

    // Initial particle positions
    std::vector<Particle> particleBuffer(PARTICLE_COUNT);
    for (auto& particle : particleBuffer) {
        particle.pos = glm::vec2(rndDist(rndEngine), rndDist(rndEngine));
        particle.vel = glm::vec2(0.0f);
        particle.gradientPos.x = particle.pos.x / 2.0f;
    }
    VkDeviceSize storageBufferSize = particleBuffer.size() * sizeof(Particle);

    particlesBuffer.allocateMemory(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, storageBufferSize, particleBuffer.data(), logicalDevice->computeFamilyIndex);

    uniformBuffer.allocateMemory(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, sizeof(UniformData));




    camera.updateAspectRatio((float)xcbUI->width / (float)xcbUI->height);
}

void PhysicsWorld::loadAssets()
{
    textures.particle.loadFromFile(getAssetBasePath() + "/textures/particle01_rgba.ktx", VK_FORMAT_R8G8B8A8_UNORM);
    textures.gradient.loadFromFile(getAssetBasePath() + "/textures/particle_gradient_rgba.ktx", VK_FORMAT_R8G8B8A8_UNORM);
}


/**
 * @brief typedef struct VkComputePipelineCreateInfo {
			VkStructureType                    sType;
			const void*                        pNext;
			VkPipelineCreateFlags              flags;
			VkPipelineShaderStageCreateInfo    stage;
			VkPipelineLayout                   layout;
			VkPipeline                         basePipelineHandle;
			int32_t                            basePipelineIndex;
		} VkComputePipelineCreateInfo;
 *
 */
void PhysicsWorld:: prepareComputePipeLine()
{
    computePipeLine.destroy();

    vks::BufferBindings bufferBindings(2);
    bufferBindings.bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bufferBindings.bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bufferBindings.bindings[0].binding = 0;
    bufferBindings.bindings[0].descriptorCount = 1;
    bufferBindings.bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bufferBindings.bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bufferBindings.bindings[1].binding = 1;
    bufferBindings.bindings[1].descriptorCount = 1;

    if (bufferBindings.resizeBufferInfoss()){
        bufferBindings.bufferInfoss[0][0] = particlesBuffer.descriptor;
        bufferBindings.bufferInfoss[1][0] = uniformBuffer.descriptor;
        computePipeLine.allocateDescriptorSet(bufferBindings);
    };


    // Create compute pipeline
    if (computePipeLine.loadShader(getShaderBasePath() + "/glsl/computeparticles/particle.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT)){
        computePipeLine.createComputePipeline();
    }

    // Semaphore for signaling completion of compute
    computePipeLine.addSemaphore("renderingBegin");
}

/**
 * @brief // Provided by VK_VERSION_1_0
    typedef struct VkGraphicsPipelineCreateInfo {
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
 *
 */
void PhysicsWorld::prepareGraphicsPipeLine()
{
    if (renderPass != VK_NULL_HANDLE){
        graphicsPipeLine.destroy();
    }
    else{
        throw std::runtime_error("renderPass is not created, cannot create graphics pipeline");
    }


    vks::ImageBindings imageBindings(2);
    imageBindings.bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    imageBindings.bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    imageBindings.bindings[0].binding = 0;
    imageBindings.bindings[0].descriptorCount = 1;
    imageBindings.bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    imageBindings.bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    imageBindings.bindings[1].binding = 1;
    imageBindings.bindings[1].descriptorCount = 1;

    if(imageBindings.resizeImageInfoss()){
        imageBindings.imageInfoss[0][0] = textures.particle.descriptor;
        imageBindings.imageInfoss[1][0] = textures.gradient.descriptor;
        graphicsPipeLine.allocateDescriptorSet(imageBindings);
    };


    // Pipeline layout
    if (graphicsPipeLine.loadShader(getShaderBasePath() + "/glsl/computeparticles/particle.vert.spv", VK_SHADER_STAGE_VERTEX_BIT)&&
    graphicsPipeLine.loadShader(getShaderBasePath() + "/glsl/computeparticles/particle.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT))
    {
        // Vertex Input state
        std::vector<VkVertexInputBindingDescription> inputBindings = {
            vks::initializers::vertexInputBindingDescription(0, sizeof(Particle), VK_VERTEX_INPUT_RATE_VERTEX)
        };
        std::vector<VkVertexInputAttributeDescription> inputAttributes = {
            // Location 0 : Position
            vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Particle, pos)),
            // Location 1 : Velocity (used for color gradient lookup)
            vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Particle, gradientPos)),
        };
        VkPipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
        vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(inputBindings.size());
        vertexInputState.pVertexBindingDescriptions = inputBindings.data();
        vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(inputAttributes.size());
        vertexInputState.pVertexAttributeDescriptions = inputAttributes.data();

        // Pipeline
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_POINT_LIST, 0, VK_FALSE);
        VkPipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
        VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
        VkPipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
        VkPipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, VK_COMPARE_OP_ALWAYS);
        VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
        VkPipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
        std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
        // Additive blending
        blendAttachmentState.colorWriteMask = 0xF;
        blendAttachmentState.blendEnable = VK_TRUE;
        blendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
        blendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;
        blendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_ALPHA;

        VkGraphicsPipelineCreateInfo pipelineCreateInfo{};
        pipelineCreateInfo.renderPass = renderPass;
        pipelineCreateInfo.subpass = subpass;
        pipelineCreateInfo.pVertexInputState = &vertexInputState;
        pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
        pipelineCreateInfo.pRasterizationState = &rasterizationState;
        pipelineCreateInfo.pColorBlendState = &colorBlendState;
        pipelineCreateInfo.pMultisampleState = &multisampleState;
        pipelineCreateInfo.pViewportState = &viewportState;
        pipelineCreateInfo.pDepthStencilState = &depthStencilState;
        pipelineCreateInfo.pDynamicState = &dynamicState;


        graphicsPipeLine.createPipeline(pipelineCreateInfo);
    }

    // Semaphore for signaling completion of rendering
    graphicsPipeLine.addSemaphore("computeBegin", true);
    graphicsPipeLine.addSemaphore("presentBegin");
}


/**
 * @brief assuming the computeCommandBuffer is already in record state
 * @brief typedef struct VkBufferMemoryBarrier {
			VkStructureType    sType;
			const void*        pNext;
			VkAccessFlags      srcAccessMask;
			VkAccessFlags      dstAccessMask;
			uint32_t           srcQueueFamilyIndex;
			uint32_t           dstQueueFamilyIndex;
			VkBuffer           buffer;
			VkDeviceSize       offset;
			VkDeviceSize       size;
		} VkBufferMemoryBarrier;
 *
 */
void PhysicsWorld::recordComputeCommandBuffer(VkCommandBuffer computeCommandBuffer)
{
    // Compute particle movement
    // Add memory barrier to ensure that the (graphics) vertex shader has fetched attributes before compute starts to write to the buffer
    if (firstRecord){
        firstRecord = false;
        computeNeedRecording = true;
        DEBUG_COUT<<"first time record compute command buffer"<<std::endl;
        if (logicalDevice->transferFamilyIndex.value()!= logicalDevice->computeFamilyIndex.value())
        {
            VkBufferMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.pNext = nullptr,
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.srcQueueFamilyIndex = logicalDevice->transferFamilyIndex.value();
            barrier.dstQueueFamilyIndex = logicalDevice->computeFamilyIndex.value();
            barrier.buffer = particlesBuffer.buffer;
            barrier.offset = 0;
            barrier.size = VK_WHOLE_SIZE;

            vkCmdPipelineBarrier(
                computeCommandBuffer,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr,
                1, &barrier,
                0, nullptr
            );
        }


    }
    else{
        DEBUG_COUT<<"at least second time record command buffer"<<std::endl;
        computeNeedRecording = false;
        if (logicalDevice->graphicsFamilyIndex.value()!= logicalDevice->computeFamilyIndex.value())
        {
            VkBufferMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.pNext = nullptr,
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.srcQueueFamilyIndex = logicalDevice->graphicsFamilyIndex.value();
            barrier.dstQueueFamilyIndex = logicalDevice->computeFamilyIndex.value();
            barrier.buffer = particlesBuffer.buffer;
            barrier.offset = 0;
            barrier.size = VK_WHOLE_SIZE;

            vkCmdPipelineBarrier(
                computeCommandBuffer,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr,
                1, &barrier,
                0, nullptr
            );
        }
    }

    // Dispatch the compute job
    vkCmdBindPipeline(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeLine.pipeline);
    vkCmdBindDescriptorSets(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeLine.pipelineLayout, 0, 1, &computePipeLine.descriptorSet, 0, 0);
    vkCmdDispatch(computeCommandBuffer, PARTICLE_COUNT / 256, 1, 1);

    // Add barrier to ensure that compute shader has finished writing to the buffer
    // Without this the (rendering) vertex shader may display incomplete results (partial data from last frame)
    if (logicalDevice->graphicsFamilyIndex.value()!= logicalDevice->computeFamilyIndex.value())
    {
        VkBufferMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.pNext = nullptr,
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = 0;
        barrier.srcQueueFamilyIndex = logicalDevice->computeFamilyIndex.value();  // Index of the compute queue family
        barrier.dstQueueFamilyIndex = logicalDevice->graphicsFamilyIndex.value(); // Index of the graphics queue family
        barrier.buffer = particlesBuffer.buffer;
        barrier.offset = 0;
        barrier.size = VK_WHOLE_SIZE;

        vkCmdPipelineBarrier(
            computeCommandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0,
            0, nullptr,
            1, &barrier,
            0, nullptr
        );
    }
}

void PhysicsWorld::acquireBarrier(VkCommandBuffer graphicsCommandBuffer)
{
    // Acquire barrier
            if (logicalDevice->graphicsFamilyIndex.value() != logicalDevice->computeFamilyIndex.value())
            {
                VkBufferMemoryBarrier barrier = {};
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barrier.pNext = nullptr,
                barrier.srcAccessMask = 0;
                barrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
                barrier.srcQueueFamilyIndex = logicalDevice->computeFamilyIndex.value();  // Index of the compute queue family
                barrier.dstQueueFamilyIndex = logicalDevice->graphicsFamilyIndex.value(); // Index of the graphics queue family
                barrier.buffer = particlesBuffer.buffer;
                barrier.offset = 0;
                barrier.size = VK_WHOLE_SIZE;

                vkCmdPipelineBarrier(
                    graphicsCommandBuffer,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                    0,
                    0, nullptr,
                    1, &barrier,
                    0, nullptr
                );
            }
};

void PhysicsWorld::releaseBarrier(VkCommandBuffer graphicsCommandBuffer)
{
    // Release barrier
    if (logicalDevice->graphicsFamilyIndex.value() != logicalDevice->computeFamilyIndex.value())
    {
        VkBufferMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.pNext = nullptr,
        barrier.srcAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
        barrier.dstAccessMask = 0;
        barrier.srcQueueFamilyIndex = logicalDevice->graphicsFamilyIndex.value();
        barrier.dstQueueFamilyIndex = logicalDevice->computeFamilyIndex.value();
        barrier.buffer = particlesBuffer.buffer;
        barrier.offset = 0;
        barrier.size = VK_WHOLE_SIZE;

        vkCmdPipelineBarrier(
            graphicsCommandBuffer,
            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
            0,
            0, nullptr,
            1, &barrier,
            0, nullptr
        );
    }
};

void PhysicsWorld::recordGraphicsCommandBuffer(VkCommandBuffer graphicsCommandBuffer)
{

    if (fpVkCmdBeginConditionalRenderingEXT != nullptr && fpVkCmdEndConditionalRenderingEXT != nullptr){
        fpVkCmdBeginConditionalRenderingEXT(graphicsCommandBuffer, &conditionalRenderingBeginInfo);

            VkViewport viewport = vks::initializers::viewport((float)xcbUI->width, (float)xcbUI->height, 0.0f, 1.0f);
            vkCmdSetViewport(graphicsCommandBuffer, 0, 1, &viewport);

            VkRect2D scissor = vks::initializers::rect2D(xcbUI->width, xcbUI->height, 0, 0);
            vkCmdSetScissor(graphicsCommandBuffer, 0, 1, &scissor);

            vkCmdBindPipeline(graphicsCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeLine.pipeline);
            vkCmdBindDescriptorSets(graphicsCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeLine.pipelineLayout, 0, 1, &graphicsPipeLine.descriptorSet, 0, NULL);

            VkDeviceSize offsets[1] = { 0 };
            if (particlesBuffer.buffer != VK_NULL_HANDLE){
                vkCmdBindVertexBuffers(graphicsCommandBuffer, 0, 1, &particlesBuffer.buffer, offsets);
            }
            else{
                DEBUG_CERR<<"particlesBuffer has null handle"<<std::endl;
            }

            vkCmdDraw(graphicsCommandBuffer, PARTICLE_COUNT, 1, 0, 0);


        fpVkCmdEndConditionalRenderingEXT(graphicsCommandBuffer);

        graphicsNeedRecording = false;
    }
    else{
        DEBUG_CERR<<"cannot start conditional recording"<<std::endl;
    }
}


void PhysicsWorld::handleEvent(KeyMouseEvent KMEvent)
{
    uniformData.deltaT = xcbUI->frameBench.frameTimer * 0.00025f;
    if (!xcbUI->mouse.left_pressed)
    {
        uniformData.destX = 0.0f;
        uniformData.destY = 0.0f;
    }
    else
    {
        float normalizedMx = (xcbUI->mouse.move_x - static_cast<float>(xcbUI->width / 2)) / static_cast<float>(xcbUI->width / 2);
        float normalizedMy = (xcbUI->mouse.move_y - static_cast<float>(xcbUI->height / 2)) / static_cast<float>(xcbUI->height / 2);
        uniformData.destX = normalizedMx;
        uniformData.destY = normalizedMy;
    }

    uniformBuffer.copyTo(&uniformData, sizeof(UniformData));

    if (KMEvent == KEY_F2_RELEASED){
        visibility = (visibility+1)%2;
        setVisibility(visibility);
    }

}
