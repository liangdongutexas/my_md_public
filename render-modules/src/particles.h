#pragma once


#include <random>
#include <glm/glm.hpp>
#include <stdexcept>

#include "app-module.h"
#include "vulkan-pipeline.h"
#include "vulkan-texture.h"
#include "vulkan-buffer.h"
#include "camera.hpp"
#include "xcbUI.h"


#define PARTICLE_COUNT 2560 * 1024






class PhysicsWorld: public AppModule
{
private:
    bool firstRecord = true;

    XcbUI* xcbUI = XcbUI::getXcbUI();

    struct {
    vks::Texture particle;
    vks::Texture gradient;
    } textures;

    // SSBO particle declaration
    struct Particle {
        glm::vec2 pos;								// Particle position
        glm::vec2 vel;								// Particle velocity
        glm::vec4 gradientPos;						// Texture coordinates for the gradient ramp map
    };
    // We use a shader storage buffer object to store the particlces
	// This is updated by the compute pipeline and displayed as a vertex buffer by the graphics pipeline
	vks::Buffer particlesBuffer;
    vks::Buffer uniformBuffer;					// Uniform buffer object containing particle system parameters
    struct UniformData {						// Compute shader uniform block object
    float deltaT;							    //		Frame delta time
    float destX;							    //		x position of the attractor
    float destY;							    //		y position of the attractor
    int32_t particleCount = PARTICLE_COUNT;
    } uniformData;


    Camera camera;

    //to be called by the prepareMemoryResources() method
    void prepareMemoryBuffers();
    void loadAssets();

    //to be called by the preparePipeline() method
    void prepareComputePipeLine();
    void prepareGraphicsPipeLine();

    //called by the handleEvent() method
    void updateUniformBuffers();

public:
    bool computeNeedRecording = true;
    bool graphicsNeedRecording = true;
    vks::PipeLine graphicsPipeLine;
    vks::PipeLine computePipeLine;


    void init() override;
    void deinit() override;
    void prepare() override;
    void unprepare() override;


    //to be called by the recordCommandBuffers() method
    void recordComputeCommandBuffer(VkCommandBuffer computeCommandBuffer);
    void acquireBarrier(VkCommandBuffer graphicsCommandBuffer);
    void releaseBarrier(VkCommandBuffer graphicsCommandBuffer);
    void recordGraphicsCommandBuffer(VkCommandBuffer graphicsCommandBuffer);

    void handleEvent(KeyMouseEvent KMEvent) override;
};


