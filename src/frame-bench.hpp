#pragma once

#include <chrono>

class FrameBench {
public:
    float accumulatedFrameTimer = 0;
    int frameCounter = 0;
    uint32_t lastFPS = 60;

	std::chrono::high_resolution_clock::time_point tBegin;

	float frameTimer = 1.0f;

    // Constructor
    FrameBench() {
        reset();
    }

	void beginFrame()
	{
		tBegin = std::chrono::high_resolution_clock::now();
	}

	void endFrame()
	{
		auto tEnd = std::chrono::high_resolution_clock::now();
		frameTimer = std::chrono::duration<float, std::milli>(tEnd - tBegin).count();
		accumulatedFrameTimer += frameTimer;
		frameCounter++;

		if (accumulatedFrameTimer > 1000.0f) {
            lastFPS = (float)frameCounter * (1000.0f / accumulatedFrameTimer);
			reset();
        }
	}

    // Reset or initialize benchmarking
    void reset() {
        accumulatedFrameTimer = 0;
        frameCounter = 0;
    }
};