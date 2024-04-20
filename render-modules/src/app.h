#include "vulkan-base.h"
#include "particles.h"
#include "UIOverlay.h"
#include "key-codes-status.h"


class App: public VulkanBase
{
private:
    PhysicsWorld world;
    UIOverlay GUI;

    std::vector<vks::VulkanCommandBuffer> graphicsCommandBuffers;
    vks::VulkanCommandBuffer computeCommandBuffer;

protected:
    bool AppPrepared = false;

    //functions needed for the creation of instance and device
    uint32_t rankIndexCombination(std::optional<uint32_t> graphicsIndex, std::optional<uint32_t> transferIndex, std::optional<uint32_t> computeIndex, std::optional<uint32_t> presentIndex) override;
    std::vector<const char*> getInstanceLayers() override;
    std::vector<const char*> getInstanceExtensions() override;
    std::vector<const char*> getDeviceExtensions() override;


    void setupRenderPass();
    void setupFrameBuffer();
    void setupCommandBuffers();

    void recordCommandBuffers();
    void reRecordCommandBuffers();

    void handleEvent(KeyMouseEvent KMEvent);
    void computeDrawSingleFrame();

public:
    App();
    ~App();

    void init() override;
    void deinit() override;
    void prepare() override;
    void unprepare() override;

    void renderLoop() override;

    void test();
};

VULKAN_APP_MAIN()