#include "vulkan-base.h"


/**
 * @brief distinction between init and prepare is that init contains set up that will not be changed during the life of the app
 * 		  while operations in prepare may be recalled dynamically during the running of the app
 *
 */
void VulkanBase::init()
{
	setupDisplay();

	createInstanceDebug();

	//pick physical logicalDevice->device
	if (!pickPhysicalDevice()){
		throw std::runtime_error("couldn't find a physical device that has all required functionalities");
	}

	createLogicalDevice();
};


void VulkanBase::prepare(){
	if (!basePrepared){
		setupSwapChain();
		setupDepthStencil();
		basePrepared = true;
	}
};

void VulkanBase::unprepare()
{
	// Ensure all operations on the logicalDevice->device have been finished before destroying
	vkDeviceWaitIdle(logicalDevice->device);

	if (basePrepared){
		DEBUG_COUT << "Destroying depth stencil." << std::endl;
		depthStencil.destroy();

		DEBUG_COUT << "Destroying swapchain." << std::endl;
		swapchain.destroy();
		basePrepared = false;
	}
};

void VulkanBase::deinit()
{
	// Ensure all operations on the logicalDevice->device have been finished before destroying
	vkDeviceWaitIdle(logicalDevice->device);

	DEBUG_COUT << "Destroying logical device." << std::endl;
	logicalDevice->destroy();

#ifdef DEBUG
	vks::debug::freeDebugCallback(instance);
#endif
    // Instance destruction is often handled separately, as it might be needed after logicalDevice->device destruction for final cleanups.
    // But if you decide to include it here:
    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
    }

	xcbUI->destroy();
};


/**
 * @brief typedef struct VkInstanceCreateInfo
 * 		  {
			VkStructureType                 sType;
			const void*						pNext;
			VkInstanceCreateFlags			flags;
			const VkApplicationInfo*		pApplicationInfo;
			uint32_t						enabledLayerCount;
			const char* const*				ppEnabledLayerNames;
			uint32_t						enabledExtensionCount;
			const char* const*				ppEnabledExtensionNames;
		   } VkInstanceCreateInfo;
 */
void VulkanBase::createInstanceDebug()
{
	VkInstanceCreateInfo instanceCreateInfo = {};
	instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

	//App information
	VkApplicationInfo appInfo = {};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = AppName.c_str();
	appInfo.pEngineName = AppName.c_str();
	//Dynamically query vulkan version
	uint32_t apiVersion=0;
	VK_CHECK_RESULT(vkEnumerateInstanceVersion(&apiVersion));
	appInfo.apiVersion = apiVersion;
	//Assign app info
	instanceCreateInfo.pApplicationInfo = &appInfo;

	std::vector<const char*> instanceExtensionsRequired = getInstanceExtensions();

	//Check availability of required extensions
	// Get extensions supported by the instance
    uint32_t extensionCount = 0;
    VK_CHECK_RESULT(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr));
    std::vector<VkExtensionProperties> extensionsAvailable(extensionCount);
    VK_CHECK_RESULT(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensionsAvailable.data()));

	// Check each required extension against available extensions
    for (const auto extensionRequired : instanceExtensionsRequired) {
        bool isAvailable = false;
        for (const auto& extensionAvailable : extensionsAvailable) {
            if (strcmp(extensionRequired, extensionAvailable.extensionName) == 0) {
                isAvailable = true;
                break;
            }
        }
        if (!isAvailable) {
			throw std::runtime_error("Required instance extension "+ std::string(extensionRequired)+ " is not available");
        }
    }

	instanceCreateInfo.enabledExtensionCount = (uint32_t)instanceExtensionsRequired.size();
	instanceCreateInfo.ppEnabledExtensionNames = instanceExtensionsRequired.size()? instanceExtensionsRequired.data(): nullptr;


	std::vector<const char*> layersRequired = getInstanceLayers();
	//Check availability of required layers
	// Check if this layer is available at instance level
	uint32_t instanceLayerCount;
	VK_CHECK_RESULT(vkEnumerateInstanceLayerProperties(&instanceLayerCount, nullptr));
	std::vector<VkLayerProperties> layersAvailable(instanceLayerCount);
	VK_CHECK_RESULT(vkEnumerateInstanceLayerProperties(&instanceLayerCount, layersAvailable.data()));

	// Check each required layer against available extensions
	for (const auto layerRequired : layersRequired) {
		bool isAvailable = false;
		for (const auto& layerAvailable : layersAvailable) {
			if (strcmp(layerRequired, layerAvailable.layerName) == 0) {
				isAvailable = true;
				break;
			}
		}
		if (!isAvailable) {
			throw std::runtime_error("Required instance layer"+ std::string(layerRequired)+ "is not available");
		}
	}

	instanceCreateInfo.enabledLayerCount = (uint32_t)layersRequired.size();
	instanceCreateInfo.ppEnabledLayerNames = layersRequired.size()? layersRequired.data() : nullptr;


	VK_CHECK_RESULT(vkCreateInstance(&instanceCreateInfo, nullptr, &instance));

	// If requested, we enable the default validation layers for debugging
#ifdef DEBUG
	vks::debugutils::setup(instance);
	vks::debug::setupDebugging(instance);
#endif
	swapchain.instance = instance;
	swapchain.initSurface(xcbUI->connection, xcbUI->window);
}

/**
 * @brief a very basic pick the first physical logicalDevice->device that have the functionality we needed. A more complex pick algorithm can be
 *        implemented in the future
 */
bool VulkanBase::pickPhysicalDevice()
{
	// Enumerate all physical logicalDevice->device
	uint32_t gpuCount = 0;
	// Get number of available physical logicalDevice->devices
	VK_CHECK_RESULT(vkEnumeratePhysicalDevices(instance, &gpuCount, nullptr));
	if (gpuCount == 0) {
		vks::tools::exitFatal("No logicalDevice->device with Vulkan support found", -1);
		return false;
	}
	// Enumerate logicalDevice->devices
	std::vector<VkPhysicalDevice> physicalDevices(gpuCount);
	VkResult err = vkEnumeratePhysicalDevices(instance, &gpuCount, physicalDevices.data());
	if (err) {
		vks::tools::exitFatal("Could not enumerate physical logicalDevice->devices : \n" + vks::tools::errorString(err), err);
		return false;
	}

	//find the first suitable device;
	//if there are more than one suitable ldevices further comparison is needed but not implemented here
	for (const auto& potentialPhysicalDevice : physicalDevices) {
		if (isDeviceSuitable(potentialPhysicalDevice)) {
			physicalDevice->setPhysicalDevice(potentialPhysicalDevice);
			return true;
		}
	}

	return false;
}

/** @brief three aspects need to be checked 1. queue family support; 2. swapchain compatibility 3. extension support;
 * 		   the first two test is by let vks:: VulkanPhysicalDevice and
*/
bool VulkanBase::isDeviceSuitable(VkPhysicalDevice potentialPhysicalDevice)
{
	//try on potential physical logicalDevice->device by vks:VulkanPhysicalDevice
	physicalDevice->setPhysicalDevice(potentialPhysicalDevice);
	//get potential indices
	std::set<std::optional<uint32_t>> graphicsIndices;
	std::set<std::optional<uint32_t>> transferIndices;
	std::set<std::optional<uint32_t>> computeIndices;
	std::set<std::optional<uint32_t>> presentIndices;

	//find queue family indices that is required for graphic, transfer, compute.
	//The returned set at least contains std::nullopt to denote no queue family is found
	graphicsIndices = physicalDevice->getQueueFamilyIndices(VK_QUEUE_GRAPHICS_BIT);
	transferIndices = physicalDevice->getQueueFamilyIndices(VK_QUEUE_TRANSFER_BIT);
	computeIndices = physicalDevice->getQueueFamilyIndices(VK_QUEUE_COMPUTE_BIT);
	//try on potential physical logicalDevice->device by vks:SwapChain
	swapchain.setPhysicalDevice();
	presentIndices = swapchain.getPresentQueueIndices();


	//try to find the best combination of indices according to a rank function: rankIndexCombination
	for (std::optional<uint32_t> transferIndex:transferIndices){
		for (std::optional<uint32_t> graphicsIndex:graphicsIndices){
			for (std::optional<uint32_t> computeIndex:computeIndices){
				for (std::optional<uint32_t> presentIndex:presentIndices){
					if (rankIndexCombination(graphicsIndex, transferIndex, computeIndex, presentIndex)>
					rankIndexCombination(logicalDevice->graphicsFamilyIndex, logicalDevice->transferFamilyIndex, logicalDevice->computeFamilyIndex, logicalDevice->presentFamilyIndex)){
						logicalDevice->graphicsFamilyIndex=graphicsIndex;
						logicalDevice->computeFamilyIndex=computeIndex;
						logicalDevice->transferFamilyIndex=transferIndex;
						logicalDevice->presentFamilyIndex=presentIndex;
					};
				}
			}
		}
	}

	/*
	for (const auto& extensionAvailable : physicalDevice->supportedExtensions) {
		DEBUG_CERR<< ("Available extension: "+ extensionAvailable);
	};
	*/

	std::vector<const char*> deviceExtensionsRequired = getDeviceExtensions();
	//check required logicalDevice extensions are available
	for (const auto extensionRequired : deviceExtensionsRequired) {
		bool isAvailable = false;
		for (const auto& extensionAvailable : physicalDevice->supportedExtensions) {
			if (strcmp(extensionRequired, extensionAvailable.data()) == 0) {
				isAvailable = true;
				break;
			}
		}
		if (!isAvailable) {
			DEBUG_CERR<< ("Required extension "+ std::string(extensionRequired)+ " is not available");
			return false;
		}
	}

	// Find a suitable depth and/or stencil format
	VkBool32 validFormat{ false };
	// Samples that make use of stencil will require a depth + stencil format, so we select from a different list
	if (requiresStencil) {
		validFormat = vks::tools::getSupportedDepthStencilFormat((*physicalDevice)(), &depthStencil.format);
	} else {
		validFormat = vks::tools::getSupportedDepthFormat((*physicalDevice)(), &depthStencil.format);
	}


	//if extension not available false is already returned
	return (rankIndexCombination(logicalDevice->graphicsFamilyIndex, logicalDevice->transferFamilyIndex, logicalDevice->computeFamilyIndex, logicalDevice->presentFamilyIndex)>0) && swapchain.isDetailsComplete() && validFormat;
}

/**
* Create the logical logicalDevice->device based on the assigned physical logicalDevice->device, also gets default queue family indices
* @brief typedef struct VkDeviceCreateInfo {
		VkStructureType					sType;
		const void*						pNext;
		VkDeviceCreateFlags				flags;
		uint32_t						queueCreateInfoCount;
		const VkDeviceQueueCreateInfo*  pQueueCreateInfos;
		uint32_t						enabledLayerCount;
		const char* const*				ppEnabledLayerNames;
		uint32_t						enabledExtensionCount;
		const char* const*				ppEnabledExtensionNames;
		const VkPhysicalDeviceFeatures* pEnabledFeatures;
		} VkDeviceCreateInfo;
* @param enabledFeatures Can be used to enable certain features upon logicalDevice->device creation
* @param pNextChain Optional chain of pointer to extension structures
* @param useSwapChain Set to false for headless rendering to omit the swapchain logicalDevice->device extensions
*
* @return VkResult of the logicalDevice->device creation call
*/
void VulkanBase::createLogicalDevice()
{
	// Desired queues need to be requested upon logical logicalDevice->device creation
	// Due to differing queue family configurations of Vulkan implementations this can be a bit tricky, especially if the application requests different queue types
	// update the queue indices by the picked physical logicalDevice->device again
	isDeviceSuitable((*physicalDevice)());

	std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
	std::set<uint32_t> uniqueQueueFamilies = {logicalDevice->graphicsFamilyIndex.value(), logicalDevice->transferFamilyIndex.value(), logicalDevice->computeFamilyIndex.value(), logicalDevice->presentFamilyIndex.value()};

	float queuePriority = 1.0f;
	for (uint32_t queueFamily : uniqueQueueFamilies) {
		VkDeviceQueueCreateInfo queueCreateInfo{};
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = queueFamily;
		queueCreateInfo.queueCount = 1;
		queueCreateInfo.pQueuePriorities = &queuePriority;
		queueCreateInfos.push_back(queueCreateInfo);
	}


	std::vector<const char*> deviceExtensionsRequired = getDeviceExtensions();

	VkDeviceCreateInfo deviceCreateInfo = {};
	deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
	deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();

	deviceCreateInfo.pEnabledFeatures = &physicalDevice->features;

	deviceCreateInfo.enabledExtensionCount = (uint32_t)deviceExtensionsRequired.size();
	deviceCreateInfo.ppEnabledExtensionNames = deviceExtensionsRequired.size()? deviceExtensionsRequired.data() : nullptr;

	VK_CHECK_RESULT(vkCreateDevice((*physicalDevice)(), &deviceCreateInfo, nullptr, &logicalDevice->device));

	logicalDevice->createQueueCommandPool();
}

void VulkanBase::setupDisplay()
{
	//create surface if direct display is enabled
	xcbUI->initxcbConnection();
	xcbUI->setupWindow(title);
}

void VulkanBase::setupSwapChain()
{
	swapchain.destroy();
	swapchain.instance=instance;
	swapchain.initSurface(xcbUI->connection, xcbUI->window);
	//connect physical logicalDevice->device and logical logicalDevice->device to
	swapchain.setPhysicalDevice();
	swapchain.create(&xcbUI->width, &xcbUI->height);

};

void VulkanBase::setupDepthStencil()
{
	depthStencil.destroy();

	VkImageCreateInfo imageCI{};
	imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageCI.imageType = VK_IMAGE_TYPE_2D;
	imageCI.format = depthStencil.format;
	imageCI.extent = { xcbUI->width, xcbUI->height, 1 };
	imageCI.mipLevels = 1;
	imageCI.arrayLayers = 1;
	imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
	imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

	depthStencil.allocateImageMemory(imageCI, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
	// Stencil aspect should only be set on depth + stencil formats (VK_FORMAT_D16_UNORM_S8_UINT..VK_FORMAT_D32_SFLOAT_S8_UINT
	if (depthStencil.format >= VK_FORMAT_D16_UNORM_S8_UINT) {
		aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
	}
	VkImageViewCreateInfo viewCreateInfo;
	viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewCreateInfo.flags = 0;
	viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewCreateInfo.format = depthStencil.format;
	viewCreateInfo.subresourceRange = {aspectMask, 0, 1, 0, 1 };
	viewCreateInfo.components = {
	VK_COMPONENT_SWIZZLE_IDENTITY,
	VK_COMPONENT_SWIZZLE_IDENTITY,
	VK_COMPONENT_SWIZZLE_IDENTITY,
	VK_COMPONENT_SWIZZLE_IDENTITY
};
	depthStencil.createView(viewCreateInfo);
}












