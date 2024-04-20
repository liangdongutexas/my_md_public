
include_directories(${XCB_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/../vulkan-base/src ${CMAKE_CURRENT_SOURCE_DIR}/../vulkan-base/external/ktx/include ${CMAKE_CURRENT_SOURCE_DIR}/../vulkan-base/external/imgui)

file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*cpp)

add_executable(App ${SRC})

target_link_libraries(App PRIVATE Vulkan::Vulkan ${XCB_LIBRARIES})