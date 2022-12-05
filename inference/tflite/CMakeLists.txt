cmake_minimum_required(VERSION 3.5)
project(speech_commands LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_BUILD_TYPE "Release")

#### specify the build flag
SET(CMAKE_CXX_FLAGS  "-std=c++17 -O2")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")
SET(TARGET_PLAT "linux_x86_64" CACHE STRING INTERNAL)

#set(TF_ROOT_PATH /mnt/e/Downloads/tensorflow)
link_directories("${TF_ROOT_PATH}/tensorflow/lite/tools/make/gen/${TARGET_PLAT}/lib/")

set(SPEECH_COMMANDS_SRC
        speech_commands.cpp)

add_executable(speech_commands ${SPEECH_COMMANDS_SRC})

target_include_directories(speech_commands PRIVATE ${TF_ROOT_PATH} ${TF_ROOT_PATH}/tensorflow/lite/tools/make/downloads/flatbuffers/include)
target_link_libraries(speech_commands PRIVATE libtensorflow-lite.a -lcjson -lstdc++ -lpthread -lm -ldl -lrt)
#target_link_libraries(speech_commands PRIVATE -ltensorflow-lite -lcjson -lstdc++ -lpthread -lm -ldl -lrt)


set(SPEECH_COMMANDS_ALSA_SRC
        speech_commands_alsa.cpp)

add_executable(speech_commands_alsa ${SPEECH_COMMANDS_ALSA_SRC})

target_include_directories(speech_commands_alsa PRIVATE ${TF_ROOT_PATH} ${TF_ROOT_PATH}/tensorflow/lite/tools/make/downloads/flatbuffers/include)
target_link_libraries(speech_commands_alsa PRIVATE libtensorflow-lite.a -lasound -lcjson -lstdc++ -lpthread -lm -ldl -lrt)
#target_link_libraries(speech_commands_alsa PRIVATE -ltensorflow-lite -lasound -lcjson -lstdc++ -lpthread -lm -ldl -lrt)
