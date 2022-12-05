cmake_minimum_required(VERSION 3.5)
project(speech_commands LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_BUILD_TYPE "Release")

#### specify the compiler flag
SET(CMAKE_CXX_FLAGS  "-std=c++17 -O2")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")

set(SPEECH_COMMANDS_SRC
        speech_commands.cpp)

#set(MNN_ROOT_PATH /mnt/d/Projects/MNN)
link_directories("${MNN_ROOT_PATH}/build/")

add_executable(speech_commands ${SPEECH_COMMANDS_SRC})

target_include_directories(speech_commands PRIVATE ${MNN_ROOT_PATH}/include/ ${MNN_ROOT_PATH}/3rd_party/imageHelper/)
target_link_libraries(speech_commands PRIVATE -lMNN -lcjson -lstdc++ -lpthread -ldl)
#target_link_libraries(speech_commands PRIVATE libMNN.a -lcjson -Wl,--whole-archive -Wl,--no-whole-archive -lstdc++ -lpthread)



set(SPEECH_COMMANDS_ALSA_SRC
        speech_commands_alsa.cpp)

add_executable(speech_commands_alsa ${SPEECH_COMMANDS_ALSA_SRC})


target_include_directories(speech_commands_alsa PRIVATE ${MNN_ROOT_PATH}/include/ ${MNN_ROOT_PATH}/3rd_party/imageHelper/)
target_link_libraries(speech_commands_alsa PRIVATE -lMNN -lasound -lcjson -lstdc++ -lpthread -ldl)
#target_link_libraries(speech_commands_alsa PRIVATE libMNN.a -lasound -lcjson -Wl,--whole-archive -Wl,--no-whole-archive -lstdc++ -lpthread)
