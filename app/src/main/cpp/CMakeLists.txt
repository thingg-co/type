cmake_minimum_required(VERSION 3.22)
project(board_llm LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# llama.cpp is vendored as a shallow clone under third_party/. It is built as static
# libraries (BUILD_SHARED_LIBS=OFF, set from app/build.gradle.kts) and folded into the
# single libboard_llm.so that the Kotlin side loads.
set(LLAMA_SRC ${CMAKE_CURRENT_LIST_DIR}/../../../../third_party/llama.cpp)
add_subdirectory(${LLAMA_SRC} llama-build EXCLUDE_FROM_ALL)

add_library(board_llm SHARED llm_jni.cpp)

target_include_directories(board_llm PRIVATE
        ${LLAMA_SRC}/include
        ${LLAMA_SRC}/ggml/include)

target_compile_options(board_llm PRIVATE -O2 -fvisibility=hidden)

target_link_libraries(board_llm PRIVATE llama android log)
