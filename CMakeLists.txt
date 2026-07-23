# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.16.0)
# project information
project(c_ops)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(LMC_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}")
if (NOT GLIBCXX_USE_CXX11_ABI)
  set(GLIBCXX_USE_CXX11_ABI 0)
endif()
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI})

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")

# for the device kernels compilation
set(SOC_VERSION ${SOC_VERSION})
set(ARCH ${ARCH})
# custom compile definition for kernels hostside compilation
set(ASCEND_AICORE_ARCH ${ASCEND_AICORE_ARCH})

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRINGS "Build type Release/Debug (default Release)" FORCE)
endif()

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(ARCH_SUBDIR "aarch64-linux")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(ARCH_SUBDIR "x86_64-linux")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Keep the v0.4.3 kvcache-ops ABI, but make its CANN version checks understand
# CANN 9.  Apply the compatibility edits to a build-tree copy so the nested
# submodule remains pinned and clean.
set(KVCACHE_OPS_SOURCE_DIR "${CMAKE_BINARY_DIR}/kvcache-ops-src")
file(REMOVE_RECURSE "${KVCACHE_OPS_SOURCE_DIR}")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/third_party/kvcache-ops/"
  DESTINATION "${KVCACHE_OPS_SOURCE_DIR}"
  PATTERN ".git" EXCLUDE
)
set(ASCENDC_WITH_DEF "${KVCACHE_OPS_SOURCE_DIR}/ascendc_with_def.cmake")
file(READ "${ASCENDC_WITH_DEF}" ASCENDC_WITH_DEF_CONTENT)
string(REPLACE
  "CANN_VERSION VERSION_EQUAL \"8.3\""
  "CANN_VERSION VERSION_GREATER_EQUAL \"8.3\""
  ASCENDC_WITH_DEF_CONTENT
  "${ASCENDC_WITH_DEF_CONTENT}"
)
string(REPLACE
  "CANN_VERSION VERSION_EQUAL \"8.5\""
  "CANN_VERSION VERSION_GREATER_EQUAL \"8.5\""
  ASCENDC_WITH_DEF_CONTENT
  "${ASCENDC_WITH_DEF_CONTENT}"
)
file(WRITE "${ASCENDC_WITH_DEF}" "${ASCENDC_WITH_DEF_CONTENT}")
add_subdirectory("${KVCACHE_OPS_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/kvcache-ops-build")

# The legacy kvcache-ops helper forwards only its custom INCLUDES property to
# generated device/host projects.  Populate it explicitly for CANN 9 so those
# projects can resolve public AscendC headers.
if(DEFINED ASCEND_AICORE_ARCH)
  ascendc_include_directories(cache_kernels PRIVATE
    ${KVCACHE_OPS_SOURCE_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/asc/impl/adv_api
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/asc/impl/basic_api
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/asc/impl/c_api
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/asc/impl/basic_api/reg_compute
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/asc/impl/simt_api
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/asc/impl/utils
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/asc
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/asc/include
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/asc/include/adv_api
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/asc/include/basic_api
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/asc/include/aicpu_api
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/asc/include/c_api
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/asc/include/basic_api/reg_compute
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/asc/include/simt_api
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/asc/include/utils
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/tikcpp/tikcfw
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/tikcpp/tikcfw/interface
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/tikcpp/tikcfw/impl
  )
endif()

if(DEFINED ENV{USE_MINDSPORE})
  find_package(Threads REQUIRED)
  add_subdirectory(csrc/mindspore)
  target_link_options(c_ops PRIVATE
    "-Wl,-rpath,$ORIGIN:$ORIGIN/lib"
    "-Wl,-rpath,${LMC_INSTALL_PATH}"
  )
  target_link_directories(
    c_ops
    PRIVATE
    ${MINDSPORE_PATH}/lib
    ${MINDSPORE_PATH}/lib/plugin
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/devlib
  )
  target_link_libraries(
    c_ops
    PUBLIC
    mindspore_common
    mindspore_ops
    mindspore_pyboost
    mindspore_frontend
    mindspore_backend_manager
    mindspore_runtime_pipeline
    mindspore_pynative
    pthread
    "${MINDSPORE_PATH}/lib/plugin/libmindspore_ascend.so.2"
    cache_kernels
    ascendcl
    platform
    ascend_hal
    tiling_api
    numa
  )

else()
  add_subdirectory(csrc)
  set(TORCH_LIBS_DIR "${TORCH_PATH}/lib")

  target_link_options(c_ops PRIVATE
    "-Wl,-rpath,$ORIGIN:$ORIGIN/lib"
    "-Wl,-rpath,${LMC_INSTALL_PATH}" 
  )

  target_link_directories(
    c_ops
    PRIVATE
    ${TORCH_LIBS_DIR}
    ${TORCH_NPU_PATH}/lib/
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/devlib
  )

  target_link_libraries(
    c_ops
    PUBLIC
    ${TORCH_LIBRARIES}
    libtorch_npu.so
    cache_kernels
    ascendcl
    platform
    ascend_hal
    tiling_api
    numa
  )
endif()

install(TARGETS c_ops cache_kernels DESTINATION ${LMC_INSTALL_PATH})
