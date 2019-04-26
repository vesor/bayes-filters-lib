
set(TEST_TARGET_NAME test_ceres)

find_package(Ceres REQUIRED)

set(OpenCV_INCLUDE_DIR /usr/local/opencv330/include)
set(OpenCV_LIB_DIR /usr/local/opencv330/lib)
set(OpenCV_LIBS opencv_core opencv_tracking opencv_imgproc opencv_highgui opencv_imgcodecs)

include_directories(
        ${CERES_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIR}
)

link_directories(
        ${OpenCV_LIB_DIR}
)

set(${TEST_TARGET_NAME}_SRC
        main.cpp
)

add_executable(${TEST_TARGET_NAME} ${${TEST_TARGET_NAME}_SRC})

target_link_libraries(${TEST_TARGET_NAME} ${CERES_LIBRARIES} ${OpenCV_LIBS})

add_test(NAME ${TEST_TARGET_NAME}
         COMMAND ${TEST_TARGET_NAME}
         WORKING_DIRECTORY $<TARGET_FILE_DIR:${TEST_TARGET_NAME}>)
