cmake_minimum_required(VERSION 2.8.11)
project(featurebr)

set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(OpenCV REQUIRED)
find_package(OpenBR REQUIRED)

link_directories(/usr/lib/)

find_package(Qt5 5.4.0 REQUIRED Core Gui Xml )

include_directories(${Qt5Widgets_INCLUDE_DIRS} 
                    ${Qt5Core_INCLUDE_DIRS} 
					                    ${Qt5Gui_INCLUDE_DIRS})

add_executable(test main.cpp)

target_link_libraries(test ${OpenCV_LIBS} ${OPENBR_LIBS} Qt5::Core) 
