# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.0

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /datas/codes/sauron/cmodel/featurebr

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /datas/codes/sauron/cmodel/featurebr/build

# Utility rule file for test_automoc.

# Include the progress variables for this target.
include CMakeFiles/test_automoc.dir/progress.make

CMakeFiles/test_automoc:
	$(CMAKE_COMMAND) -E cmake_progress_report /datas/codes/sauron/cmodel/featurebr/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Automatic moc for target test"
	/usr/bin/cmake -E cmake_autogen /datas/codes/sauron/cmodel/featurebr/build/CMakeFiles/test_automoc.dir/ ""

test_automoc: CMakeFiles/test_automoc
test_automoc: CMakeFiles/test_automoc.dir/build.make
.PHONY : test_automoc

# Rule to build all files generated by this target.
CMakeFiles/test_automoc.dir/build: test_automoc
.PHONY : CMakeFiles/test_automoc.dir/build

CMakeFiles/test_automoc.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_automoc.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_automoc.dir/clean

CMakeFiles/test_automoc.dir/depend:
	cd /datas/codes/sauron/cmodel/featurebr/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /datas/codes/sauron/cmodel/featurebr /datas/codes/sauron/cmodel/featurebr /datas/codes/sauron/cmodel/featurebr/build /datas/codes/sauron/cmodel/featurebr/build /datas/codes/sauron/cmodel/featurebr/build/CMakeFiles/test_automoc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_automoc.dir/depend

