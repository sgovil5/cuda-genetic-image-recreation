OPENCV_DIR = C:/Users/shiti/Downloads/opencv/build

all: output_program

output_program: main.cu
	nvcc main.cu -o output_program -I"$(OPENCV_DIR)/include" -L"$(OPENCV_DIR)/x64/vc16/lib" -lopencv_world4100 -Xcudafe --diag_suppress=611

clean:
	del output_program.exe