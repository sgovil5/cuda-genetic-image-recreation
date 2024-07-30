OPENCV_DIR = C:/Users/shiti/Downloads/opencv/build

all: genetic

genetic: main.cu visualize.cu initialize.cu random_utils.cu fitness.cu
	nvcc main.cu visualize.cu initialize.cu random_utils.cu fitness.cu -o genetic -I"$(OPENCV_DIR)/include" -L"$(OPENCV_DIR)/x64/vc16/lib" -lopencv_world4100 -Xcudafe --diag_suppress=611 -Xcudafe --diag_suppress=550

clean:
	del genetic.exe