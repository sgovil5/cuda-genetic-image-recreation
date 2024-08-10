OPENCV_DIR = C:/Users/shiti/Downloads/opencv/build
CUDA_PATH = C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1

all: genetic

genetic: main.cu random_utils.cu visualize.cu initialize.cu fitness.cu tournament.cu
	nvcc main.cu random_utils.cu visualize.cu initialize.cu fitness.cu tournament.cu -o genetic -I"$(OPENCV_DIR)/include" -L"$(OPENCV_DIR)/x64/vc16/lib" -lopencv_world4100 -Xcudafe --diag_suppress=611 -Xcudafe --diag_suppress=550

clean:
	del genetic.exe