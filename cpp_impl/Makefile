CXX = g++
CXXFLAGS = -std=c++11 -Wall

genetic_cpp: main.o visualize.o stb_impl.o
	$(CXX) $(CXXFLAGS) main.o visualize.o stb_impl.o -o genetic_cpp

main.o: main.cpp main.h
	$(CXX) $(CXXFLAGS) -c main.cpp

visualize.o: visualize.cpp main.h
	$(CXX) $(CXXFLAGS) -c visualize.cpp

stb_impl.o: stb_impl.cpp
	$(CXX) $(CXXFLAGS) -c stb_impl.cpp

clean:
	-del /Q *.o genetic_cpp.exe