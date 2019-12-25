g++ -std=c++11 ./src/train_lenet.cpp 	\
			   ./src/tensor/*.cpp 	\
			   ./src/utils/*.cpp 	\
			   ./src/nn/*.cpp		\
			   ./src/models/*.cpp	\
			   ./src/data/*.cpp		\
    -o ./bin/lenet.out
