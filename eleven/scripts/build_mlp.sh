g++ -std=c++11 ./src/train_mlp.cpp 	\
			   ./src/tensor/*.cpp 	\
			   ./src/utils/*.cpp 	\
			   ./src/nn/*.cpp		\
			   ./src/models/*.cpp	\
			   ./src/data/*.cpp		\
    -o ./bin/mlp.out
