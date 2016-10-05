CXX := clang++
CXXFLAGS := -std=c++11 -O3 -march=native
LDFLAGS := -lboost_program_options

all : mmult
