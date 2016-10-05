// ********** INCLUDES ********** //

#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <utility>
#include <cstdlib>

#include <boost/program_options.hpp>
#define BOOST_DISABLE_ASSERTS 1
#include <boost/multi_array.hpp>

// ********** CONSTANTS ********** //

uint32_t TRIALS {5};
uint32_t SEED {0};
const uint32_t CHECKSUM_MAX {10000};
std::fstream EMPTY_STREAM {"/dev/null"};

// ********** CONSTEXPR UTILITIES ********** //

template <std::uint32_t Index>
class Char {
	const char letter;

public:
	template <std::uint32_t N>
	constexpr Char (const char(&arr)[N]) :	letter (arr[Index]) {
		static_assert (Index < N, "Invalid index");
	}

	constexpr operator char() const {
		return letter;
	}
};

// ********** USING DECLARATIONS ********** //

using ValueType = int;
using Matrix2x2 = boost::multi_array <ValueType, 2>;

using EngineType = std::mt19937_64;
using DistributionType = std::uniform_int_distribution <ValueType>;
using GeneratorType = std::function <ValueType()>;

using ResultKeyType = std::pair <std::uint32_t, std::string>;
using ResultValueType = std::pair <float, ValueType>;
using ResultsType = std::map <ResultKeyType, ResultValueType>;
using FunctionType = std::function <std::uint64_t (const Matrix2x2 &, const Matrix2x2 &, Matrix2x2 &, const std::uint32_t N)>;

// ********** FORWARD FUNCTION DECLARATIONS ********** //

// Prints to passed ostream if flag evaluates to true
std::ostream& conditionalPrint (std::ostream&, bool);

// Prints out results of interest with a callable
template <typename Callable>
std::string print (const std::string &, std::vector <std::string> &, std::vector <std::uint32_t> &, Callable && c);

// Matrix Multiplication
template <char L1, char L2, char L3>
std::uint64_t multiply (const Matrix2x2 &, const Matrix2x2 &, Matrix2x2 &, const std::uint32_t);

// Runs a single configuration
std::pair <float, ValueType> runSingle (FunctionType, const Matrix2x2 &, const Matrix2x2 &, Matrix2x2 &, const std::uint32_t);

// ********** MAIN ********** //

int main (int argc, char* argv[]) {

	// Lists used for iteration
	std::vector <std::string> orderList;
	std::vector <std::uint32_t> sizeList;

	namespace po = boost::program_options;
	po::options_description desc ("Permitted options");
	desc.add_options()
		("help,h", "Print help message")
		("all,a", "Evaluate default dataset (100-500 with all ijk permutations)")
		("iterations,i", po::value <std::uint32_t>(&TRIALS)->default_value(TRIALS),
		 "Number of iterations per invocation")
		("seed,s", po::value <std::uint32_t> (&SEED)->default_value (SEED),
		 "RNG seed for matrix generation")
		("sizes,N", po::value <std::vector <std::uint32_t>> (&sizeList)->multitoken(),
		 "Sizes to evaluate (space separated)")
		("traversals,t", po::value <std::vector <std::string>> (&orderList)->multitoken(),
		 "Traversals to evaluate (space separated)");
	po::variables_map vm;
	po::store (po::parse_command_line (argc, argv, desc), vm);
	po::notify (vm);

	if (vm.count ("help")) {
		std::cout << desc << std::endl;
		std::exit (EXIT_SUCCESS);
	}

	// Check arguments to see if we should automatically run all tests
	const bool RUN_ALL {vm.count ("all") > 0};
	const bool CUSTOM {(vm.count ("sizes") > 0 && vm.count ("traversals") > 0) || RUN_ALL};

	#define CREATE_MAPPING(X) \
  		{ X , &multiply <Char<0>{X}, Char<1>{X}, Char<2>{X}> }

	const std::map <std::string, FunctionType> FUNCTION_MAP = {
		CREATE_MAPPING ("ijk"),
		CREATE_MAPPING ("ikj"),
		CREATE_MAPPING ("jik"),
		CREATE_MAPPING ("jki"),
		CREATE_MAPPING ("kij"),
		CREATE_MAPPING ("kji")
	};

	#undef CREATE_MAPPING

	if (RUN_ALL) {
		sizeList = { 100, 200, 300, 400, 500 };
		std::transform (std::begin (FUNCTION_MAP), std::end (FUNCTION_MAP),
			std::back_inserter (orderList),
			[] (const std::pair <std::string, FunctionType> &p) {
				return std::get <0> (p);
			});
	} else if (!CUSTOM) {

		// read dataset from user
		std::uint32_t N;
		std::string order;
		std::cout << "-- BEGIN INPUT --" << std::endl;
		std::cout << "N     ==> ";
		std::cin >> N;
		std::cout << "Order ==> ";
		std::cin >> order;
		std::cout << "-- END INPUT --" << std::endl;
		sizeList = { N };
		orderList = { order };
	}

	// Initialize RNG
	EngineType eng {SEED};
	DistributionType dist {0, 4};
	GeneratorType gen {std::bind (dist, eng)};

	ResultsType results;

	for (auto N : sizeList) {
		Matrix2x2 A {boost::extents[N][N]};
		Matrix2x2 B {boost::extents[N][N]};
		Matrix2x2 C {boost::extents[N][N]};

		std::generate_n (A.data(), A.num_elements(), gen);
		std::generate_n (B.data(), B.num_elements(), gen);

		for (auto order : orderList) {
			try {
				conditionalPrint (std::cerr, CUSTOM)
					<< "Trials for " << N << " with order " << order << "    " << '\r';

				results.insert ({{N, order}, runSingle (FUNCTION_MAP.at (order), A, B, C, N)});

				conditionalPrint (std::cout, !CUSTOM)
					<< "-- BEGIN OUTPUT --\n"
					<< "Time (us) = " << results[{N, order}].first << "\n"
					<< "Sum       = " << results[{N, order}].second << "\n"
					<< "-- END OUTPUT --" << std::endl;
			} catch (std::out_of_range &ex) {
				std::cerr << "invalid traversal provided: " << order << std::endl;
				std::exit (EXIT_FAILURE);
			}
		}
	}

	conditionalPrint (std::cout, CUSTOM)
		<< "Done!                                " << std::endl
		<< print ("TIMES (MICROSECONDS):", orderList, sizeList,
			[&results] (const ResultKeyType & key) {
				return std::get <0> (results.at (key));
			})
		<< print ("SUMS:", orderList, sizeList,
			[&results] (const ResultKeyType & key) {
				return std::get <1> (results.at (key));
			});

	return EXIT_SUCCESS;
}

ResultValueType
runSingle (FunctionType mmult, const Matrix2x2 & A, const Matrix2x2 & B, Matrix2x2 & C, const std::uint32_t N) {
	std::uint64_t timeSum {0};
	for (std::uint32_t count {0}; count < TRIALS; ++count) {
		std::fill_n (C.data(), C.num_elements(), 0);
		timeSum += mmult (A, B, C, N);
	}
	uint32_t elements = C.num_elements();
	return {
		1.0 * timeSum / TRIALS,
		std::accumulate (C.data(), C.data() + std::min (CHECKSUM_MAX, elements), 0)
	};
}

template <typename Callable>
std::string
print (const std::string & title, std::vector <std::string> & orderList, std::vector <std::uint32_t> & sizeList, Callable && dataAccessor) {
	const static std::uint32_t FP_PRECISION {1};
	const static std::uint32_t HEADING_WIDTH {7};
	const static std::uint32_t DATA_WIDTH {15};

	std::ostringstream oss;

	oss << std::fixed;
	oss.precision (FP_PRECISION);

	oss << "\n\n" << title << "\n\n";
	oss << std::setw (HEADING_WIDTH) << 'N' << ' ';;
	for (auto order : orderList)
		oss << std::setw (DATA_WIDTH) << order << ' ';
	oss << std::endl;
	oss << std::setw (HEADING_WIDTH) << "=====" << ' ';
	for (auto order : orderList)
		oss << std::setw (DATA_WIDTH) << "==========" << ' ';
	oss << std::endl;
	for (auto N : sizeList) {
		oss << std::setw (HEADING_WIDTH) << N << ' ';
		for (auto order : orderList)
			oss << std::setw (DATA_WIDTH) << dataAccessor ({N, order}) << ' ';
		oss << std::endl;
	}
	return oss.str();
}

std::ostream& conditionalPrint (std::ostream & os, bool flag) {
	return (flag ? os : EMPTY_STREAM);
}

template <typename T, T Value>
constexpr std::uint32_t getIndex (std::uint32_t) {
	return 0;
}

template <typename T, T Value, T Other, T ... Args>
constexpr std::uint32_t getIndex (std::uint32_t index) {
	return (Value == Other)
		? index
		: getIndex <T, Value, Args...> (index + 1);
}

#define TO_STR(X) #X

#define _(X) \
	std::get <getIndex <char, Char <0> {TO_STR (X)}, L1, L2, L3> (0)> (std::tie (i, j, k))

template <char L1, char L2, char L3>
std::uint64_t multiply (const Matrix2x2 & A, const Matrix2x2 & B, Matrix2x2 & C, const uint32_t N) {
	auto startTime = std::chrono::high_resolution_clock::now();
	for (std::uint32_t i{0}; i < N; ++i)
		for (std::uint32_t j{0}; j < N; ++j)
			for (std::uint32_t k{0}; k < N; ++k)
				C[_(i)][_(j)] += A[_(i)][_(k)] * B[_(k)][_(j)];
	auto stopTime = std::chrono::high_resolution_clock::now();
	return std::chrono::duration_cast <std::chrono::microseconds> (stopTime - startTime).count();
}

#undef TO_STR
#undef _
