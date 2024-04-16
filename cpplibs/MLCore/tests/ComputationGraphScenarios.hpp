#ifndef MLCORE_TESTS_COMPUTATIONGRAPHSCENARIOS_HPP
#define MLCORE_TESTS_COMPUTATIONGRAPHSCENARIOS_HPP

#include <map>
#include <vector>

#include "OperatorStatsUtils.hpp"

namespace mlCoreTests
{
// clang-format off

/// Simpliest architecture with three fully-connected layers.
inline const std::vector<std::pair<std::string, std::string>> treeConfigOneRoot
{
	{"Input", "PLACEHOLDER_(256,1)"},

	{"L1W", "VARIABLE_(200,256)"},
	{"L1B", "VARIABLE_(200,1)"},
	{"Layer1", "MATMUL_L1W_Input"},
	{"Layer1biased", "ADD_Layer1_L1B"},
	{"Layer1Act", "RELU_Layer1biased"},

	{"L2W", "VARIABLE_(200,200)"},
	{"L2B", "VARIABLE_(200,1)"},
	{"Layer2", "MATMUL_L2W_Layer1Act"},
	{"Layer2biased", "ADD_Layer2_L2B"},
	{"Layer2Act", "SIGMOID_Layer2biased"},

	{"L3W", "VARIABLE_(1,200)"},
	{"L3B", "VARIABLE_(1,1)"},
	{"Layer3", "MATMUL_L3W_Layer2Act"},
	{"Layer3biased", "ADD_Layer3_L3B"},
	{"Layer3Act", "SIGMOID_Layer3biased"},

	{"OutputLayer", "LN_Layer3Act"}
};

inline const std::map<mlCoreTests::OperatorStats::WrapperLogChannel, std::vector<std::string>> expectedLogsOneRoot
{
	{mlCoreTests::OperatorStats::WrapperLogChannel::UPDATE_VALUE,
	{"L1W -> Layer1", "Input -> Layer1",
	"Layer1 -> Layer1biased", "L1B -> Layer1biased",
	"Layer1biased -> Layer1Act",

	"L2W -> Layer2", "Layer1Act -> Layer2",
	"Layer2 -> Layer2biased", "L2B -> Layer2biased",
	"Layer2biased -> Layer2Act",

	"L3W -> Layer3", "Layer2Act -> Layer3",
	"Layer3 -> Layer3biased", "L3B -> Layer3biased",
	"Layer3biased -> Layer3Act",

	"Layer3Act -> OutputLayer"}},
	{mlCoreTests::OperatorStats::WrapperLogChannel::COMPUTE_DERIVATIVE,
	{"Layer3Act <- OutputLayer",
	"Layer3biased <- Layer3Act",
	"Layer3 <- Layer3biased", "L3B <- Layer3biased",
	"L3W <- Layer3", "Layer2Act <- Layer3",
	"Layer2biased <- Layer2Act",
	"Layer2 <- Layer2biased", "L2B <- Layer2biased",
	"L2W <- Layer2", "Layer1Act <- Layer2",
	"Layer1biased <- Layer1Act",
	"Layer1 <- Layer1biased", "L1B <- Layer1biased",
	"L1W <- Layer1", "Input <- Layer1"}}
};

/// Architecture with input ports that are connected to a shared layer.
/// The shared layer is connected to two paths, joined at the end.
inline const std::vector<std::pair<std::string, std::string>> treeConfigXShape
{
	// Left input
	{"LeftInput/L1/Input", 		"PLACEHOLDER_(256,1)"},

	{"LeftInput/L1/W", 			"VARIABLE_(200,256)"},
	{"LeftInput/L1/B", 			"VARIABLE_(200,1)"},
	{"LeftInput/L1/Matmul", 	"MATMUL_LeftInput/L1/W_LeftInput/L1/Input"},
	{"LeftInput/L1/Biased", 	"ADD_LeftInput/L1/Matmul_LeftInput/L1/B"},
	{"LeftInput/L1/Act", 		"RELU_LeftInput/L1/Biased"},

	{"LeftInput/L2/W", 			"VARIABLE_(200,200)"},
	{"LeftInput/L2/B", 			"VARIABLE_(200,1)"},
	{"LeftInput/L2/Matmul", 	"MATMUL_LeftInput/L2/W_LeftInput/L1/Act"},
	{"LeftInput/L2/Biased", 	"ADD_LeftInput/L2/Matmul_LeftInput/L2/B"},
	{"LeftInput/L2/Act", 		"RELU_LeftInput/L2/Biased"},

	{"LeftInput/L3/W", 			"VARIABLE_(200,200)"},
	{"LeftInput/L3/B", 			"VARIABLE_(200,1)"},
	{"LeftInput/L3/Matmul", 	"MATMUL_LeftInput/L3/W_LeftInput/L2/Act"},
	{"LeftInput/L3/Biased", 	"ADD_LeftInput/L3/Matmul_LeftInput/L3/B"},
	{"LeftInput/L3/Act", 		"RELU_LeftInput/L3/Biased"},

	// Right input
	{"RightInput/L1/Input", 	"PLACEHOLDER_(256,1)"},

	{"RightInput/L1/W", 		"VARIABLE_(200,256)"},
	{"RightInput/L1/B", 		"VARIABLE_(200,1)"},
	{"RightInput/L1/Matmul", 	"MATMUL_RightInput/L1/W_RightInput/L1/Input"},
	{"RightInput/L1/Biased", 	"ADD_RightInput/L1/Matmul_RightInput/L1/B"},
	{"RightInput/L1/Act", 		"RELU_RightInput/L1/Biased"},

	{"RightInput/L2/W", 		"VARIABLE_(200,200)"},
	{"RightInput/L2/B", 		"VARIABLE_(200,1)"},
	{"RightInput/L2/Matmul", 	"MATMUL_RightInput/L2/W_RightInput/L1/Act"},
	{"RightInput/L2/Biased", 	"ADD_RightInput/L2/Matmul_RightInput/L2/B"},
	{"RightInput/L2/Act", 		"RELU_RightInput/L2/Biased"},

	{"RightInput/L3/W", 		"VARIABLE_(200,200)"},
	{"RightInput/L3/B", 		"VARIABLE_(200,1)"},
	{"RightInput/L3/Matmul", 	"MATMUL_RightInput/L3/W_RightInput/L2/Act"},
	{"RightInput/L3/Biased", 	"ADD_RightInput/L3/Matmul_RightInput/L3/B"},
	{"RightInput/L3/Act", 		"RELU_RightInput/L3/Biased"},

	// Shared layer
	{"SharedLayer/Add",			"ADD_LeftInput/L3/Act_RightInput/L3/Act"},
	{"SharedLayer/Act",			"SIGMOID_SharedLayer/Add"},

	// Left output
	{"LeftOutput/L1/W", 			"VARIABLE_(200,200)"},
	{"LeftOutput/L1/B", 			"VARIABLE_(200,1)"},
	{"LeftOutput/L1/Matmul", 	"MATMUL_LeftOutput/L1/W_SharedLayer/Act"},
	{"LeftOutput/L1/Biased", 	"ADD_LeftOutput/L1/Matmul_LeftOutput/L1/B"},
	{"LeftOutput/L1/Act", 		"RELU_LeftOutput/L1/Biased"},

	{"LeftOutput/L2/W", 			"VARIABLE_(200,200)"},
	{"LeftOutput/L2/B", 			"VARIABLE_(200,1)"},
	{"LeftOutput/L2/Matmul", 	"MATMUL_LeftOutput/L2/W_LeftOutput/L1/Act"},
	{"LeftOutput/L2/Biased", 	"ADD_LeftOutput/L2/Matmul_LeftOutput/L2/B"},
	{"LeftOutput/L2/Act", 		"RELU_LeftOutput/L2/Biased"},

	{"LeftOutput/L3/W", 			"VARIABLE_(200,200)"},
	{"LeftOutput/L3/B", 			"VARIABLE_(200,1)"},
	{"LeftOutput/L3/Matmul", 	"MATMUL_LeftOutput/L3/W_LeftOutput/L2/Act"},
	{"LeftOutput/L3/Biased", 	"ADD_LeftOutput/L3/Matmul_LeftOutput/L3/B"},
	{"LeftOutput/L3/Act", 		"RELU_LeftOutput/L3/Biased"},

	// Right output
	{"RightOutput/L1/W", 		"VARIABLE_(200,200)"},
	{"RightOutput/L1/B", 		"VARIABLE_(200,1)"},
	{"RightOutput/L1/Matmul", 	"MATMUL_RightOutput/L1/W_SharedLayer/Act"},
	{"RightOutput/L1/Biased", 	"ADD_RightOutput/L1/Matmul_RightOutput/L1/B"},
	{"RightOutput/L1/Act", 		"RELU_RightOutput/L1/Biased"},

	{"RightOutput/L2/W", 		"VARIABLE_(200,200)"},
	{"RightOutput/L2/B", 		"VARIABLE_(200,1)"},
	{"RightOutput/L2/Matmul", 	"MATMUL_RightOutput/L2/W_RightOutput/L1/Act"},
	{"RightOutput/L2/Biased", 	"ADD_RightOutput/L2/Matmul_RightOutput/L2/B"},
	{"RightOutput/L2/Act", 		"RELU_RightOutput/L2/Biased"},

	{"RightOutput/L3/W", 		"VARIABLE_(200,200)"},
	{"RightOutput/L3/B", 		"VARIABLE_(200,1)"},
	{"RightOutput/L3/Matmul", 	"MATMUL_RightOutput/L3/W_RightOutput/L2/Act"},
	{"RightOutput/L3/Biased", 	"ADD_RightOutput/L3/Matmul_RightOutput/L3/B"},
	{"RightOutput/L3/Act", 		"RELU_RightOutput/L3/Biased"},

	// Output layer
	{"OutputLayer", "ADD_LeftOutput/L3/Act_RightOutput/L3/Act"}
};

inline const std::map<mlCoreTests::OperatorStats::WrapperLogChannel, std::vector<std::string>> expectedLogsXShape
{
	{mlCoreTests::OperatorStats::WrapperLogChannel::UPDATE_VALUE,
	{"RightInput/L3/Act -> SharedLayer/Add",
	 "LeftInput/L3/Act -> SharedLayer/Add",
	 "SharedLayer/Act -> LeftOutput/L1/Matmul",
	 "SharedLayer/Act -> RightOutput/L1/Matmul",}},
	{mlCoreTests::OperatorStats::WrapperLogChannel::COMPUTE_DERIVATIVE,
	{"SharedLayer/Act <- RightOutput/L1/Matmul",
	 "SharedLayer/Act <- LeftOutput/L1/Matmul",
	 "RightInput/L3/Act <- SharedLayer/Add",
	 "LeftInput/L3/Act <- SharedLayer/Add",
	 "RightInput/L3/Act <- SharedLayer/Add",
	 "LeftInput/L3/Act <- SharedLayer/Add"}}
};

// clang-format on
} // namespace mlCoreTests

#endif