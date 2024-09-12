#ifndef LAYERS_TESTS_LAYERSTESTFIXTURE_HPP
#define LAYERS_TESTS_LAYERSTESTFIXTURE_HPP

#include <Layers/BaseLayer.h>
#include <gtest/gtest.h>

#include "Utilities.hpp"

namespace testUtilities::fixtures
{
/**
 * @brief Defines workflow for test fixtures testing classes from layers::BaseLayer hierarchy.
 *
 * @details The functions the class defines should be overriden in specific test fixture classes related to
 * concrete BaseLayer subclasses. The basic form of the framework is intended to operate on a hardcoded
 * instance of the tested class. Some functions implicitly define the expected use cases for the concrete
 * class.
 */
class LayersTestFixture : public ::testing::Test
{
protected:
	void SetUp() override
	{
		_layer = _createLayerInstance();
	}

	/// @brief Spawns an instance of the tested class.
	virtual layers::BaseLayerPtr _createLayerInstance() = 0;

	/// @brief Checks if some of the functions of the tested class are prevented from being called before the
	/// layer is built.
	void _testPreventingCallsOnNotBuiltLayer()
	{
		{
			ASSERT_THROW(_layer->getOutputShape(), std::runtime_error);
		}

		{
			ASSERT_THROW(_layer->getTrainableWeights(), std::runtime_error);
		}

		{
			const auto weightsPath = testUtilities::createTempFile();

			ASSERT_THROW(_layer->loadWeights(weightsPath), std::runtime_error);
		}

		{
			const auto weightsPath = testUtilities::createTempFile();

			ASSERT_THROW(_layer->saveWeights(weightsPath), std::runtime_error);
		}

		{
			ASSERT_THROW(_layer->call({nullptr}), std::runtime_error);
		}
	}

	/// @brief Checks if the tested layer properly builds its internal structure based on provided input
	/// shapes and parameters.
	/// @details Can be used to test whether trainable weights are properly set and the output shape is as
	/// expected.
	virtual void _testBuildingLayer() = 0;

	/// @brief Checks if the tested layer properly saves and loads its weights.
	virtual void _testSavingAndLoadingWeights() = 0;

	virtual void _testCalling() = 0;

	layers::BaseLayerPtr _layer{};
};
} // namespace testUtilities::fixtures

#endif