#include <Layers/DenseLayer.h>
#include <LayersModules/Activations.h>

#include "LayersTestFixture.hpp"
#include "Utilities.hpp"

namespace
{
class TestDenseLayer : public testUtilities::fixtures::LayersTestFixture
{
protected:
	layers::BaseLayerPtr _createLayerInstance() override
	{
		return std::make_shared<layers::DenseLayer>("TestDenseLayer", 10, layers::activations::sigmoid());
	}

	void _testBuildingLayer() override
	{
		const std::vector<mlCore::TensorShape> inputShapes = {mlCore::TensorShape({32, 100})};

		_layer->build(inputShapes);

		ASSERT_EQ(_layer->getOutputShape(), mlCore::TensorShape({32, 10}));

		const auto weights = _layer->getTrainableWeights();

		ASSERT_EQ(weights.size(), 2);

		ASSERT_EQ(weights[0]->getOutputShape(), mlCore::TensorShape({100, 10}));
		ASSERT_EQ(weights[1]->getOutputShape(), mlCore::TensorShape({10}));
	}

	void _testSavingAndLoadingWeights() override
	{
		const std::filesystem::path weightsPath = testUtilities::createTempFile();
		_layer->build({mlCore::TensorShape({32, 100})});
		const auto originalWeights = _layer->getTrainableWeights();

		{
			_layer->saveWeights(weightsPath);

			ASSERT_TRUE(testUtilities::areSavedWeightsAsExpected(
				weightsPath, {originalWeights[0]->getValue(), originalWeights[1]->getValue()}));
		}

		_layer = _createLayerInstance();
		_layer->build({mlCore::TensorShape({32, 100})});

		{
			_layer->loadWeights(weightsPath);
			const auto newWeights = _layer->getTrainableWeights();

			ASSERT_TRUE(
				testUtilities::areTensorsEqual(originalWeights[0]->getValue(), newWeights[0]->getValue()));
			ASSERT_TRUE(
				testUtilities::areTensorsEqual(originalWeights[1]->getValue(), newWeights[1]->getValue()));
		}
	}
};
} // namespace

TEST_F(TestDenseLayer, PreventsCallsOnNotBuiltLayer)
{
	_testPreventingCallsOnNotBuiltLayer();
}

TEST_F(TestDenseLayer, BuildsLayer)
{
	_testBuildingLayer();
}

TEST_F(TestDenseLayer, SavesAndLoadsWeights)
{
	_testSavingAndLoadingWeights();
}