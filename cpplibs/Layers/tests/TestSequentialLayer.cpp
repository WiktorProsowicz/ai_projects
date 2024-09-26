#include <Layers/SequentialLayer.h>

#include "DummyLayer.hpp"
#include "LayersTestFixture.hpp"
#include "Utilities.hpp"

namespace
{
class TestSequentialLayer : public testUtilities::fixtures::LayersTestFixture
{
protected:
	layers::BaseLayerPtr _createLayerInstance() override
	{
		return std::make_shared<layers::SequentialLayer>(
			"TestSequentialLayer",
			std::vector<layers::BaseLayerPtr>{
				std::make_shared<testUtilities::fixtures::DummyLayer>("DummyLayer1", 10),
				std::make_shared<testUtilities::fixtures::DummyLayer>("DummyLayer2", 10),
				std::make_shared<testUtilities::fixtures::DummyLayer>("DummyLayer3", 10),
			});
	}

	void _testBuildingLayer() override
	{
		const std::vector<mlCore::TensorShape> inputShapes = {mlCore::TensorShape({32, 10})};

		_layer->build(inputShapes);

		ASSERT_EQ(_layer->getOutputShape(), mlCore::TensorShape({32, 10}));

		const auto weights = _layer->getTrainableWeights();

		ASSERT_EQ(weights.size(), 3);

		for(size_t i = 0; i < 3; ++i)
		{
			ASSERT_EQ(weights[i]->getOutputShape(), mlCore::TensorShape({10}));
		}
	}

	void _testSavingAndLoadingWeights() override
	{
		const std::filesystem::path weightsPath = testUtilities::createTempFile();
		_layer->build({mlCore::TensorShape({32, 10})});
		const auto originalWeights = _layer->getTrainableWeights();

		{
			_layer->saveWeights(weightsPath);

			ASSERT_TRUE(testUtilities::areSavedWeightsAsExpected(weightsPath / "DummyLayer1",
																 {originalWeights[0]->getValue()}));

			ASSERT_TRUE(testUtilities::areSavedWeightsAsExpected(weightsPath / "DummyLayer2",
																 {originalWeights[1]->getValue()}));

			ASSERT_TRUE(testUtilities::areSavedWeightsAsExpected(weightsPath / "DummyLayer3",
																 {originalWeights[2]->getValue()}));
		}

		_layer = _createLayerInstance();
		_layer->build({mlCore::TensorShape({32, 10})});

		{
			_layer->loadWeights(weightsPath);
			const auto newWeights = _layer->getTrainableWeights();

			for(size_t i = 0; i < 3; ++i)
			{
				ASSERT_TRUE(testUtilities::areTensorsEqual(originalWeights[i]->getValue(),
														   newWeights[i]->getValue()));
			}
		}
	}

	void _testCalling() override
	{
		const auto input = std::make_shared<autoDiff::Constant>(mlCore::Tensor{mlCore::TensorShape{32, 100}});

		_layer->build({input->getOutputShape()});
		const auto output = _layer->call({input});

		const mlCore::TensorShape expectedOutputShape{32, 100};
		ASSERT_EQ(output->getOutputShape(), expectedOutputShape);
	}
};
} // namespace

TEST_F(TestSequentialLayer, PreventsCallsOnNotBuiltLayer)
{
	_testPreventingCallsOnNotBuiltLayer();
}

TEST_F(TestSequentialLayer, BuildsLayer)
{
	_testBuildingLayer();
}

TEST_F(TestSequentialLayer, SavesAndLoadsWeights)
{
	_testSavingAndLoadingWeights();
}

TEST_F(TestSequentialLayer, IsCalledCorrectly)
{
	_testCalling();
}