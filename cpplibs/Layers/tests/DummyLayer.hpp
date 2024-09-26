#ifndef LAYERS_TESTS_DUMMYLAYER_HPP
#define LAYERS_TESTS_DUMMYLAYER_HPP

#include <AutoDiff/Operations.h>
#include <Layers/BaseLayer.h>
#include <LoggingLib/LoggingLib.hpp>
#include <MLCore/TensorInitializers/GaussianInitializer.hpp>
#include <Serialization/WeightsSerializer.h>

namespace testUtilities::fixtures
{
/**
 * @brief Dummy layer class for testing. Contains a simple weight vector and adds it to the input.
 */
class DummyLayer : public layers::BaseLayer
{
public:
	DummyLayer() = delete;

	/**
	 * @brief Constructs a new DummyLayer object.
	 *
	 * @param name Name of the layer.
	 * @param nWeights Length of the weight vector.
	 */
	DummyLayer(std::string name, const size_t nWeights)
		: layers::BaseLayer(std::move(name))
		, _nWeights(nWeights)
	{}

	DummyLayer(const DummyLayer&) = delete;
	DummyLayer(DummyLayer&&) = delete;
	DummyLayer& operator=(const DummyLayer&) = delete;
	DummyLayer& operator=(DummyLayer&&) = delete;

	~DummyLayer() override = default;

	autoDiff::OperatorPtr call(const std::vector<autoDiff::NodePtr>& inputs) override
	{
		_assertBuilt();

		if(inputs.size() != 1)
		{
			LOG_ERROR("DummyLayer", "Dummy layer must have exactly one input.");
		}

		return autoDiff::ops::add(inputs[0], _weights);
	}

	mlCore::TensorShape getOutputShape() const override
	{
		_assertBuilt();

		return _outputShape;
	}

	std::vector<autoDiff::NodePtr> getTrainableWeights() const override
	{
		_assertBuilt();

		return {_weights};
	}

	std::string getDescription() const override
	{
		return "DummyLayer";
	}

	void build(const std::vector<mlCore::TensorShape>& inputShapes) override
	{
		if(inputShapes.size() != 1)
		{
			LOG_ERROR("DummyLayer", "Dummy layer must have exactly one input!");
		}

		mlCore::Tensor weightsTensor(mlCore::TensorShape{_nWeights});
		weightsTensor.fill(mlCore::tensorInitializers::GaussianInitializer(0.0, 1.0));

		_weights = std::make_shared<autoDiff::Variable>(std::move(weightsTensor));

		_outputShape = inputShapes[0];

		_setBuilt();
	}

	void saveWeights(const std::string& path) const override
	{
		_assertBuilt();

		const auto serializer = layers::serialization::WeightsSerializer::open(path);

		serializer->addNewTensor(_weights->getValue());
	}

	void loadWeights(const std::string& path) override
	{
		_assertBuilt();

		const auto serializer = layers::serialization::WeightsSerializer::open(path);
		const auto handles = serializer->getTensorHandles();

		if(handles.size() != 1)
		{
			LOG_ERROR("DummyLayer", "Invalid number of weights in the file!");
		}

		_setWeight(_weights, handles[0]->get());
	}

private:
	void _assertBuilt() const
	{
		if(!_isBuilt())
		{
			LOG_ERROR("DummyLayer", "Layer must be built before calling some of its functions!");
		}
	}

	size_t _nWeights;
	mlCore::TensorShape _outputShape{};
	autoDiff::VariablePtr _weights{};
};
} // namespace testUtilities::fixtures

#endif