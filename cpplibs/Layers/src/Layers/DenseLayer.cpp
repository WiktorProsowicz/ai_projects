#include "Layers/DenseLayer.h"

#include <AutoDiff/Operations.h>
#include <LoggingLib/LoggingLib.hpp>
#include <MLCore/TensorInitializers/GaussianInitializer.hpp>

#include "LayersModules/IActivationFactory.hpp"
#include "Serialization/WeightsSerializer.h"

namespace layers
{
DenseLayer::DenseLayer(std::string name, size_t units, IActivationFactoryPtr activationFactory)
	: BaseLayer(std::move(name))
	, _units(units)
	, _activationFactory(std::move(activationFactory))
{}

autoDiff::OperatorPtr DenseLayer::call(const std::vector<autoDiff::NodePtr>& inputs)
{
	if(!_isBuilt())
	{
		LOG_ERROR("Layers::DenseLayer", "Layer must be built before calling it.");
	}

	if(inputs.size() != 1)
	{
		LOG_ERROR("Layers::DenseLayer", "Dense layer must have exactly one input.");
	}

	const auto output = autoDiff::ops::matmul(
		_weights, inputs[0], mlCore::MatrixSpec::Default, mlCore::MatrixSpec::ColumnVector, true);

	const auto biased = autoDiff::ops::add(output, _bias);

	return _activationFactory->apply(biased);
}

mlCore::TensorShape DenseLayer::getOutputShape() const
{
	if(!_isBuilt())
	{
		LOG_ERROR("Layers::DenseLayer", "Layer must be built before getting output shape.");
	}

	auto outputShape = _inputShape;
	*outputShape.rbegin() = _units;

	return outputShape;
}

std::vector<autoDiff::NodePtr> DenseLayer::getTrainableWeights() const
{
	if(!_isBuilt())
	{
		LOG_ERROR("Layers::DenseLayer", "Layer must be built before getting trainable weights.");
	}

	return {_weights, _bias};
}

std::string DenseLayer::getDescription() const
{
	constexpr const char* descriptionTemplate = R"({} (DenseLayer) Units: {}. Activation: {})";

	return fmt::format(descriptionTemplate, getName(), _units, _activationFactory->getDescription());
}

void DenseLayer::build(const std::vector<mlCore::TensorShape>& inputShapes)
{
	if(inputShapes.size() != 1)
	{
		LOG_ERROR("Layers::DenseLayer", "Dense layer must have exactly one input.");
	}

	const auto& inputShape = inputShapes[0];

	if(inputShape.size() < 1)
	{
		LOG_ERROR("Layers::DenseLayer", "The input shape must have at least one dimension.");
	}

	mlCore::Tensor weightsTensor(mlCore::TensorShape{_units, *inputShape.crbegin()});
	weightsTensor.fill(mlCore::tensorInitializers::GaussianInitializer(0.0, 1.0));

	mlCore::Tensor biasTensor(mlCore::TensorShape{_units});
	biasTensor.fill(mlCore::tensorInitializers::GaussianInitializer(0.0, 1.0));

	_weights = std::make_shared<autoDiff::Variable>(std::move(weightsTensor));
	_bias = std::make_shared<autoDiff::Variable>(std::move(biasTensor));

	_inputShape = inputShape;

	_setBuilt();
}

void DenseLayer::saveWeights(const std::string& path) const
{
	if(!_isBuilt())
	{
		LOG_ERROR("Layers::DenseLayer", "Layer must be built before saving its weights.");
	}

	auto serializer = serialization::WeightsSerializer::open(path);

	const auto& tensorHandles = serializer->getTensorHandles();

	if(tensorHandles.empty())
	{
		serializer->addNewTensor(_weights->getValue());
		serializer->addNewTensor(_bias->getValue());
		return;
	}

	if(tensorHandles.size() != 2)
	{
		LOG_ERROR(
			"Layers::DenseLayer",
			fmt::format("Provided weights file '{}' contains invalid number of tensors! Expected 2, got {}!",
						path,
						tensorHandles.size()));
	}

	tensorHandles[0]->save(_weights->getValue());
	tensorHandles[1]->save(_bias->getValue());
}

void DenseLayer::loadWeights(const std::string& path)
{
	if(!_isBuilt())
	{
		LOG_ERROR("Layers::DenseLayer", "Layer must be built before loading its weights.");
	}

	auto serializer = serialization::WeightsSerializer::open(path);

	const auto& tensorHandles = serializer->getTensorHandles();

	if(tensorHandles.size() != 2)
	{
		LOG_ERROR(
			"Layers::DenseLayer",
			fmt::format("Provided path '{}' contains an invalid number of tensor! Expected 2, found {}!",
						path,
						tensorHandles.size()));
	}

	_setWeight(_weights, tensorHandles[0]->get());
	_setWeight(_bias, tensorHandles[1]->get());
}
} // namespace layers