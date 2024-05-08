#include "Layers/DenseLayer.h"

#include <AutoDiff/Operations.h>
#include <LoggingLib/LoggingLib.hpp>
#include <MLCore/TensorInitializers/GaussianInitializer.hpp>

#include "LayersModules/IActivationFactory.hpp"

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

	const auto output = autoDiff::ops::matmul(_weights, inputs[0]);
	const auto biased = autoDiff::ops::add(output, _bias);
	auto activated = _activationFactory->apply(biased);

	activated->updateValue();

	return activated;
}

mlCore::TensorShape DenseLayer::getOutputShape() const
{
	if(!_isBuilt())
	{
		LOG_ERROR("Layers::DenseLayer", "Layer must be built before getting output shape.");
	}

	return {_units, 1};
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

	if(inputShape.size() != 2 || inputShape[1] != 1)
	{
		LOG_ERROR("Layers::DenseLayer", "Input shape must be a column vector.");
	}

	mlCore::Tensor weightsTensor({_units, inputShape[0]});
	weightsTensor.fill(mlCore::tensorInitializers::GaussianInitializer(0.0, 0.1));

	mlCore::Tensor biasTensor({_units, 1});
	biasTensor.fill(mlCore::tensorInitializers::GaussianInitializer(0.0, 0.1));

	_weights = std::make_shared<autoDiff::Variable>(std::move(weightsTensor));
	_bias = std::make_shared<autoDiff::Variable>(std::move(biasTensor));

	_setBuilt();
}
} // namespace layers