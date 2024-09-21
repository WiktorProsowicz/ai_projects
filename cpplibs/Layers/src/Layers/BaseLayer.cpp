#include "Layers/BaseLayer.h"

#include <fmt/format.h>

namespace layers
{
BaseLayer::BaseLayer(std::string name)
	: _name(std::move(name))
{
	if(_name.empty())
	{
		LOG_ERROR("Layers::BaseLayer", "Layer name cannot be empty.");
	}
}

const std::string& BaseLayer::getName() const
{
	return _name;
}

bool BaseLayer::_isBuilt() const
{
	return _built;
}

void BaseLayer::_setBuilt()
{
	_built = true;
}

namespace
{
std::string stringifyShape(const mlCore::TensorShape& shape)
{
	return fmt::format("({})", fmt::join(shape, ", "));
}
} // namespace

void BaseLayer::_setWeight(const autoDiff::VariablePtr& weight, mlCore::Tensor value)
{
	if(weight->getOutputShape() != value.shape())
	{
		LOG_ERROR("Layers::BaseLayer",
				  fmt::format("Cannot assign a value with shape {} to a weight '{}' with shape {}!",
							  stringifyShape(value.shape()),
							  weight->getName(),
							  stringifyShape(weight->getOutputShape())));
	}

	weight->setValue(std::move(value));
}
} // namespace layers