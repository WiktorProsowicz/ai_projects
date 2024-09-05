#include "Layers/BaseLayer.h"

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

void BaseLayer::_setWeight(const autoDiff::VariablePtr& weight, mlCore::Tensor value)
{
	if(weight->getOutputShape() != value.shape())
	{
		LOG_ERROR("Layers::BaseLayer",
				  fmt::format("Cannot assign a value with shape {} to a weight '{}' with shape {}!",
							  mlCore::stringifyVector(value.shape()),
							  weight->getName(),
							  mlCore::stringifyVector(weight->getOutputShape())));
	}

	weight->setValue(std::move(value));
}
} // namespace layers