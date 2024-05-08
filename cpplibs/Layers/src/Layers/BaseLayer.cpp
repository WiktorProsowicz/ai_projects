#include "Layers/BaseLayer.h"

namespace layers
{
BaseLayer::BaseLayer(std::string name)
	: _name(std::move(name))
{}

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
} // namespace layers