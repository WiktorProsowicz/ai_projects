// __Related headers__
#include <Layers/BaseLayer.h>

namespace layers
{
BaseLayer::BaseLayer()
	: inputLayer_(nullptr)
	, graph_(nullptr)
	, name_()
	, built_(false)
{ }

mlCore::models::ILayerPtr BaseLayer::getInputLayer() const
{
	return inputLayer_;
}

void BaseLayer::setName(const std::string& name)
{
	name_ = name;
}

const std::string& BaseLayer::getName() const
{
	return name_;
}

std::shared_ptr<mlCore::autoDiff::ComputationGraph> BaseLayer::getGraph() const
{
	return graph_;
}
} // namespace layers