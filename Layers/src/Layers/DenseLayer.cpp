// __Related headers__
#include <Layers/DenseLayer.h>

namespace layers
{
DenseLayer::DenseLayer(const DenseLayerParams& params)
	: BaseLayer()
	, params_(params)
{
	setGraph(params_.graph);
}

} // namespace layers