#ifndef LAYERS_INCLUDE_LAYERS_DENSELAYER_H
#define LAYERS_INCLUDE_LAYERS_DENSELAYER_H

// __Own software headers__
#include <Layers/BaseLayer.h>

namespace layers
{
/**
 * @brief Contains parameters describing internal configuration of the dense layer.
 * 
 */
struct DenseLayerParams
{
	size_t nUnits;

	std::shared_ptr<mlCore::autoDiff::ComputationGraph> graph;
};

/**
 * @brief Basic lowest-level linear layer.
 * 
 * Used for creating fully connected parts of the layer-chains. Consists of a single
 * sequence of neurons, each of them connected with each input neuron. 
 * 
 */
class DenseLayer : public layers::BaseLayer
{
public:
	DenseLayer() = delete;

	DenseLayer(const DenseLayerParams& params);

	DenseLayer(const DenseLayer&) = delete;			   // Copy constructor.
	DenseLayer(DenseLayer&&) = delete;				   // Move constructor.
	DenseLayer& operator=(const DenseLayer&) = delete; // Copy assignment.
	DenseLayer& operator=(DenseLayer&&) = delete;	   // Move assignment.

	~DenseLayer() override = default; // Default destructor.

	void setInputLayer(const mlCore::models::ILayerPtr& inputLayer) override;

	void setGraph(const std::shared_ptr<mlCore::autoDiff::ComputationGraph>& graph) override;

	mlCore::autoDiff::NodePtr build() override;

	mlCore::Tensor compute() override;

	std::vector<mlCore::autoDiff::NodePtr> getAllWeights() const override;

	std::vector<mlCore::autoDiff::NodePtr> getTrainableWeights() const override;

	std::string getDescription() const override;

private:
	static inline size_t layersCounter_ = 0;

	DenseLayerParams params_;
};
} // namespace layers

#endif