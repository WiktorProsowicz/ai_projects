#ifndef LAYERS_INCLUDE_LAYERS_SEQUENTIALLAYER_H
#define LAYERS_INCLUDE_LAYERS_SEQUENTIALLAYER_H

#include "Layers/BaseLayer.h"

namespace layers
{
/**
 * @brief Implements a layer that is a sequence of other layers.
 *
 * @details The sequential layer is a container for other layers. The output of each layer is the input of the
 * next layer in the sequence. The sequential layer is a way to create a neural network model by stacking
 * layers on top of each other.
 */
class SequentialLayer : public BaseLayer
{
public:
	SequentialLayer() = delete;

	SequentialLayer(std::string name, std::vector<BaseLayerPtr> layers);

	SequentialLayer(const SequentialLayer&) = delete;
	SequentialLayer(SequentialLayer&&) = delete;
	SequentialLayer& operator=(const SequentialLayer&) = delete;
	SequentialLayer& operator=(SequentialLayer&&) = delete;

	~SequentialLayer() override = default;

	autoDiff::OperatorPtr call(const std::vector<autoDiff::NodePtr>& inputs) override;

	mlCore::TensorShape getOutputShape() const override;

	std::vector<autoDiff::NodePtr> getTrainableWeights() const override;

	std::string getDescription() const override;

	void build(const std::vector<mlCore::TensorShape>& inputShapes) override;

	void saveWeights(const std::string& path) const override;

	void loadWeights(const std::string& path) override;

private:
	void _validateWeightsPath(const std::string& path) const;

	std::map<std::string, BaseLayerPtr> _getSavePathsNames() const;

	std::vector<BaseLayerPtr> _layers;
};
} // namespace layers

#endif