#ifndef LAYER_INCLUDE_LAYER_DENSELAYER_H
#define LAYER_INCLUDE_LAYER_DENSELAYER_H

#include "Layers/BaseLayer.h"
#include "LayersModules/IActivationFactory.hpp"

namespace layers
{
/**
 * @brief Implements the most simple neural network layer.
 *
 * @details The dense layer is a fully connected layer where each neuron is connected to all neurons in the
 * input layer. The output is a linear combination of the input values and the weights of the layer. An
 * additional weight providing a bias is added to the output. In order to apply non-linearity to the function
 * approximated by the layer, an activation function shall be applied to the output.
 */
class DenseLayer : public BaseLayer
{
public:
	DenseLayer() = delete;

	DenseLayer(std::string name, size_t units, IActivationFactoryPtr activationFactory);

	DenseLayer(const DenseLayer&) = default;
	DenseLayer(DenseLayer&&) = default;
	DenseLayer& operator=(const DenseLayer&) = default;
	DenseLayer& operator=(DenseLayer&&) = default;

	~DenseLayer() override = default;

	autoDiff::OperatorPtr call(const std::vector<autoDiff::NodePtr>& inputs) override;

	mlCore::TensorShape getOutputShape() const override;

	std::vector<autoDiff::NodePtr> getTrainableWeights() const override;

	std::string getDescription() const override;

	void build(const std::vector<mlCore::TensorShape>& inputShapes) override;

	void saveWeights(const std::string& path) const override;

	void loadWeights(const std::string& path) override;

private:
	autoDiff::VariablePtr _weights{};
	autoDiff::VariablePtr _bias{};
	size_t _units;
	IActivationFactoryPtr _activationFactory;
};
} // namespace layers

#endif