#ifndef LAYERS_INCLUDE_LAYERS_BASELAYER_H
#define LAYERS_INCLUDE_LAYERS_BASELAYER_H

#include <Interfaces/ILayer.hpp>

namespace layers
{
/**
 * @brief Base class for other layers. Contains common methods and attributes.
 *
 * @details Since the base layer is abstract, the concrete implementations must override certain methods in
 * order to provide complete functionality.
 *
 */
class BaseLayer : public interfaces::ILayer
{
public:
	BaseLayer() = delete;

	/**
	 * @brief Constructs a new BaseLayer object.
	 *
	 * @param name Name of the layer.
	 */
	BaseLayer(std::string name);

	BaseLayer(const BaseLayer&) = delete;
	BaseLayer(BaseLayer&&) = delete;
	BaseLayer& operator=(const BaseLayer&) = delete;
	BaseLayer& operator=(BaseLayer&&) = delete;

	virtual ~BaseLayer() override = default;

	/**
	 * @brief Initializes the layer's internal state.
	 *
	 * @param inputShapes Shapes of the layer's inputs.
	 */
	virtual void build(const std::vector<mlCore::TensorShape>& inputShapes) = 0;

	/**
	 * @brief Saves the layer's internal state to a file.
	 *
	 * @param path Path to the file where the state should be saved.
	 */
	virtual void saveWeights(const std::string& path) const = 0;

	/**
	 * @brief Loads the layer's internal state from a file.
	 *
	 * @details The file given with the path shall be checked by the concrete layer type.
	 *
	 * @param path Path to the file the state should be loaded from.
	 */
	virtual void loadWeights(const std::string& path) = 0;

	/**
	 * @brief Returns the string identifier of the layer.
	 *
	 */
	const std::string& getName() const;

protected:
	/// Tells whether the layer has been built.
	bool _isBuilt() const;

	/// Sets the layer as built. This method should be used once the `build` method is called.
	void _setBuilt();

	/// Sets the vlaue of the given weight and checks whether the shapes are compatible.
	void _setWeight(const autoDiff::VariablePtr& weight, mlCore::Tensor value);

private:
	std::string _name;
	/// Tells whether the layer's internal state has been initialized.
	bool _built = false;
};

/// @brief Shared pointer to a BaseLayer object.
using BaseLayerPtr = std::shared_ptr<BaseLayer>;
} // namespace layers

#endif