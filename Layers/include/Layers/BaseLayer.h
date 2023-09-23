#ifndef LAYERS_INCLUDE_LAYERS_BASELAYER_H
#define LAYERS_INCLUDE_LAYERS_BASELAYER_H

// __Own software headers__
#include <Models/ILayer.hpp>
#include <AutoDiff/ComputationGraph.h>

namespace layers
{
/**
 * @brief Abstract class defining common methods and attributes for other layers.
 * 
 */
class BaseLayer : public mlCore::models::ILayer
{
public:
	BaseLayer();

	BaseLayer(const BaseLayer&) = default;			  // Copy constructor.
	BaseLayer(BaseLayer&&) = default;				  // Move constructor.
	BaseLayer& operator=(const BaseLayer&) = default; // Copy assignment.
	BaseLayer& operator=(BaseLayer&&) = default;	  // Move assignment.

	~BaseLayer() override = default;

	/**
     * @brief Sets layer's input. It is the concrete descendant's task to perform additional checks about the set input. 
     * 
     * @param inputLayer New layer's input. 
     */
	virtual void setInputLayer(const mlCore::models::ILayerPtr& inputLayer) = 0;

	/**
     * @brief Return's the layer's input.
     */
	mlCore::models::ILayerPtr getInputLayer() const;

	/**
     * @brief Sets the layer's name. The name is the base of the layer's description.
     * 
     * @param name New layer's name.
     */
	void setName(const std::string& name);

	/**
     * @brief Returns the layer's name.
     */
	const std::string& getName() const;

	/**
     * @brief Sets the layer's associated computation graph. 
     * 
     * Method detaches the layer from the previously set graph,
     * therefore the behavior of the method has to be defined by the descendant.
     * 
     * @param graph New computation graph.
     */
	virtual void setGraph(const std::shared_ptr<mlCore::autoDiff::ComputationGraph>& graph) = 0;

	/**
     * @brief Returns the layer's associated graph.
     */
	std::shared_ptr<mlCore::autoDiff::ComputationGraph> getGraph() const;

protected:
	mlCore::models::ILayerPtr inputLayer_;
	std::shared_ptr<mlCore::autoDiff::ComputationGraph> graph_;
	std::string name_;
	bool built_;
};
} // namespace layers

#endif