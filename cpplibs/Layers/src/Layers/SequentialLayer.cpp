#include "Layers/SequentialLayer.h"

#include <algorithm>
#include <filesystem>
#include <ranges>
#include <set>

namespace layers
{
SequentialLayer::SequentialLayer(std::string name, std::vector<BaseLayerPtr> layers)
	: BaseLayer(std::move(name))
	, _layers(std::move(layers))
{
	if(_layers.empty())
	{
		LOG_ERROR("Layers::SequentialLayer", "Sequential layer must have at least one layer.");
	}

	if(std::set<layers::BaseLayerPtr>{_layers.begin(), _layers.end()}.size() != _layers.size())
	{
		LOG_ERROR("Layers::SequentialLayer", "Sequential layer must have unique layers.");
	}

	if(std::any_of(_layers.cbegin(), _layers.cend(), [](const auto& layer) { return layer == nullptr; }))
	{
		LOG_ERROR("Layers::SequentialLayer", "Sequential layer must not have null layers.");
	}
}

autoDiff::OperatorPtr SequentialLayer::call(const std::vector<autoDiff::NodePtr>& inputs)
{
	if(!_isBuilt())
	{
		LOG_ERROR("Layers::SequentialLayer", "Layer must be built before calling it.");
	}

	if(inputs.size() != 1)
	{
		LOG_ERROR("Layers::SequentialLayer", "Sequential layer must have exactly one input.");
	}

	auto output = inputs;
	for(const auto& layer : _layers | std::views::take(_layers.size() - 1))
	{
		output = {layer->call(output)};
	}

	return _layers.back()->call(output);
}

mlCore::TensorShape SequentialLayer::getOutputShape() const
{
	if(!_isBuilt())
	{
		LOG_ERROR("Layers::SequentialLayer", "Layer must be built before getting output shape.");
	}

	return _layers.back()->getOutputShape();
}

std::vector<autoDiff::NodePtr> SequentialLayer::getTrainableWeights() const
{
	if(!_isBuilt())
	{
		LOG_ERROR("Layers::SequentialLayer", "Layer must be built before getting trainable weights.");
	}

	std::vector<autoDiff::NodePtr> weights;
	for(const auto& layer : _layers)
	{
		const auto layerWeights = layer->getTrainableWeights();
		weights.insert(weights.end(), layerWeights.begin(), layerWeights.end());
	}

	return weights;
}

std::string SequentialLayer::getDescription() const
{
	constexpr const char* descriptionTemplate = R"({} (SequentialLayer) Layers: [{}])";

	auto layersDescription = fmt::join(
		_layers | std::views::transform([](const auto& layer) { return layer->getDescription(); }), ", ");

	return fmt::format(descriptionTemplate, getName(), std::move(layersDescription));
}

void SequentialLayer::build(const std::vector<mlCore::TensorShape>& inputShapes)
{
	if(inputShapes.size() != 1)
	{
		LOG_ERROR("Layers::SequentialLayer", "Sequential layer must have exactly one input.");
	}

	auto outputShapes = inputShapes;
	for(const auto& layer : _layers)
	{
		layer->build(outputShapes);
		outputShapes = {layer->getOutputShape()};
	}

	_setBuilt();
}

void SequentialLayer::saveWeights(const std::string& path) const
{
	if(!_isBuilt())
	{
		LOG_ERROR("Layers::SequentialLayer", "Layer must be built before saving weights.");
	}

	if(std::filesystem::exists(path))
	{
		_validateWeightsPath(path);
	}
	else
	{
		std::filesystem::create_directory(path);
	}

	for(const auto& [saveFileName, layer] : _getSavePathsNames())
	{
		layer->saveWeights(fmt::format("{}/{}", path, saveFileName));
	}
}

void SequentialLayer::loadWeights(const std::string& path)
{
	if(!_isBuilt())
	{
		LOG_ERROR("Layers::SequentialLayer", "Layer must be built before loading weights.");
	}

	if(!std::filesystem::exists(path))
	{
		LOG_ERROR("Layers::SequentialLayer", "The path to load the weights does not exist.");
	}

	_validateWeightsPath(path);

	for(const auto& layer : _layers)
	{
		layer->loadWeights(fmt::format("{}/{}", path, layer->getName()));
	}
}

void SequentialLayer::_validateWeightsPath(const std::string& path) const
{
	if(!std::filesystem::is_directory(path))
	{
		LOG_ERROR("Layers::SequentialLayer", "The path to save the weights must be a directory.");
	}

	for(const auto layersWithNames = _getSavePathsNames();
		const auto& subPath : std::filesystem::directory_iterator(path))
	{
		if(std::ranges::count_if(layersWithNames | std::views::keys,
								 [&subPath](const auto& layerName)
								 { return layerName == subPath.path().filename().string(); }) != 1)
		{
			LOG_ERROR("Layers::SequentialLayer",
					  fmt::format("The path to save thr weights there should be exactly one path for each "
								  "layer in the sequence. The path {} is not valid.",
								  subPath.path().string()));
		}
	}
}

namespace
{
/// @brief Cleans a given name so that it can be used as a file name.
std::string sanitizeLayerName(const std::string& name)
{
	std::string sanitizedName = name;

	std::replace_if(
		sanitizedName.begin(), sanitizedName.end(), [](char c) { return !std::isalnum(c); }, '_');

	return sanitizedName;
}
} // namespace

std::map<std::string, BaseLayerPtr> SequentialLayer::_getSavePathsNames() const
{
	std::map<std::string, BaseLayerPtr> savePathsNames;

	for(const auto& layer : _layers)
	{
		auto saveFileName = sanitizeLayerName(layer->getName());

		if(savePathsNames.contains(saveFileName))
		{
			LOG_ERROR("Layers::SequentialLayer",
					  fmt::format("The sanitized layer's name '{}' is not unique in the sequence.",
								  layer->getName()));
		}

		savePathsNames.emplace(std::move(saveFileName), layer);
	}

	return savePathsNames;
}
} // namespace layers