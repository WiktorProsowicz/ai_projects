#ifndef MLCORE_INCLUDE_MODELS_IMETRIC_HPP
#define MLCORE_INCLUDE_MODELS_IMETRIC_HPP

#include <memory>

namespace mlCore::models
{

class IMeasurable;

/**
 * @brief Empty base class for objects delivering context for metrics.
 * 
 */
struct MetricContext
{
	virtual ~MetricContext() = default;
};

using MetricContextPtr = std::shared_ptr<MetricContext>;

/**
 * @brief Interface for subscriber metric classes that extract data from context object while being notified. 
 * 
 */
class IMetric
{
public:
	/**
     * @brief Notifies the metric about a specific state of the model learning/inference process.
     * 
     * @param context Object containing data for the metric.
     */
	virtual void notify(MetricContextPtr context = nullptr) = 0;

	virtual ~IMetric() = default;
};

using IMetricPtr = std::shared_ptr<IMetric>;

} // namespace mlCore::models

#endif