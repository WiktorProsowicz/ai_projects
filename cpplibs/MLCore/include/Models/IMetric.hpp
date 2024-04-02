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
	MetricContext() = default;

	MetricContext(const MetricContext&) = default;
	MetricContext(MetricContext&&) = default;
	MetricContext& operator=(const MetricContext&) = default;
	MetricContext& operator=(MetricContext&&) = default;

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
	IMetric() = default;

	IMetric(const IMetric&) = default;
	IMetric(IMetric&&) = default;
	IMetric& operator=(const IMetric&) = default;
	IMetric& operator=(IMetric&&) = default;

	virtual ~IMetric() = default;

	/**
	 * @brief Notifies the metric about a specific state of the model learning/inference process.
	 *
	 * @param context Object containing data for the metric.
	 */
	virtual void notify(MetricContextPtr context) = 0;
};

using IMetricPtr = std::shared_ptr<IMetric>;

} // namespace mlCore::models

#endif
