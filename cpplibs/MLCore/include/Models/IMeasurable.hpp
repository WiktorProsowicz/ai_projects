#ifndef MLCORE_INCLUDE_MODELS_IMEASURABLE_HPP
#define MLCORE_INCLUDE_MODELS_IMEASURABLE_HPP

#include <memory>
#include <vector>
#include <string>

#include <Models/IMetric.hpp>

namespace mlCore::models
{
/**
 * @brief Interface for publisher classes that are measured by metrics.
 * 
 */
class IMeasurable
{
public:
	/**
     * @brief Subscribes the publisher and adds the metrics to its subscribers.
     * 
     * @param metric Metric to subscribe the IMeasurable object.
     */
	virtual void registerMetric(std::shared_ptr<IMetric> metric) = 0;

	/**
     * @brief Unsubscribes the publisher and removes the metric from its subscribers.
     * 
     * @param metric Metric to unsubscribe the IMeasurable object.
     */
	virtual void unregisterMetric(std::shared_ptr<IMetric> metric) = 0;

	/**
      * @brief Check whether the publisher has registered metric;
      * 
      * @param metric Metric to look for amongst publisher's metric.
      * @return true If the measurable has the metric.
      * @return false If the measurable does not have the metric.
      */
	virtual bool hasMetric(std::shared_ptr<IMetric> metric) const = 0;

	/**
     * @brief Notifies the subscribers.
     * 
     */
	virtual void notifyMetrics() = 0;

	/**
     * @brief Returns a textual identifier referring to the measured object.
     * 
     * @return Value identifying the measurable.
     */
	virtual std::string getIdentifier() = 0;

	virtual ~IMeasurable() = default;
};

using IMeasurablePtr = std::shared_ptr<IMeasurable>;
} // namespace mlCore::models

#endif