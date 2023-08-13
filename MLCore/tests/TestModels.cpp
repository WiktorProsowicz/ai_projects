#include <vector>

#include <gtest/gtest.h>

#include <Models/ILayer.hpp>
#include <Models/Callback.hpp>
#include <Models/IMeasurable.hpp>
#include <Models/IMetric.hpp>
#include <Models/IOptimizer.hpp>

namespace
{

/*********************************************
 * 
 * Classes implementing the tested interfaces
 * 
 *********************************************/

/// Test class for checking ILayer code building
class TestLayer : public mlCore::models::ILayer
{
public:
	TestLayer() = default;
	~TestLayer() = default;

	mlCore::autoDiff::NodePtr build() override
	{
		return {};
	}

	mlCore::Tensor compute() override
	{
		return mlCore::Tensor({}, 0);
	}

	std::vector<mlCore::autoDiff::NodePtr> getAllWeights() const override
	{
		return {};
	}

	std::vector<mlCore::autoDiff::NodePtr> getTrainableWeights() const override
	{
		return {};
	}

	std::string getDescription() const override
	{
		return "";
	}
};

/// Test class for checking IOptimizer code building
class TestOptimizer : public mlCore::models::IOptimizer
{
public:
	TestOptimizer() = default;
	~TestOptimizer() = default;

	void applyGradient(mlCore::autoDiff::NodePtr weight, const mlCore::Tensor& derivative) override
	{
		weight->setValue(derivative);
	}
};

/// Test class for checking IMeasurable code building
class TestMeasurable : public mlCore::models::IMeasurable
{
public:
	TestMeasurable() = default;
	~TestMeasurable() = default;

	void registerMetric(mlCore::models::IMetricPtr metric) override
	{
		metrics_.push_back(metric);
	}

	void unregisterMetric(mlCore::models::IMetricPtr metric) override
	{
		metrics_.erase(std::remove_if(metrics_.begin(), metrics_.end(), [&metric](const auto m) { return m == metric; }),
					   metrics_.end());
	}

	bool hasMetric(std::shared_ptr<mlCore::models::IMetric> metric) const override
	{
		return std::find(metrics_.cbegin(), metrics_.cend(), metric) != metrics_.end();
	}

	void notifyMetrics() override
	{
		for(auto metric : metrics_)
		{
			auto context = std::make_shared<mlCore::models::MetricContext>();

			metric->notify(context);
		}
	}

	std::string getIdentifier() override
	{
		return "";
	}

private:
	std::vector<mlCore::models::IMetricPtr> metrics_{};
};

/// Test class for checking IMetric code building
class TestMetric : public mlCore::models::IMetric
{
public:
	TestMetric() = default;
	~TestMetric() = default;

	void notify(mlCore::models::MetricContextPtr context) override
	{
		notified = true;
	}

	bool notified = false;
};

/// Test class for checking Callback code building
class TestCallback : public mlCore::models::Callback
{
public:
	TestCallback() = default;
	~TestCallback() = default;

	void call() override { }
};

/*************
 * 
 * Test cases
 * 
 *************/

TEST(TestModels, testTestLayer)
{
	TestLayer layer;
}

TEST(TestModels, testTestOptimizer)
{
	mlCore::models::IOptimizerPtr optimizer = std::make_shared<TestOptimizer>();

	auto node = std::make_shared<mlCore::autoDiff::Node>(mlCore::Tensor(std::vector<size_t>{}));
	mlCore::Tensor tensor({}, 0);

	optimizer->applyGradient(node, tensor);
}

TEST(TestModels, testTestMeasurable)
{
	TestMeasurable measurable;

	auto metric = std::make_shared<TestMetric>();

	measurable.registerMetric(metric);

	ASSERT_TRUE(measurable.hasMetric(metric));

	measurable.unregisterMetric(metric);

	ASSERT_FALSE(measurable.hasMetric(metric));

	measurable.registerMetric(metric);

	measurable.notifyMetrics();

	ASSERT_TRUE(metric->notified);
}

TEST(TestModels, testTestCallback)
{
	TestCallback callback;

	callback.addMode(mlCore::models::CallbackMode::END_OF_BATCH);
	callback.addMode(mlCore::models::CallbackMode::END_OF_TRAINING);

	EXPECT_TRUE(callback.hasMode(mlCore::models::CallbackMode::END_OF_BATCH));
	EXPECT_TRUE(callback.hasMode(mlCore::models::CallbackMode::END_OF_TRAINING));

	callback.removeMode(mlCore::models::CallbackMode::END_OF_BATCH);

	EXPECT_FALSE(callback.hasMode(mlCore::models::CallbackMode::END_OF_BATCH));
}

} // namespace