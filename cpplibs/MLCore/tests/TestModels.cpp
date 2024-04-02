#include <vector>

#include <Models/Callback.hpp>
#include <Models/ILayer.hpp>
#include <Models/IMeasurable.hpp>
#include <Models/IMetric.hpp>
#include <Models/IOptimizer.hpp>
#include <gtest/gtest.h>

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

	TestLayer(const TestLayer&) = default;
	TestLayer(TestLayer&&) = default;
	TestLayer& operator=(const TestLayer&) = default;
	TestLayer& operator=(TestLayer&&) = default;

	~TestLayer() override = default;

	mlCore::autoDiff::NodePtr build() override
	{
		return {};
	}

	mlCore::Tensor compute() override
	{
		return 0.0;
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

	TestOptimizer(const TestOptimizer&) = default;
	TestOptimizer(TestOptimizer&&) = default;
	TestOptimizer& operator=(const TestOptimizer&) = default;
	TestOptimizer& operator=(TestOptimizer&&) = default;

	~TestOptimizer() override = default;

	void applyGradient(mlCore::autoDiff::NodePtr weight, mlCore::Tensor derivative) override
	{
		weight->getValue() = std::move(derivative);
	}
};

/// Test class for checking IMeasurable code building
class TestMeasurable : public mlCore::models::IMeasurable
{
public:
	TestMeasurable() = default;

	TestMeasurable(const TestMeasurable&) = default;
	TestMeasurable(TestMeasurable&&) = default;
	TestMeasurable& operator=(const TestMeasurable&) = default;
	TestMeasurable& operator=(TestMeasurable&&) = default;

	~TestMeasurable() override = default;

	void registerMetric(mlCore::models::IMetricPtr metric) override
	{
		_metrics.push_back(metric);
	}

	void unregisterMetric(mlCore::models::IMetricPtr metric) override
	{
		_metrics.erase(std::remove_if(_metrics.begin(),
									  _metrics.end(),
									  [&metric](const auto met) { return met == metric; }),
					   _metrics.end());
	}

	bool hasMetric(std::shared_ptr<mlCore::models::IMetric> metric) const override
	{
		return std::find(_metrics.cbegin(), _metrics.cend(), metric) != _metrics.end();
	}

	void notifyMetrics() override
	{
		for(const auto& metric : _metrics)
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
	std::vector<mlCore::models::IMetricPtr> _metrics;
};

/// Test class for checking IMetric code building
class TestMetric : public mlCore::models::IMetric
{
public:
	TestMetric() = default;

	TestMetric(const TestMetric&) = default;
	TestMetric(TestMetric&&) = default;
	TestMetric& operator=(const TestMetric&) = default;
	TestMetric& operator=(TestMetric&&) = default;

	~TestMetric() override = default;

	void notify(mlCore::models::MetricContextPtr /*context*/) override
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

	TestCallback(const TestCallback&) = default;
	TestCallback(TestCallback&&) = default;
	TestCallback& operator=(const TestCallback&) = default;
	TestCallback& operator=(TestCallback&&) = default;

	~TestCallback() override = default;

	void call() override {}
};

/*************
 *
 * Test cases
 *
 *************/

TEST(TestModels, testTestLayer)
{
	const TestLayer layer;
}

TEST(TestModels, testTestOptimizer)
{
	const mlCore::models::IOptimizerPtr optimizer = std::make_shared<TestOptimizer>();

	auto node = std::make_shared<mlCore::autoDiff::Node>(mlCore::Tensor(std::vector<size_t>{}));
	const mlCore::Tensor tensor({}, 0);

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
