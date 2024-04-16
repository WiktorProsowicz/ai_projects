#ifndef MLCORE_TESTS_OPERATORWRAPPER_HPP
#define MLCORE_TESTS_OPERATORWRAPPER_HPP

#include <shared_mutex>

#include <AutoDiff/GraphNodes.hpp>

namespace mlCoreTests
{
/// Used to log the flow of data in the computation graph.
class OperatorStats
{
public:
	// Enum specifying type of log made by computation graph node wrappers.
	enum class WrapperLogChannel : uint8_t
	{
		UPDATE_VALUE,
		COMPUTE_DERIVATIVE
	};

	/// Logs a text message to a given log channel.
	void log(const std::string& log, WrapperLogChannel channel)
	{
		const std::unique_lock lock(_logsMutex);
		_logs[channel].push_back(log);
	}

	/// Returns logs for a given channel.
	const std::vector<std::string>& getLogs(WrapperLogChannel channel) const
	{
		return _logs.at(channel);
	}

private:
	std::shared_mutex _logsMutex{};
	std::map<WrapperLogChannel, std::vector<std::string>> _logs{};
};

/// Wraps an operator and apart from delegating the calls to the wrapped operator, logs the flow of data in
/// the computation graph.
class OperatorDecorator : public autoDiff::Operator
{
public:
	OperatorDecorator(const std::vector<autoDiff::NodePtr>& inputs,
					  autoDiff::OperatorPtr wrappedOper,
					  const std::shared_ptr<OperatorStats>& opStats)
		: Operator(inputs)
		, _wrappedOper(std::move(wrappedOper))
		, _opStats(opStats)

	{
		setName(_wrappedOper->getName());
	}

	const mlCore::Tensor& getValue() const override
	{
		return _wrappedOper->getValue();
	}

	autoDiff::NodePtr copy() const override
	{
		return _wrappedOper->copy();
	}

	void updateValue() override
	{
		_wrappedOper->updateValue();

		for(const auto& input : getInputs())
		{
			_opStats->log(input->getName() + " -> " + getName(),
						  OperatorStats::WrapperLogChannel::UPDATE_VALUE);
		}
	}

	std::vector<mlCore::Tensor> computeDerivative(const mlCore::Tensor& outerDerivative) const override
	{
		for(const auto& input : getInputs())
		{
			_opStats->log(input->getName() + " <- " + getName(),
						  OperatorStats::WrapperLogChannel::COMPUTE_DERIVATIVE);
		}

		return _wrappedOper->computeDerivative(outerDerivative);
	}

	std::vector<mlCore::Tensor> computeDirectDerivative() const override
	{
		return _wrappedOper->computeDirectDerivative();
	}

	const std::vector<size_t>& getOutputShape() const override
	{
		return _wrappedOper->getOutputShape();
	}

private:
	autoDiff::OperatorPtr _wrappedOper;
	std::shared_ptr<OperatorStats> _opStats;
};
}; // namespace mlCoreTests

#endif
