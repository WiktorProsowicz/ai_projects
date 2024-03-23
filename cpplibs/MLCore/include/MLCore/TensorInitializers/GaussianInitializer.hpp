#ifndef MLCORE_INCLUDE_MLCORE_TENSORINITIALIZERS_GAUSSIANINITIALIZER_HPP
#define MLCORE_INCLUDE_MLCORE_TENSORINITIALIZERS_GAUSSIANINITIALIZER_HPP

#include <MLCore/TensorInitializers/ITensorInitializer.hpp>
#include <random>

namespace mlCore::tensorInitializers
{
/**
 * @brief Initializer class yielding values sampled from the gaussian distribution.
 *
 */
template <typename ValueType>
class GaussianInitializer : public ITensorInitializer<ValueType>
{
public:
	explicit GaussianInitializer(ValueType mean = 0, ValueType stddev = 1)
		: distribution_(mean, stddev)
		, engine_(std::random_device{}())
	{}

	GaussianInitializer(const GaussianInitializer&) = default;			  // Copy constructor.
	GaussianInitializer(GaussianInitializer&&) = default;				  // Move constructor.
	GaussianInitializer& operator=(const GaussianInitializer&) = default; // Copy assignment.
	GaussianInitializer& operator=(GaussianInitializer&&) = default;	  // Move assignment.

	bool canYield() const override
	{
		return true;
	}

	ValueType yield() const override
	{
		return distribution_(engine_);
	}

	~GaussianInitializer() override = default;

private:
	mutable std::normal_distribution<ValueType> distribution_;
	mutable std::default_random_engine engine_;
};
} // namespace mlCore::tensorInitializers

#endif
