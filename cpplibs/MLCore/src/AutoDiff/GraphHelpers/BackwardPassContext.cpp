#include "AutoDiff/GraphHelpers/BackwardPassContext.h"

namespace autoDiff::detail
{
BackwardPassContext::BackwardPassContext(const BackwardPassParams& params)
	: _params(params)
	, _graphInfoExtractor(params.root)
{}

void BackwardPassContext::run() {}
} // namespace autoDiff::detail