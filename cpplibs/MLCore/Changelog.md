# Changelog

MLCore is a module containing the core components for other data processing / machine learning modules. Defines basic structures for efficient data management and processing, automatic differentiation grounds and interfaces for machine learning models.

## 1.0.0

- Introduced `BasicTensor`
- Introduced `TensorIterator`
- Introduced `ITensorInitializer` interface
- Added new `TensorInitializers`
    - `RangeTensorInitializer`
    - `GaussianTensorInitializer`
- Introduced base `GraphNodes` interface (`Node`)
- Added new `GraphNodes`
    - `Variable`
    - `Constant`
    - `Placeholder`
- Introduced base `UnaryOperators` interface (`UnaryOperator`)
- Added new `UnaryOperators`
    - `ReluOperator`
    - `SigmoidOperator`
- Introduced base `BinaryOperators` interface (`BinaryOperator`)
- Added new `BinaryOperators`:
    - `AddOperator`
    - `SubtractOperator`
    - `DivideOperator`
    - `MultiplyOperator`
    - `MatmulOperator`
    - `PowerOperator`
- Introduced `ComputationGraph`
- Introduced `TensorOperations`
- Introduced `Models` interfaces:
    - `ILayer`
    - `Callback`
    - `IMeasurable`
    - `IMetric`
    - `IOptimizer`
