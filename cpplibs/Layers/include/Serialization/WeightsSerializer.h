#include <fstream>

#include <AutoDiff/GraphNodes.hpp>

namespace layers::serialization
{
class TensorHandle;

/**
 * @brief Opens a connection to a file storing layers' weights.
 *
 * @details The saver is responsible for saving and retrieving the weights of the layers from the file
 * associated with it. The format of the file is as follows:
 *
 * {
 *   n_items - uint64_t	// Number of blocks in the file
 * 	 // Each block contain
 * 	 {
 *     n_dimensions - uint64_t	// Number of dimensions of the tensor
 * 	   dimensions - n_dimensions * uint64_t	// Dimensions of the tensor
 *     data - product(dimensions) * double	// Data of the tensor
 * 	 }
 * }
 */
class WeightsSerializer
{
public:
	/**
	 * @brief Creates a serializer and connects it to a file with the given path.
	 *
	 * The given path is checked and the file is created if it does not exist. In the case the file already
	 * exists, it is validated.
	 */
	static std::unique_ptr<WeightsSerializer> open(const std::string& path);

	WeightsSerializer() = delete;

	WeightsSerializer(const WeightsSerializer&) = delete;
	WeightsSerializer(WeightsSerializer&&) = delete;
	WeightsSerializer& operator=(const WeightsSerializer&) = delete;
	WeightsSerializer& operator=(WeightsSerializer&&) = delete;

	~WeightsSerializer() = default;

	/**
	 * @brief Returns handles for each tensor stored in the file.
	 */
	const std::vector<TensorHandle>& getTensorHandles() const
	{
		return _tensorHandles;
	}

private:
	WeightsSerializer(std::unique_ptr<std::fstream> fileStream);

	/// Checks if the given file is a valid weights file.
	static void _validateFile(const std::string& path);

	/// Initializes the handles for the tensors stored in the pointed file.
	void _initHandles();

	/// Creates a new tensor handle at a given position and sets its basic fields.
	void _allocateHandle(const std::streampos& position);

	std::vector<TensorHandle> _tensorHandles;
	std::unique_ptr<std::fstream> _fileStream;
};

/**
 * @brief Points to a place in a file where a tensor is serialized.
 *
 * @details The handle can be used to either save or retrieve the tensor from the file.
 */
class TensorHandle
{
public:
	/**
	 * @brief Creates a handle pointing to the given position in the file.
	 */
	TensorHandle(std::fstream& file, std::streampos position);

	TensorHandle() = delete;

	TensorHandle(const TensorHandle&) = default;
	TensorHandle(TensorHandle&&) = delete;

	TensorHandle& operator=(const TensorHandle&) = default;
	TensorHandle& operator=(TensorHandle&&) = delete;

	~TensorHandle() = default;

	bool operator==(const TensorHandle& other) const
	{
		return _position == other._position;
	}

	/**
	 * @brief Saves the tensor to the file.
	 */
	void save(const mlCore::Tensor& tensor);

	/**
	 * @brief Retrieves the tensor from the file.
	 */
	mlCore::Tensor get() const;

	/**
	 * @brief Returns the shape of the tensor.
	 */
	mlCore::TensorShape getShape() const;

private:
	std::fstream& _file;
	std::streampos _position;
};
} // namespace layers::serialization