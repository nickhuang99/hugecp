#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

bool ends_with(const std::string &str, const std::string &suffix) {
  return str.size() >= suffix.size() &&
         0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

typedef uint16_t bfloat16;

bfloat16 float_to_bfloat16(float f) {
  uint32_t f_bits = *reinterpret_cast<uint32_t *>(&f);
  uint32_t bf16_bits = (f_bits >> 16);
  return *reinterpret_cast<bfloat16 *>(&bf16_bits);
}

float bfloat16_to_float(bfloat16 bf) {
  uint32_t bf16_bits = *reinterpret_cast<uint16_t *>(&bf);
  uint32_t f_bits = bf16_bits << 16;
  return *reinterpret_cast<float *>(&f_bits);
}

void update_progress(int progress) {
  int bar_length = 40; // Modify this to change the bar's length
  int filled_length = (int)(bar_length * progress / 100.0);
  char bar[bar_length + 1]; // +1 for the null terminator
  for (int i = 0; i < bar_length; i++) {
    if (i < filled_length) {
      bar[i] = '=';
    } else {
      bar[i] = '-';
    }
  }
  bar[bar_length] = '\0'; // Null-terminate the string
  printf("\r[%s] %d%%", bar, progress);
  fflush(stdout); // Ensure output is written immediately
}

std::vector<bfloat16>
weight_dequant_cpu(const std::vector<uint8_t> &quantized_weight,
                   const std::vector<float> &scale_inv, long long M,
                   long long N, int block_size = 128) {
  if (quantized_weight.empty() || scale_inv.empty() || M <= 0 || N <= 0 ||
      block_size <= 0) {
    std::cerr << "Error: Invalid input to weight_dequant_cpu." << std::endl;
    return {};
  }
  if (quantized_weight.size() != M * N) {
    std::cerr << "Error: quantized_weight size does not match M * N."
              << std::endl;
    return {};
  }

  long long num_row_blocks = (M + block_size - 1) / block_size;
  long long num_col_blocks = (N + block_size - 1) / block_size;
  if (scale_inv.size() != num_row_blocks * num_col_blocks) {
    std::cerr << "Error: scale_inv size does not match the expected number of "
                 "blocks ("
              << num_row_blocks * num_col_blocks << " vs " << scale_inv.size()
              << ")." << std::endl;
    return {};
  }

  std::vector<bfloat16> dequantized_weight(M * N);

  for (long long row_block_idx = 0; row_block_idx < num_row_blocks;
       ++row_block_idx) {
    for (long long col_block_idx = 0; col_block_idx < num_col_blocks;
         ++col_block_idx) {
      float current_scale_inv =
          scale_inv[row_block_idx * num_col_blocks + col_block_idx];
      float current_scale = 1.0f / current_scale_inv;

      for (int row_offset = 0; row_offset < block_size; ++row_offset) {
        for (int col_offset = 0; col_offset < block_size; ++col_offset) {
          long long row_index = row_block_idx * block_size + row_offset;
          long long col_index = col_block_idx * block_size + col_offset;

          if (row_index < M && col_index < N) {
            long long weight_index = row_index * N + col_index;
            float dequantized_value =
                static_cast<float>(quantized_weight[weight_index]) *
                current_scale;
            dequantized_weight[weight_index] =
                float_to_bfloat16(dequantized_value);
          }
        }
      }
    }
  }

  return dequantized_weight;
}

template <typename T>
std::vector<T> load_tensor(const std::string &filename,
                           const std::string &tensor_name,
                           std::vector<long long> &shape) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return {};
  }

  uint64_t metadata_len;
  if (!file.read(reinterpret_cast<char *>(&metadata_len),
                 sizeof(metadata_len))) {
    std::cerr << "Error reading metadata length from " << filename << std::endl;
    return {};
  }

  std::string metadata_str(metadata_len, '\0');
  if (!file.read(metadata_str.data(), metadata_len)) {
    std::cerr << "Error reading metadata from " << filename << std::endl;
    return {};
  }

  nlohmann::json metadata;
  try {
    metadata = nlohmann::json::parse(metadata_str);
  } catch (const nlohmann::json::parse_error &e) {
    std::cerr << "Error parsing JSON metadata in " << filename << ": "
              << e.what() << std::endl;
    return {};
  }

  if (!metadata.contains("__metadata__") || !metadata.contains(tensor_name)) {
    std::cerr << "Error: Tensor '" << tensor_name
              << "' not found in metadata of " << filename << std::endl;
    return {};
  }

  auto tensor_info = metadata[tensor_name];
  std::vector<long long> loaded_shape =
      tensor_info["shape"].get<std::vector<long long>>();
  std::string dtype_str = tensor_info["dtype"].get<std::string>();
  std::vector<int64_t> data_offsets_array =
      tensor_info["data_offsets"].get<std::vector<int64_t>>();
  int64_t data_start = data_offsets_array[0];

  shape = loaded_shape;
  size_t tensor_num_elements = 1;
  for (long long dim : shape) {
    tensor_num_elements *= dim;
  }
  size_t element_size = 0;
  if (dtype_str == "uint8" || dtype_str == "int8" || dtype_str == "F8_E4M3")
    element_size = 1;
  else if (dtype_str == "uint16" || dtype_str == "int16" ||
           dtype_str == "bfloat16" || dtype_str == "BF16")
    element_size = 2;
  else if (dtype_str == "uint32" || dtype_str == "int32" ||
           dtype_str == "float32" || dtype_str == "F32")
    element_size = 4;
  else if (dtype_str == "uint64" || dtype_str == "int64" ||
           dtype_str == "float64")
    element_size = 8;
  else {
    std::cerr << "Error: Unsupported data type '" << dtype_str
              << "' for tensor '" << tensor_name << "'" << std::endl;
    return {};
  }

  if (sizeof(T) != element_size) {
    std::cerr << "Error: C++ data type size does not match tensor data type "
                 "size for tensor '"
              << tensor_name << "'" << std::endl;
    return {};
  }

  file.seekg(data_start, std::ios::beg);
  std::vector<T> data(tensor_num_elements);
  if (!file.read(reinterpret_cast<char *>(data.data()),
                 tensor_num_elements * sizeof(T))) {
    std::cerr << "Error reading tensor data for '" << tensor_name << "' from "
              << filename << std::endl;
    return {};
  }

  file.close();
  return data;
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <input_fp8_path> <output_bf16_path>"
              << std::endl;
    return 1;
  }

  std::string fp8_path = argv[1];
  std::string bf16_path = argv[2];
  std::filesystem::create_directories(bf16_path);

  std::string model_index_file = fp8_path + "/model.safetensors.index.json";
  std::ifstream f(model_index_file);
  if (!f.is_open()) {
    std::cerr << "Error: Could not open " << model_index_file << std::endl;
    return 1;
  }
  nlohmann::json model_index;
  f >> model_index;
  f.close();

  auto weight_map =
      model_index["weight_map"].get<std::map<std::string, std::string>>();
  std::map<std::string,
           std::tuple<std::vector<uint8_t>, std::vector<long long>>>
      fp8_weights;
  std::map<std::string, std::tuple<std::vector<float>, std::vector<long long>>>
      float_weights;
  std::map<std::string,
           std::tuple<std::vector<bfloat16>, std::vector<long long>>>
      bf16_weights;
  std::map<std::string, std::tuple<std::vector<float>, std::vector<long long>>>
      scale_inv_weights;

  std::vector<std::string> safetensor_files;
  for (const auto &entry : std::filesystem::directory_iterator(fp8_path)) {
    if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
      safetensor_files.push_back(entry.path().filename().string());
    }
  }
  std::sort(safetensor_files.begin(), safetensor_files.end());

  std::cout << "begin reading " << safetensor_files.size()
            << " safetensor files..." << std::endl;
  int read_file_counter = 0;
  for (const auto &file_name : safetensor_files) {
    update_progress((read_file_counter++) * 100 / safetensor_files.size());
    std::string safetensor_file_path = fp8_path + "/" + file_name;

    for (const auto &[weight_name, weight_in_file] : weight_map) {
      if (weight_in_file == file_name) {
        std::vector<long long> weight_shape;
        if (weight_name.find("weight") != std::string::npos &&
            !ends_with(weight_name, "_scale_inv")) {
          std::vector<uint8_t> weight_data = load_tensor<uint8_t>(
              safetensor_file_path, weight_name, weight_shape);
          if (!weight_data.empty()) {
            fp8_weights[weight_name] =
                std::make_tuple(weight_data, weight_shape);
          } else {
            std::cerr << "Warning: Could not load FP8 weight " << weight_name
                      << " in " << file_name << std::endl;
          }
        } else if (ends_with(weight_name, "_scale_inv")) {
          std::string base_weight_name = weight_name.substr(
              0, weight_name.length() - std::string("_scale_inv").length());
          std::vector<float> scale_inv_data = load_tensor<float>(
              safetensor_file_path, weight_name, weight_shape);
          if (!scale_inv_data.empty()) {
            scale_inv_weights[base_weight_name] =
                std::make_tuple(scale_inv_data, weight_shape);
          } else {
            std::cerr << "Warning: Could not load scale_inv " << weight_name
                      << " in " << file_name << std::endl;
          }
        } else {
          std::vector<float> weight_data = load_tensor<float>(
              safetensor_file_path, weight_name, weight_shape);
          if (!weight_data.empty()) {
            float_weights[weight_name] =
                std::make_tuple(weight_data, weight_shape);
          } else {
            std::cerr << "Warning: Could not load float weight " << weight_name
                      << " in " << file_name << std::endl;
          }
        }
      }
    }
  }
  std::cout << "begin dequantize " << fp8_weights.size() << " weight data..."
            << std::endl;
  std::map<std::string, std::pair<std::vector<char>, std::vector<long long>>>
      combined_data;
  nlohmann::json new_metadata_json;
  new_metadata_json["__metadata__"] = {{"format", "pt"}};
  uint64_t current_offset = 0;
  int dequant_weight_counter = 0;
  for (const auto &[weight_name, fp8_tuple] : fp8_weights) {
    update_progress((dequant_weight_counter++) * 100 / fp8_weights.size());
    if (scale_inv_weights.count(weight_name)) {
      const auto &[quantized_data, weight_shape] = fp8_tuple;
      const auto &[scale_inv_data, scale_shape] =
          scale_inv_weights.at(weight_name);
      long long M = 0;
      long long N = 0;
      if (weight_shape.size() == 2) {
        M = weight_shape[0];
        N = weight_shape[1];
        std::vector<bfloat16> bf16_data =
            weight_dequant_cpu(quantized_data, scale_inv_data, M, N);
        size_t data_size = bf16_data.size() * sizeof(bfloat16);
        std::vector<char> char_data(
            reinterpret_cast<const char *>(bf16_data.data()),
            reinterpret_cast<const char *>(bf16_data.data()) + data_size);
        combined_data[weight_name] = {char_data, weight_shape};
        new_metadata_json[weight_name] = {
            {"dtype", "BF16"},
            {"shape", weight_shape},
            {"data_offsets", {current_offset, current_offset + data_size}}};
        current_offset += data_size;
      } else {
        std::cerr << "Warning: FP8 weight " << weight_name
                  << " is not 2D, skipping dequantization." << std::endl;
      }
    } else {
      std::cerr << "Warning: Missing scale_inv for FP8 weight " << weight_name
                << std::endl;
    }
  }

  for (const auto &[weight_name, float_tuple] : float_weights) {
    const auto &[float_data, weight_shape] = float_tuple;
    std::vector<bfloat16> bf16_data(float_data.size());
    for (size_t i = 0; i < float_data.size(); ++i) {
      bf16_data[i] = float_to_bfloat16(float_data[i]);
    }
    size_t data_size = bf16_data.size() * sizeof(bfloat16);
    std::vector<char> char_data(
        reinterpret_cast<const char *>(bf16_data.data()),
        reinterpret_cast<const char *>(bf16_data.data()) + data_size);
    combined_data[weight_name] = {char_data, weight_shape};
    new_metadata_json[weight_name] = {
        {"dtype", "BF16"},
        {"shape", weight_shape},
        {"data_offsets", {current_offset, current_offset + data_size}}};
    current_offset += data_size;
  }

  std::string output_file_path = bf16_path + "/model.safetensors";
  std::ofstream outfile(output_file_path, std::ios::binary);
  if (!outfile.is_open()) {
    std::cerr << "Error: Could not open output file " << output_file_path
              << std::endl;
    return 1;
  }

  std::string metadata_str = new_metadata_json.dump();
  uint64_t metadata_len = metadata_str.length();

  outfile.write(reinterpret_cast<const char *>(&metadata_len),
                sizeof(metadata_len));
  outfile.write(metadata_str.data(), metadata_len);
  std::cout << "begin write " << combined_data.size() << " weight data..."
            << std::endl;
  int write_weight_counter = 0;
  for (const auto &[_, data_pair] : combined_data) {
    update_progress((write_weight_counter++) * 100 / combined_data.size());
    outfile.write(data_pair.first.data(), data_pair.first.size());
  }

  outfile.close();

  // Create the new index file
  nlohmann::json new_index_json;
  new_index_json["weight_map"] = nlohmann::json::object();
  for (const auto &[weight_name, _] : combined_data) {
    new_index_json["weight_map"][weight_name] = "model.safetensors";
  }

  std::ofstream index_outfile(bf16_path + "/model.safetensors.index.json");
  index_outfile << std::setw(4) << new_index_json << std::endl;
  index_outfile.close();

  std::cout << "Dequantization and merging complete. BF16 model saved to "
            << bf16_path << std::endl;

  return 0;
}