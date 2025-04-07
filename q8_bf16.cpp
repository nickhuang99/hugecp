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
  int bar_length = 40;
  int filled_length = (int)(bar_length * progress / 100.0);
  char bar[bar_length + 1];
  for (int i = 0; i < bar_length; i++) {
    if (i < filled_length) {
      bar[i] = '=';
    } else {
      bar[i] = '-';
    }
  }
  bar[bar_length] = '\0';
  printf("\r[%s] %d%%", bar, progress);
  fflush(stdout);
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
std::vector<T> load_tensor_data(const std::string &filename, int64_t offset,
                                size_t num_bytes) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return {};
  }
  file.seekg(offset, std::ios::beg);
  std::vector<T> data(num_bytes / sizeof(T));
  if (!file.read(reinterpret_cast<char *>(data.data()), num_bytes)) {
    std::cerr << "Error reading " << num_bytes << " bytes from " << filename
              << " at offset " << offset << std::endl;
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
  std::map<std::string, std::pair<std::vector<char>, std::vector<long long>>>
      combined_data;
  nlohmann::json new_metadata_json;
  new_metadata_json["__metadata__"] = {{"format", "pt"}};
  uint64_t current_offset = 0;

  std::vector<std::string> safetensor_files;
  for (const auto &entry : std::filesystem::directory_iterator(fp8_path)) {
    if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
      safetensor_files.push_back(entry.path().filename().string());
    }
  }
  std::sort(safetensor_files.begin(), safetensor_files.end());

  std::cout << "Processing " << safetensor_files.size()
            << " safetensor files..." << std::endl;
  int file_counter = 0;
  for (const auto &file_name : safetensor_files) {
    update_progress((file_counter++) * 100 / safetensor_files.size());
    std::string safetensor_file_path = fp8_path + "/" + file_name;
    std::ifstream infile(safetensor_file_path, std::ios::binary);
    if (!infile.is_open()) {
      std::cerr << "Error: Could not open file " << safetensor_file_path
                << std::endl;
      continue;
    }

    uint64_t metadata_len;
    if (!infile.read(reinterpret_cast<char *>(&metadata_len),
                     sizeof(metadata_len))) {
      std::cerr << "Error reading metadata length from " << safetensor_file_path
                << std::endl;
      continue;
    }
    std::string metadata_str(metadata_len, '\0');
    if (!infile.read(metadata_str.data(), metadata_len)) {
      std::cerr << "Error reading metadata from " << safetensor_file_path
                << std::endl;
      continue;
    }
    nlohmann::json chunk_metadata;
    try {
      chunk_metadata = nlohmann::json::parse(metadata_str);
    } catch (const nlohmann::json::parse_error &e) {
      std::cerr << "Error parsing JSON metadata in " << safetensor_file_path
                << ": " << e.what() << std::endl;
      continue;
    }

    for (const auto &[local_tensor_name, tensor_info] :
         chunk_metadata.items()) {
      if (local_tensor_name == "__metadata__")
        continue;

      std::string global_tensor_name;
      for (const auto &[global_name, file] : weight_map) {
        if (file == file_name && global_name == local_tensor_name) {
          global_tensor_name = global_name;
          break;
        }
      }
      if (global_tensor_name.empty())
        continue;

      std::string dtype_str = tensor_info["dtype"].get<std::string>();
      std::vector<long long> shape =
          tensor_info["shape"].get<std::vector<long long>>();
      std::vector<int64_t> data_offsets =
          tensor_info["data_offsets"].get<std::vector<int64_t>>();
      int64_t data_start = data_offsets[0];
      int64_t data_end = data_offsets[1];
      size_t tensor_num_bytes = data_end - data_start;

      if (ends_with(global_tensor_name, "_scale_inv")) {
        continue; // Skip scale tensors
      }

      if (dtype_str == "F8_E4M3" &&
          weight_map.count(global_tensor_name + "_scale_inv")) {
        std::vector<uint8_t> quantized_data = load_tensor_data<uint8_t>(
            safetensor_file_path, data_start, tensor_num_bytes);
        std::vector<long long> scale_shape;
        std::string scale_file = weight_map[global_tensor_name + "_scale_inv"];
        // Assuming scale is in the same file for simplicity. Adjust if needed.
        nlohmann::json scale_chunk_metadata;
        std::ifstream scale_infile(fp8_path + "/" + scale_file,
                                   std::ios::binary);
        if (scale_infile.is_open()) {
          uint64_t scale_metadata_len;
          scale_infile.seekg(8, std::ios::beg);
          if (scale_infile.read(reinterpret_cast<char *>(&scale_metadata_len),
                                sizeof(scale_metadata_len))) {
            std::string scale_metadata_str(scale_metadata_len, '\0');
            if (scale_infile.read(scale_metadata_str.data(),
                                  scale_metadata_len)) {
              try {
                scale_chunk_metadata =
                    nlohmann::json::parse(scale_metadata_str);
                if (scale_chunk_metadata.contains(global_tensor_name +
                                                  "_scale_inv")) {
                  auto scale_info =
                      scale_chunk_metadata[global_tensor_name + "_scale_inv"];
                  std::vector<int64_t> scale_offsets =
                      scale_info["data_offsets"].get<std::vector<int64_t>>();
                  int64_t scale_start = scale_offsets[0];
                  int64_t scale_end = scale_offsets[1];
                  size_t scale_num_bytes = scale_end - scale_start;
                  std::vector<float> scale_inv_data =
                      load_tensor_data<float>(fp8_path + "/" + scale_file,
                                              scale_start, scale_num_bytes);
                  if (!quantized_data.empty() && !scale_inv_data.empty() &&
                      shape.size() == 2) {
                    std::vector<bfloat16> bf16_data = weight_dequant_cpu(
                        quantized_data, scale_inv_data, shape[0], shape[1]);
                    size_t bf16_data_size = bf16_data.size() * sizeof(bfloat16);
                    std::vector<char> char_data(
                        reinterpret_cast<const char *>(bf16_data.data()),
                        reinterpret_cast<const char *>(bf16_data.data()) +
                            bf16_data_size);
                    combined_data[global_tensor_name] = {char_data, shape};
                    new_metadata_json[global_tensor_name] = {
                        {"dtype", "BF16"},
                        {"shape", shape},
                        {"data_offsets",
                         {current_offset, current_offset + bf16_data_size}}};
                    current_offset += bf16_data_size;
                  } else {
                    std::cerr << "Warning: Could not dequantize "
                              << global_tensor_name << std::endl;
                  }
                }
              } catch (const nlohmann::json::parse_error &e) {
                std::cerr << "Error parsing scale metadata: " << e.what()
                          << std::endl;
              }
            }
          }
          scale_infile.close();
        }
      } else if (dtype_str == "BF16" || dtype_str == "float32" ||
                 dtype_str == "F32") {
        std::vector<char> tensor_data(tensor_num_bytes);
        infile.seekg(data_start, std::ios::beg);
        if (infile.read(tensor_data.data(), tensor_num_bytes)) {
          std::string target_dtype = "BF16";
          std::vector<char> converted_data;
          if (dtype_str == "float32" || dtype_str == "F32") {
            std::vector<float> float_data(tensor_num_bytes / sizeof(float));
            std::memcpy(float_data.data(), tensor_data.data(),
                        tensor_num_bytes);
            std::vector<bfloat16> bf16_data(float_data.size());
            for (size_t i = 0; i < float_data.size(); ++i) {
              bf16_data[i] = float_to_bfloat16(float_data[i]);
            }
            converted_data.assign(
                reinterpret_cast<const char *>(bf16_data.data()),
                reinterpret_cast<const char *>(bf16_data.data()) +
                    bf16_data.size() * sizeof(bfloat16));
          } else {
            converted_data = tensor_data;
          }
          combined_data[global_tensor_name] = {converted_data, shape};
          new_metadata_json[global_tensor_name] = {
              {"dtype", target_dtype},
              {"shape", shape},
              {"data_offsets",
               {current_offset, current_offset + converted_data.size()}}};
          current_offset += converted_data.size();
        }
      }
    }
    infile.close();
  }

  std::string metadata_str = new_metadata_json.dump();
  uint64_t metadata_len = metadata_str.length();

  std::string output_file_path = bf16_path + "/model.safetensors";
  std::ofstream outfile(output_file_path, std::ios::binary);
  if (!outfile.is_open()) {
    std::cerr << "Error: Could not open output file " << output_file_path
              << std::endl;
    return 1;
  }

  outfile.write(reinterpret_cast<const char *>(&metadata_len),
                sizeof(metadata_len));
  outfile.write(metadata_str.data(), metadata_len);
  std::cout << "Writing combined tensor data..." << std::endl;
  uint64_t written_bytes = 0;
  for (const auto &[_, data_pair] : combined_data) {
    outfile.write(data_pair.first.data(), data_pair.first.size());
    written_bytes += data_pair.first.size();
    update_progress(static_cast<int>(
        (static_cast<double>(written_bytes) / current_offset) * 100.0));
  }
  std::cout << "\nFinished writing tensor data." << std::endl;

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

