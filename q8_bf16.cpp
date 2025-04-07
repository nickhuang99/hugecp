#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <numeric> // For std::accumulate
#include <string>
#include <vector>

// Assume these utility functions are defined elsewhere
bool ends_with(const std::string &str, const std::string &suffix);
typedef uint16_t bfloat16;
bfloat16 float_to_bfloat16(float f);
float bfloat16_to_float(bfloat16 bf);
std::vector<bfloat16>
weight_dequant_cpu(const std::vector<uint8_t> &quantized_weight,
                   const std::vector<float> &scale_inv, long long M,
                   long long N, int block_size);
template <typename T>
std::vector<T> load_tensor_data(const std::string &filename, int64_t offset,
                                size_t num_bytes);
std::vector<bfloat16>
dequantizeOneweight(const std::string &weight_name,
                    const std::string &model_path,
                    const std::map<std::string, std::string> &weight_map,
                    const std::map<std::string, std::vector<nlohmann::json>>
                        &chunk_weight_details);
void writeOneTensorToFile(std::ofstream &outfile,
                          const std::vector<bfloat16> &tensor_data);
void writeOneTensorToFile(std::ofstream &outfile,
                          const std::vector<char> &tensor_data);
std::pair<nlohmann::json, std::map<std::string, std::vector<nlohmann::json>>>
calculateMetaDataRevised(const std::string &model_path);
void update_progress(int progress); // Assume this is defined

std::pair<nlohmann::json, std::map<std::string, std::vector<nlohmann::json>>>
calculateMetaDataRevised(const std::string &model_path) {
  nlohmann::json final_metadata_json;
  final_metadata_json["__metadata__"] = {{"format", "pt"}};
  std::map<std::string, std::vector<nlohmann::json>> chunk_weight_details;
  uint64_t current_offset = 0;

  std::string model_index_file = model_path + "/model.safetensors.index.json";
  std::ifstream f(model_index_file);
  if (!f.is_open()) {
    std::cerr << "Error: Could not open " << model_index_file << std::endl;
    return {final_metadata_json, chunk_weight_details};
  }
  nlohmann::json model_index;
  f >> model_index;
  f.close();

  auto weight_map =
      model_index["weight_map"].get<std::map<std::string, std::string>>();

  std::map<std::string, nlohmann::json> all_chunk_metadata;
  std::vector<std::string> safetensor_files;
  for (const auto &entry : std::filesystem::directory_iterator(model_path)) {
    if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
      safetensor_files.push_back(entry.path().filename().string());
    }
  }
  std::sort(safetensor_files.begin(), safetensor_files.end());

  // Read all chunk metadata
  for (const auto &file_name : safetensor_files) {
    std::string safetensor_file_path = model_path + "/" + file_name;
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
      infile.close();
      continue;
    }
    std::string metadata_str(metadata_len, '\0');
    if (!infile.read(metadata_str.data(), metadata_len)) {
      std::cerr << "Error reading metadata from " << safetensor_file_path
                << std::endl;
      infile.close();
      continue;
    }
    try {
      all_chunk_metadata[file_name] = nlohmann::json::parse(metadata_str);
    } catch (const nlohmann::json::parse_error &e) {
      std::cerr << "Error parsing JSON metadata in " << safetensor_file_path
                << ": " << e.what() << std::endl;
    }
    infile.close();
  }

  // Pre-calculate final metadata and chunk weight details
  for (const auto &[global_tensor_name, chunk_file_name] : weight_map) {
    if (all_chunk_metadata.count(chunk_file_name) &&
        all_chunk_metadata[chunk_file_name].count(global_tensor_name)) {
      nlohmann::json tensor_info =
          all_chunk_metadata[chunk_file_name][global_tensor_name];
      std::string dtype_str = tensor_info["dtype"].get<std::string>();
      std::vector<long long> shape =
          tensor_info["shape"].get<std::vector<long long>>();
      std::vector<int64_t> data_offsets =
          tensor_info["data_offsets"].get<std::vector<int64_t>>();

      nlohmann::json weight_detail;
      weight_detail["name"] = global_tensor_name;
      weight_detail["dtype"] = dtype_str;
      weight_detail["shape"] = shape;
      weight_detail["data_offsets"] = data_offsets;

      if (chunk_weight_details.find(chunk_file_name) ==
          chunk_weight_details.end()) {
        chunk_weight_details[chunk_file_name] = {weight_detail};
      } else {
        chunk_weight_details[chunk_file_name].push_back(weight_detail);
      }

      if (dtype_str == "F8_E4M3" || dtype_str == "BF16" ||
          dtype_str == "float32" || dtype_str == "F32") {
        if (shape.size() != 2) {
          std::cerr << "Error: Tensor " << global_tensor_name
                    << " has shape of size " << shape.size()
                    << ", which is not 2. Skipping for BF16 conversion."
                    << std::endl;
          // Copy original metadata
          final_metadata_json[global_tensor_name] = tensor_info;
          final_metadata_json[global_tensor_name]["data_offsets"] = {
              current_offset,
              current_offset +
                  (tensor_info.contains("num_bytes")
                       ? tensor_info["num_bytes"].get<size_t>()
                       : (shape.empty()
                              ? 0
                              : std::accumulate(shape.begin(), shape.end(), 1LL,
                                                std::multiplies<long long>()) *
                                    (dtype_str == "BF16"
                                         ? sizeof(bfloat16)
                                         : (dtype_str == "float32" ||
                                                    dtype_str == "F32"
                                                ? sizeof(float)
                                                : 1))))};
          current_offset +=
              final_metadata_json[global_tensor_name]["data_offsets"][1]
                  .get<uint64_t>() -
              final_metadata_json[global_tensor_name]["data_offsets"][0]
                  .get<uint64_t>();

        } else {
          size_t tensor_size_bytes =
              std::accumulate(shape.begin(), shape.end(), 1LL,
                              std::multiplies<long long>()) *
              sizeof(bfloat16);
          final_metadata_json[global_tensor_name] = {
              {"dtype", "BF16"},
              {"shape", shape},
              {"data_offsets",
               {current_offset, current_offset + tensor_size_bytes}}};
          current_offset += tensor_size_bytes;
        }
      } else {
        // Copy metadata for other dtypes
        final_metadata_json[global_tensor_name] = tensor_info;
        final_metadata_json[global_tensor_name]["data_offsets"] = {
            current_offset,
            current_offset +
                (tensor_info.contains("num_bytes")
                     ? tensor_info["num_bytes"].get<size_t>()
                     : (shape.empty()
                            ? 0
                            : std::accumulate(shape.begin(), shape.end(), 1LL,
                                              std::multiplies<long long>()) *
                                  (dtype_str == "BF16"
                                       ? sizeof(bfloat16)
                                       : (dtype_str == "float32" ||
                                                  dtype_str == "F32"
                                              ? sizeof(float)
                                              : 1))))};
        current_offset +=
            final_metadata_json[global_tensor_name]["data_offsets"][1]
                .get<uint64_t>() -
            final_metadata_json[global_tensor_name]["data_offsets"][0]
                .get<uint64_t>();
      }
    }
  }

  return {final_metadata_json, chunk_weight_details};
}
bool ends_with(const std::string &str, const std::string &suffix) {
  return str.size() >= suffix.size() &&
         0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

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
void writeOneTensorToFile(std::ofstream &outfile,
                          const std::vector<bfloat16> &tensor_data) {
  if (outfile.is_open() && !tensor_data.empty()) {
    outfile.write(reinterpret_cast<const char *>(tensor_data.data()),
                  tensor_data.size() * sizeof(bfloat16));
  } else if (!outfile.is_open()) {
    std::cerr << "Error: Output file is not open." << std::endl;
  } else if (tensor_data.empty()) {
    std::cerr << "Warning: Tensor data is empty, nothing to write."
              << std::endl;
  }
}

void writeOneTensorToFile(std::ofstream &outfile,
                          const std::vector<char> &tensor_data) {
  if (outfile.is_open() && !tensor_data.empty()) {
    outfile.write(tensor_data.data(), tensor_data.size());
  } else if (!outfile.is_open()) {
    std::cerr << "Error: Output file is not open." << std::endl;
  } else if (tensor_data.empty()) {
    std::cerr << "Warning: Tensor data is empty, nothing to write."
              << std::endl;
  }
}

// Assume these utility functions are defined elsewhere
bool ends_with(const std::string &str, const std::string &suffix);
typedef uint16_t bfloat16;
bfloat16 float_to_bfloat16(float f);
float bfloat16_to_float(bfloat16 bf);
std::vector<bfloat16>
weight_dequant_cpu(const std::vector<uint8_t> &quantized_weight,
                   const std::vector<float> &scale_inv, long long M,
                   long long N, int block_size);
template <typename T>
std::vector<T> load_tensor_data(const std::string &filename, int64_t offset,
                                size_t num_bytes);

std::vector<bfloat16>
dequantizeOneweight(const std::string &weight_name,
                    const std::string &model_path,
                    const std::map<std::string, std::string>
                        &weight_map, // We might not even need this anymore!
                    const std::map<std::string, std::vector<nlohmann::json>>
                        &chunk_weight_details) {

  if (!weight_map.count(weight_name)) {
    std::cerr << "Error: Weight name '" << weight_name
              << "' not found in weight map." << std::endl;
    return {};
  }

  std::string chunk_file_name = weight_map.at(weight_name);
  if (!chunk_weight_details.count(chunk_file_name)) {
    std::cerr << "Error: Chunk file details for '" << chunk_file_name
              << "' not found." << std::endl;
    return {};
  }

  const auto &weight_list = chunk_weight_details.at(chunk_file_name);
  nlohmann::json weight_info;
  bool found = false;
  for (const auto &wd : weight_list) {
    if (wd["name"].get<std::string>() == weight_name) {
      weight_info = wd;
      found = true;
      break;
    }
  }

  if (!found) {
    std::cerr << "Error: Details for weight '" << weight_name
              << "' not found in chunk details." << std::endl;
    return {};
  }

  std::string dtype_str = weight_info["dtype"].get<std::string>();
  std::vector<long long> shape =
      weight_info["shape"].get<std::vector<long long>>();
  std::vector<int64_t> data_offsets =
      weight_info["data_offsets"].get<std::vector<int64_t>>();
  int64_t data_start = data_offsets[0];
  size_t tensor_num_bytes =
      (data_offsets.size() > 1 ? data_offsets[1] : 0) - data_start;
  std::string safetensor_file_path = model_path + "/" + chunk_file_name;

  if (dtype_str == "F8_E4M3" && weight_map.count(weight_name + "_scale_inv")) {
    std::vector<uint8_t> quantized_data = load_tensor_data<uint8_t>(
        safetensor_file_path, data_start, tensor_num_bytes);

    std::string scale_name = weight_name + "_scale_inv";
    std::string scale_file_name;
    bool scale_file_found = false;
    for (const auto &[file, details] : chunk_weight_details) {
      for (const auto &detail : details) {
        if (detail["name"].get<std::string>() == scale_name) {
          scale_file_name = file;
          scale_file_found = true;
          break;
        }
      }
      if (scale_file_found) {
        break;
      }
    }

    if (!scale_file_found) {
      std::cerr << "Error: Chunk file for scale tensor '" << scale_name
                << "' not found." << std::endl;
      return {};
    }

    const auto &scale_list = chunk_weight_details.at(scale_file_name);
    nlohmann::json scale_info;
    bool scale_found_in_chunk = false;
    for (const auto &sd : scale_list) {
      if (sd["name"].get<std::string>() == scale_name) {
        scale_info = sd;
        scale_found_in_chunk = true;
        break;
      }
    }
    if (!scale_found_in_chunk) {
      std::cerr << "Error: Details for scale '" << scale_name
                << "' not found in chunk details." << std::endl;
      return {};
    }
    std::vector<int64_t> scale_offsets =
        scale_info["data_offsets"].get<std::vector<int64_t>>();
    int64_t scale_start = scale_offsets[0];
    size_t scale_num_bytes =
        (scale_offsets.size() > 1 ? scale_offsets[1] : 0) - scale_start;
    std::vector<float> scale_inv_data = load_tensor_data<float>(
        model_path + "/" + scale_file_name, scale_start, scale_num_bytes);

    if (!quantized_data.empty() && !scale_inv_data.empty() &&
        shape.size() == 2) {
      return weight_dequant_cpu(quantized_data, scale_inv_data, shape[0],
                                shape[1]);
    } else {
      std::cerr << "Warning: Could not dequantize FP8 weight '" << weight_name
                << "' due to missing data or incorrect shape." << std::endl;
      return {};
    }
  } else if (dtype_str == "BF16") {
    return load_tensor_data<bfloat16>(safetensor_file_path, data_start,
                                      tensor_num_bytes);
  } else if (dtype_str == "float32" || dtype_str == "F32") {
    std::vector<float> float_data = load_tensor_data<float>(
        safetensor_file_path, data_start, tensor_num_bytes);
    std::vector<bfloat16> bf16_data(float_data.size());
    for (size_t i = 0; i < float_data.size(); ++i) {
      bf16_data[i] = float_to_bfloat16(float_data[i]);
    }
    return bf16_data;
  } else {
    std::cerr << "Warning: Skipping dequantization/conversion for dtype '"
              << dtype_str << "' of weight '" << weight_name << "'."
              << std::endl;
    return {};
  }
}
int main(int argc, char *argv[]) {
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage: " << argv[0]
              << " <input_fp8_path> <output_bf16_path> [--dry-run]"
              << std::endl;
    return 1;
  }

  std::string fp8_path = argv[1];
  std::string bf16_path = argv[2];
  bool dry_run = false;

  if (argc == 4 && std::string(argv[3]) == "--dry-run") {
    dry_run = true;
    std::cout << "Dry-run mode enabled. No output files will be written."
              << std::endl;
  }

  // 1. Calculate Metadata
  auto [final_metadata, chunk_details_map] = calculateMetaDataRevised(fp8_path);

  std::cout << "\n--- Final Metadata (Dry-Run) ---" << std::endl;
  std::cout << std::setw(4) << final_metadata << std::endl;
  if (dry_run &&
      argc == 3) { // If only input/output paths are given with dry-run
    return 0;      // Just print metadata and exit
  }

  if (!dry_run) {
    std::filesystem::create_directories(bf16_path);

    // 2. Prepare Final Result File and Write Metadata
    std::string metadata_str = final_metadata.dump();
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

    // Load the index once for the initial weight mapping
    std::string model_index_file = fp8_path + "/model.safetensors.index.json";
    std::ifstream f_index(model_index_file);
    nlohmann::json model_index;
    f_index >> model_index;
    f_index.close();
    auto weight_map =
        model_index["weight_map"].get<std::map<std::string, std::string>>();

    std::cout << "Processing and writing weights..." << std::endl;
    int weight_counter = 0;
    int num_weights = final_metadata.size() -
                      (final_metadata.contains("__metadata__") ? 1 : 0);

    for (const auto &[weight_name, tensor_info] : final_metadata.items()) {
      if (weight_name == "__metadata__") {
        continue;
      }
      update_progress((weight_counter++) * 100 / num_weights);

      std::string dtype_str = tensor_info["dtype"].get<std::string>();

      if (dtype_str == "F8_E4M3") {
        std::vector<bfloat16> bf16_tensor = dequantizeOneweight(
            weight_name, fp8_path, weight_map, chunk_details_map);
        if (!bf16_tensor.empty()) {
          writeOneTensorToFile(outfile, bf16_tensor);
        } else {
          std::cerr << "Warning: Skipping writing empty dequantized tensor "
                    << weight_name << std::endl;
        }
      } else if (dtype_str == "BF16" || dtype_str == "float32" ||
                 dtype_str == "F32") {
        std::vector<bfloat16> bf16_tensor = dequantizeOneweight(
            weight_name, fp8_path, weight_map, chunk_details_map);
        if (!bf16_tensor.empty()) {
          writeOneTensorToFile(outfile, bf16_tensor);
        } else {
          std::cerr << "Warning: Skipping writing empty converted tensor "
                    << weight_name << std::endl;
        }
      } else {
        // Copy original data for other types
        if (tensor_info.contains("chunk_file")) {
          std::string chunk_file_name =
              tensor_info["chunk_file"].get<std::string>();
          if (chunk_details_map.count(chunk_file_name)) {
            const auto &weight_list = chunk_details_map.at(chunk_file_name);
            for (const auto &wd : weight_list) {
              if (wd["name"].get<std::string>() == weight_name) {
                std::vector<int64_t> original_offsets =
                    wd["data_offsets"].get<std::vector<int64_t>>();
                int64_t original_start = original_offsets[0];
                size_t original_num_bytes =
                    (original_offsets.size() > 1 ? original_offsets[1] : 0) -
                    original_start;
                std::vector<char> original_tensor_data =
                    load_tensor_data<char>(fp8_path + "/" + chunk_file_name,
                                           original_start, original_num_bytes);
                writeOneTensorToFile(outfile, original_tensor_data);
                break;
              }
            }
          }
        }
      }
    }
    std::cout << "\nFinished writing weight data." << std::endl;
    outfile.close();

    // Create the new index file
    nlohmann::json new_index_json;
    new_index_json["weight_map"] = nlohmann::json::object();
    for (const auto &item : final_metadata.items()) {
      if (item.key() != "__metadata__") {
        new_index_json["weight_map"][item.key()] = "model.safetensors";
      }
    }

    std::ofstream index_outfile(bf16_path + "/model.safetensors.index.json");
    index_outfile << std::setw(4) << new_index_json << std::endl;
    index_outfile.close();

    std::cout << "Dequantization and merging complete. BF16 model saved to "
              << bf16_path << std::endl;
  } else {
    std::cout << "\nDry-run complete. No output files were written."
              << std::endl;
  }

  return 0;
}