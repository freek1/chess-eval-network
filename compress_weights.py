class FloatCompressor:
    def __init__(self, min_value, max_value, num_unique_values):
        self.min_value = min_value
        self.max_value = max_value
        self.max_value_ulong = num_unique_values - 1
        self.range = round((max_value - min_value) / num_unique_values, 5)

    def compress(self, value):
        value = max(self.min_value, min(self.max_value, value))
        compressed_value = round((value - self.min_value) / self.range)
        return min(compressed_value, self.max_value_ulong)  # Clamp to the maximum representable value

    def decompress(self, compressed_value):
        compressed_value = min(compressed_value, self.max_value_ulong)  # Clamp to the maximum representable value
        decompressed_value = self.min_value + compressed_value * self.range
        return decompressed_value

# Example usage:
# Define compression parameters (you can adjust these as needed)
min_value = -1.1
max_value = 1.1
num_unique_values = 23093 # /8

# Initialize the compressor
compressor = FloatCompressor(min_value, max_value, num_unique_values)

# Define the file path
file_path = "geohotz_20k_cpu.txt"  # Replace with the actual file path

# Read the file and load weights into a list
with open(file_path, 'r') as file:
    original_float_array = [float(line.strip()) for line in file if '[' not in line]

# Compress the float array into a ulong array
compressed_array = [compressor.compress(f) for f in original_float_array]

# Save the compressed data to a new file
with open('geohotz_20k_comp.txt', 'w') as file:
    for compressed_value in compressed_array:
        file.write(f"{compressed_value}\n")

# Decompress the ulong array back to the original float array
decompressed_array = [compressor.decompress(c) for c in compressed_array]

