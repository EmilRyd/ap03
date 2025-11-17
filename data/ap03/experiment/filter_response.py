#%%
import numpy as np

# Read data from ch9.txt
data_file = '/Users/emilryd/Personal/school/Oxford/Computing and Data Analysis/third year/AP03/data/ap03/data/ch9.txt'

wavelength = []
response = []

with open(data_file, 'r') as f:
    for line in f:
        line = line.strip()
        # Skip comment lines, metadata, and end marker
        if line.startswith('!') or '=' in line or line == '---' or line == '':
            continue
        # Parse data lines with comma-separated values
        if ',' in line:
            parts = line.split(',')
            wavelength.append(float(parts[0]))
            response.append(float(parts[1]))

# Convert to numpy arrays
wavelength = np.array(wavelength)
response = np.array(response)

# Calculate dot product of wavelength and response
dot_product = np.dot(wavelength, response)

# Calculate L1 norm of response vector (sum of absolute values)
response_norm = np.sum(np.abs(response))

# Divide dot product by L1 norm of response
result = dot_product / response_norm

print("Channel 9:")
print(f"Number of data points: {len(wavelength)}")
print(f"Dot product (wavelength · response): {dot_product:.6f}")
print(f"L1 norm of response vector: {response_norm:.6f}")
print(f"Result (dot product / L1 norm): {result:.6f}")


# %%
# Read data from ch10.txt
data_file = '/Users/emilryd/Personal/school/Oxford/Computing and Data Analysis/third year/AP03/data/ap03/data/ch10.txt'

wavelength = []
response = []

with open(data_file, 'r') as f:
    for line in f:
        line = line.strip()
        # Skip comment lines, metadata, and end marker
        if line.startswith('!') or '=' in line or line.startswith('---') or line == '':
            continue
        # Parse data lines with comma-separated values
        if ',' in line:
            parts = line.split(',')
            wavelength.append(float(parts[0]))
            response.append(float(parts[1]))

# Convert to numpy arrays
wavelength = np.array(wavelength)
response = np.array(response)

# Calculate dot product of wavelength and response
dot_product = np.dot(wavelength, response)

# Calculate L1 norm of response vector (sum of absolute values)
response_norm = np.sum(np.abs(response))

# Divide dot product by L1 norm of response
result = dot_product / response_norm

print("\nChannel 10:")
print(f"Number of data points: {len(wavelength)}")
print(f"Dot product (wavelength · response): {dot_product:.6f}")
print(f"L1 norm of response vector: {response_norm:.6f}")
print(f"Result (dot product / L1 norm): {result:.6f}")


# %%
