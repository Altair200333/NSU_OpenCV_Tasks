file = open('src/data.txt', mode='r')
text = file.read()

print(text)

import base64
import numpy as np

vtu_to_numpy_type = {
    "Float32": np.dtype(np.float32),
    "Float64": np.dtype(np.float64),
    "Int8": np.dtype(np.int8),
    "Int16": np.dtype(np.int16),
    "Int32": np.dtype(np.int32),
    "Int64": np.dtype(np.int64),
    "UInt8": np.dtype(np.uint8),
    "UInt16": np.dtype(np.uint16),
    "UInt32": np.dtype(np.uint32),
    "UInt64": np.dtype(np.uint64),
}
numpy_to_vtu_type = {v: k for k, v in vtu_to_numpy_type.items()}

header_type = "UInt64"
data = np.arange(10, dtype=np.float64)

data_bytes = data.tobytes()
# collect header
header = np.array(
    len(data_bytes), dtype=vtu_to_numpy_type[header_type]
)
print(base64.b64encode(header.tobytes() + data_bytes).decode())
print(base64.b64encode(header.tobytes()).decode())
print(base64.b64encode(header.tobytes()).decode() + base64.b64encode(data_bytes).decode())