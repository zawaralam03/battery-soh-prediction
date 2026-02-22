# 📊 Dataset Information

   ## CALCE Battery Dataset

   This project uses the **CALCE Battery Research Group dataset**.

   ### Download Instructions

   1. Visit: https://web.calce.umd.edu/batteries/data.htm
   2. Download the following battery data:
      - CS2_35
      - CS2_36
      - CS2_37
      - CS2_38
   3. Extract the files

   ### Dataset Structure

   After extraction, organize your data folder like this:
   Dataset/
├── CS2_35/
│   ├── charge/
│   └── discharge/
├── CS2_36/
├── CS2_37/
└── CS2_38/

### Update Path in Notebook

   In the notebook, update the path to your dataset:
```python
   dir_path = "/path/to/your/Dataset/"
```

   ### Features

   The dataset includes:
   - **Voltage** measurements
   - **Current** profiles
   - **Temperature** readings
   - **Capacity** data
   - **SOH** (State of Health) values

   ### Drive Cycles

   - **US06**: Highway driving (high power)
   - **FUDS**: Urban driving (stop-and-go)
   - **DST**: Dynamic stress test

   ### Note

   ⚠️ Dataset files are NOT included in this repository due to size.  
   Please download separately from the CALCE website.
