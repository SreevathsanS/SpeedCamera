# SpeedCamera: Automatic Number Plate Recognition (ANPR) & Vehicle Speed Detection

## Overview
SpeedCamera is a full-stack project for automatic number plate recognition (ANPR), vehicle detection, and speed monitoring using computer vision and deep learning. It features a Python backend for video/image analysis and a modern React frontend for dashboard visualization and management.

---

## Features
- **Vehicle Detection & Tracking:** Detects vehicles in video streams using YOLO models.
- **Number Plate Recognition:** Identifies and extracts license plates from detected vehicles.
- **Speed Calculation:** Tracks vehicle movement and calculates speed using perspective transformation.
- **Challan Generation:** Automatically generates challans for overspeeding vehicles.
- **Database Logging:** Stores logs and challan data in SQLite.
- **Admin & Public Dashboards:** React-based dashboards for monitoring, viewing logs, and managing data.
- **Authentication:** Basic authentication for admin access.

---

## Project Structure
```
Backend/
  api/           # FastAPI server and authentication
  data/          # Sample images and videos
  models/        # YOLO models for vehicle and plate detection
  outputs/       # Database and output folders
  scripts/       # Main logic, utilities, and database management
frontend/
  src/           # React app source code
  public/        # Static assets
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- pip (Python package manager)


### Backend Setup
#### 1. Install NVIDIA GPU Driver, CUDA Toolkit, cuDNN, and TensorRT (for GPU acceleration)
See the top of `Backend/requirements.txt` for complete installation instructions:

1. **Install NVIDIA GPU Driver**
   - https://www.nvidia.com/Download/index.aspx
2. **Install CUDA Toolkit**
   - https://developer.nvidia.com/cuda-downloads
3. **Install cuDNN**
   - https://developer.nvidia.com/cudnn
   - Extract and copy files to your CUDA directory.
4. **Install TensorRT**
   - https://developer.nvidia.com/tensorrt
   - Extract and add TensorRT 'bin' directory to your system PATH.
5. **Install Python TensorRT bindings**
   - pip install <path_to_tensorrt_wheel_file.whl>
6. **Verify Installation**
   - import tensorrt
   - print(tensorrt.__version__)

#### 2. Install Python dependencies
   ```powershell
   cd Backend
   pip install -r requirements.txt
   ```


#### 3. Run the backend server (handles all backend functions)
   ```powershell
   python api/server.py
   ```

#### (Optional) Run camera feed for testing only
   ```powershell
   python scripts/main.py
   ```


### Frontend Setup
1. **Install dependencies:**
   ```powershell
   cd frontend
   npm install
   ```
2. **Start frontend development server:**
   ```powershell
   npm run dev
   ```

---


## Usage
- To run the full project, start the backend server:
   ```powershell
   python Backend/api/server.py
   ```
- Then start the frontend:
   ```powershell
   cd frontend
   npm run dev
   ```
- Access dashboards via the frontend (`http://localhost:5173` by default).
- Upload videos/images to `Backend/data/videos` or `Backend/data/images`.
- View logs, detected plates, and generated challans in `Backend/outputs`.

- For camera feed testing only, run:
   ```powershell
   python Backend/scripts/main.py
   ```

---

## Model Files
- YOLO models for vehicle and plate detection are stored in `Backend/models/VehicleDetector` and `Backend/models/NumberPlateDetector`.
- You may need to download or train your own models for best results.

---

## Contributing
Pull requests and suggestions are welcome! Please open issues for bugs or feature requests.

---

## License
This project is licensed under the MIT License.

---

## Authors
- Sreevathsan S
- Contributors: [Add your name here]
