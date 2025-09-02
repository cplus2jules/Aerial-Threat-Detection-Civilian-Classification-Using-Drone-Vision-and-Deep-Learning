# Final Project (SALAS): "Aerial Threat Detection: Civilian Classification Using Drone Vision and Deep Learning" 

_Developed for the coursework CSC 126_

**IMPORTANT NOTE:**
*   This `run_yolo_look.py` script is primarily a **Python (.py) version for running detections using a pre-trained model.**
*   While the script includes a training function, **it was not the primary method used or extensively tested for training the model for this project.**
*   **For details on the model training process, please refer to the original Google Colab notebook (`.ipynb` file).** It is recommended to train your model using the Colab notebook, download the resulting weights (`best.pt` file), and then use *this* local script for running detections with those downloaded weights.
---

## 💻 System Requirements
Before you begin, make sure you have the following installed on your computer:
*   **Python:** Version 3.8 or newer. You can download it from [python.org](https://www.python.org/downloads/).
*   **pip:** Python's package installer. It usually comes with Python.

## ⚙️ Setup Instructions

Follow these steps carefully to get the project ready to run:

**Step 1: Get the Project Files**
*   Download all the project files and folders. This includes `run_yolo_look.py`, and you should create folders like `trained_models/`, `input_media/`, and `output_media/`.
*   Place them together in one main project folder on your computer (e.g., `C:\Users\YourName\Desktop\YOLO_Detection_Project` or `/Users/YourName/Documents/YOLO_Detection_Project`). We'll call this your "project directory".

**Step 2: Install Required Software Libraries**
1.  Open your computer's command line interface:
    *   **Windows:** Search for "Command Prompt" or "PowerShell".
    *   **macOS/Linux:** Search for "Terminal".
2.  Navigate to your project directory using the `cd` command. For example:
    ```bash
    cd /Users/YourName/Documents/YOLO_Detection_Project
    ```
    (Replace with the actual path to your project directory).
3.  Install the necessary Python libraries by running this command:
    ```bash
    pip install ultralytics opencv-python
    ```
    This will download and install the `ultralytics` library (for YOLO) and `opencv-python` (for image and video processing).

**Step 3: Folder Structure (Recommended for Detections)**
For running detections effectively, organize your files as follows:

your_project_directory/
├── run_yolo_local.py # The main Python script
├── trained_models/ # << Place your pre-trained .pt model file HERE
│ └── your_best_model.pt # (e.g., best.pt from Colab)
├── input_media/ # << Place images/videos you want to test HERE
│ ├── test_image.jpg
│ └── test_video.mp4
└── output_media/ # Annotated images/videos will be saved here by the script
├── images/
└── videos/

*   Create these folders if they don't exist.
*   Place your pre-trained model file (e.g., `best.pt` that you downloaded) into the `trained_models/` folder.
*   Place any images or videos you want to test into the `input_media/` folder.

## ▶️ How to Use the Script for Detection

You will run the script from your command line interface (Command Prompt, PowerShell, or Terminal) that you opened in Setup Step 2. Make sure you are in your project directory.

The script uses command-line arguments to know what to do and which files to use.

**Core Argument for Detection:**
*   `--action`: You MUST specify what you want to do.
    *   Use `--action predict_image` for detecting on a single image.
    *   Use `--action predict_video` for detecting on a video.

**Key Arguments for Specifying Files (Important!):**
*   `--trained_model "path/to/your/model.pt"`:
    *   Tells the script where to find your pre-trained AI model.
    *   Example: `--trained_model "trained_models/best.pt"`
*   `--image_input "path/to/your/image.jpg"`:
    *   Tells the script which image to process (used with `--action predict_image`).
    *   Example: `--image_input "input_media/my_photo.jpg"`
*   `--video_input "path/to/your/video.mp4"`:
    *   Tells the script which video to process (used with `--action predict_video`).
    *   Example: `--video_input "input_media/my_drone_footage.mp4"`

**Optional Arguments for Prediction:**
*   `--output_dir_img "path/to/save/images/"`: Specifies where to save annotated images. (Default is `output_media/images/`)
*   `--output_dir_vid "path/to/save/videos/"`: Specifies where to save annotated videos. (Default is `output_media/videos/`)
*   `--display`: If you add this when using `--action predict_video`, a window will pop up showing the video frames being processed. This can be slower. (Example: `python run_yolo_look.py --action predict_video ... --display`)

**Example Commands for Detection:**

1.  **To Detect People in a Single Image:**
    *   Let's say your model is `trained_models/my_aerial_model.pt`
    *   And your image is `input_media/scene1.jpg`
    ```bash
    python run_yolo_look.py --action predict_image --trained_model "trained_models/my_aerial_model.pt" --image_input "input_media/scene1.jpg"
    ```
    *   An image window will pop up with detections. Press any key to close it.
    *   The result will be saved in `output_media/images/scene1_annotated.jpg`.

2.  **To Detect People in a Video:**
    *   Model: `trained_models/my_aerial_model.pt`
    *   Video: `input_media/flight_footage.mp4`
    ```bash
    python run_yolo_local.py --action predict_video --trained_model "trained_models/my_aerial_model.pt" --video_input "input_media/flight_footage.mp4"
    ```
    *   The processed video will be saved in `output_media/videos/flight_footage_annotated.mp4`.

3.  **Using Default Paths Set in the Script:**
    If you've carefully set the `DEFAULT_TRAINED_MODEL`, `DEFAULT_IMAGE_TO_PREDICT`, etc., variables *inside* the `run_yolo_local.py` script to point to your desired files, you *can* run a simpler command:
    ```bash
    python run_yolo_local.py --action predict_image
    ```
    However, for clarity and avoiding errors, **it's often better to specify the paths directly in the command line as shown in examples 1 and 2.**

---
**Training a New Model (Reference Only - See Important Note at the Top)**
*   While the script contains a training function, it's recommended to use the Colab notebook for training.
*   If you were to use the local script for training (ensure dataset is prepared as per earlier Setup Step 4 for training):
    ```bash
    python run_yolo_local.py --action train --data_yaml "datasets/Your_Dataset_Name/data.yaml" --run_name "local_training_experiment" --epochs 50
    ```
    Training is computationally intensive and requires a correctly formatted dataset.
---

## 💡 Understanding the Output
*   **Image Prediction:** An annotated image (with boxes drawn around detected people) will be saved in the folder specified by `--output_dir_img` (default: `output_media/images/`). An image window will also pop up during the process.
*   **Video Prediction:** An annotated video (with boxes drawn around detected people in each frame) will be saved in the folder specified by `--output_dir_vid` (default: `output_media/videos/`).

