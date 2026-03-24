# "Aerial Threat Detection: Civilian Classification Using Drone Vision and Deep Learning" 

_Developed as a final project for the coursework CSC 126_

**IMPORTANT NOTE:**
*   This `run_yolo_look.py` script is primarily a **Python (.py) version for running detections using a pre-trained model.**
*   While the script includes a training function, **it was not the primary method used or extensively tested for training the model for this project.**
*   **For details on the model training process, please refer to the original Google Colab notebook (`.ipynb` file).** It is recommended to train your model using the Colab notebook, download the resulting weights (`best.pt` file), and then use *this* local script for running detections with those downloaded weights.
*   You can directly check the Output and Test folders to see the already processed files.
---

**Instruction to run the project locally. _(Please contact me via email at julian.salas@carsu.edu.ph if you encounter any problems)_**

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
