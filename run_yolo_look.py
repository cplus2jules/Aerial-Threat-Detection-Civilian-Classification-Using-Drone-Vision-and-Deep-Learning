# (SALAS) for pre-trained models only

import cv2
import os
from ultralytics import YOLO
import argparse 

DEFAULT_BASE_MODEL = "yolov8n.pt"

# Dataset paths (for training)
DEFAULT_DATASET_DIR = "datasets/Look_Down_Folks_Dataset" 
DEFAULT_DATA_YAML = os.path.join(DEFAULT_DATASET_DIR, "data.yaml")

# Path to your pre-trained model (after training)
DEFAULT_TRAINED_MODEL = "/Users/juliansalas/Desktop/CSC 126/LABS/FINAL LAB PROJECT/weights/best.pt"

# Input media for prediction
DEFAULT_IMAGE_TO_PREDICT = "/Users/juliansalas/Desktop/CSC 126/LABS/FINAL LAB PROJECT/test/images/test.jpeg" 
DEFAULT_VIDEO_TO_PREDICT = "/Users/juliansalas/Desktop/CSC 126/LABS/FINAL LAB PROJECT/test/videos/People Walking in the Street Drone Footage.mp4" 

# Output paths for predictions
DEFAULT_OUTPUT_IMAGE_DIR = "/Users/juliansalas/Desktop/CSC 126/LABS/FINAL LAB PROJECT/output/images"
DEFAULT_OUTPUT_VIDEO_DIR = "/Users/juliansalas/Desktop/CSC 126/LABS/FINAL LAB PROJECT/output/videos"

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
def train_model(data_yaml_path, base_model_name, epochs, img_size, batch_size, patience, run_name):
    print(f"Starting training with data: {data_yaml_path}")
    model = YOLO(base_model_name)
    
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        patience=patience,
        name=run_name,
        project="runs/detect"
    )
    
    print("Training finished.")
    print(f"Results saved to: {results.save_dir}")
    print(f"Best model saved at: {os.path.join(results.save_dir, 'weights/best.pt')}")

# --- Image Prediction Function ---
def predict_on_image(model_path, image_path, output_dir):
    ensure_dir(output_dir)
    print(f"Loading model for image prediction: {model_path}")
    model = YOLO(model_path)
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print(f"Predicting on image: {image_path}")
    results = model(image_path)
    
    for i, r in enumerate(results):
        annotated_frame = r.plot()
        
        cv2.imshow(f"Annotated Image {i+1}", annotated_frame)
        
        base_filename = os.path.basename(image_path)
        name, ext = os.path.splitext(base_filename)
        output_filename = os.path.join(output_dir, f"{name}_annotated{ext}")
        cv2.imwrite(output_filename, annotated_frame)
        print(f"Annotated image saved to: {output_filename}")

    print("Press any key to close image windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Video Prediction Function ---
def predict_on_video(model_path, video_path, output_dir, display_video=False):
    ensure_dir(output_dir)
    print(f"Loading model for video prediction: {model_path}")
    model = YOLO(model_path)

    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    base_filename = os.path.basename(video_path)
    name, ext = os.path.splitext(base_filename)
    output_filename = os.path.join(output_dir, f"{name}_annotated.mp4") # Always save as .mp4 for simplicity

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Error: Could not open VideoWriter for output path {output_filename}")
        cap.release()
        return

    print(f"Processing video: {video_path}")
    print(f"Output will be saved to: {output_filename}")
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processing frame: {frame_count}")

            results = model(frame, stream=True, verbose=False)
            
            annotated_frame = None
            for r in results:
                annotated_frame = r.plot()
                out.write(annotated_frame)

            if display_video and annotated_frame is not None:
                cv2.imshow("Annotated Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break
            
    print(f"Finished processing {frame_count} frames.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Annotated video saved successfully to: {output_filename}")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Local Runner for Training and Prediction")
    parser.add_argument("--action", required=True, choices=["train", "predict_image", "predict_video"],
                        help="Action to perform: train, predict_image, or predict_video")
    
    parser.add_argument("--data_yaml", default=DEFAULT_DATA_YAML, help="Path to data.yaml for training")
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL, help="Base model for training (e.g., yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--run_name", default="yolov8_local_run", help="Name for the training run")

    parser.add_argument("--trained_model", default=DEFAULT_TRAINED_MODEL, 
                        help="Path to your trained .pt model file for prediction")
    parser.add_argument("--image_input", default=DEFAULT_IMAGE_TO_PREDICT, help="Path to the image for prediction")
    parser.add_argument("--video_input", default=DEFAULT_VIDEO_TO_PREDICT, help="Path to the video for prediction")
    parser.add_argument("--output_dir_img", default=DEFAULT_OUTPUT_IMAGE_DIR, help="Directory to save annotated images")
    parser.add_argument("--output_dir_vid", default=DEFAULT_OUTPUT_VIDEO_DIR, help="Directory to save annotated videos")
    parser.add_argument("--display", action="store_true", help="Display video frames during processing (can be slow)")


    args = parser.parse_args()

    ensure_dir(args.output_dir_img)
    ensure_dir(args.output_dir_vid)
    
    if args.action == "train":
        train_model(
            data_yaml_path=args.data_yaml,
            base_model_name=args.base_model,
            epochs=args.epochs,
            img_size=args.imgsz,
            batch_size=args.batch,
            patience=args.patience,
            run_name=args.run_name
        )
    elif args.action == "predict_image":
        if not os.path.exists(args.trained_model):
            print(f"Error: Trained model not found at {args.trained_model}. Please train a model or provide a valid path.")
        else:
            predict_on_image(
                model_path=args.trained_model,
                image_path=args.image_input,
                output_dir=args.output_dir_img
            )
    elif args.action == "predict_video":
        if not os.path.exists(args.trained_model):
            print(f"Error: Trained model not found at {args.trained_model}. Please train a model or provide a valid path.")
        else:
            predict_on_video(
                model_path=args.trained_model,
                video_path=args.video_input,
                output_dir=args.output_dir_vid,
                display_video=args.display
            )
    else:
        print("Invalid action. Please choose 'train', 'predict_image', or 'predict_video'.")