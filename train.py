from ultralytics import YOLO
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv8 model on custom dataset')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for training')
    parser.add_argument('--img-size', type=int, default=416, help='input image size')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                      help='YOLOv8 model size: n(nano), s(small), m(medium), l(large), x(xlarge)')
    parser.add_argument('--device', type=str, default='0' if torch.cuda.is_available() else 'cpu',
                      help='device to train on (cuda device number or "cpu")')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize model
    model = YOLO(f'yolov8{args.model_size}.pt')
    
    # Training arguments
    training_args = {
        'data': 'data.yaml',          # Path to data configuration file
        'epochs': args.epochs,         # Number of epochs to train for
        'batch': args.batch_size,      # Batch size
        'imgsz': args.img_size,        # Input image size
        'device': args.device,         # Training device
        'workers': 4,                  # Number of worker threads (reduced to prevent bottleneck)
        'patience': 25,                # Early stopping patience (reduced)
        'save': True,                  # Save training checkpoints
        'cache': True,                 # Cache images for faster training
        'pretrained': True,            # Use pretrained weights
        'optimizer': 'auto',           # Optimizer (SGD, Adam, etc.)
        'verbose': True,               # Print verbose output
        'seed': 42,                    # Random seed for reproducibility
        'exist_ok': True,              # Overwrite existing experiment
        'resume': False,               # Resume training from last checkpoint
        'mosaic': 0.0,                # Disable mosaic augmentation for speed
        'rect': True,                 # Rectangular training for better batch loading
        'amp': True,                  # Enable automatic mixed precision
    }
    
    # Train the model
    results = model.train(**training_args)
    
    # Evaluate the model on validation set
    results = model.val()
    
    print("Training completed! Model weights saved in 'runs/detect/train' directory")

if __name__ == '__main__':
    main() 