from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import time
from pathlib import Path

# use arguments from command line
def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with YOLOv8 model on images or videos')
    parser.add_argument('--weights', type=str, default='runs/detect/train/weights/best.pt',
                      help='path to model weights')
    parser.add_argument('--conf', type=float, default=0.25,
                      help='confidence threshold')
    parser.add_argument('--source', type=str, required=True,
                      help='path to input image/video file or webcam number (0, 1, etc.)')
    parser.add_argument('--output', type=str, default=None,
                      help='path to save output (defaults to "output_[source_name]")')
    parser.add_argument('--show', action='store_true',
                      help='display output window while processing')
    return parser.parse_args()

# create masks for regions of interest
def create_roi_masks(image_shape, trapezoid_points=None):
    """Create binary masks for the regions of interest (ROI)"""
    height, width = image_shape[:2]
    
    if trapezoid_points is None:
        # Define trapezoid points [top_left, top_right, bottom_right, bottom_left]
        trapezoid_points = np.array([
            [int(width * 0.25), int(height * 0.5)],
            [int(width * 0.75), int(height * 0.5)],
            [int(width * 0.99), height - 1],
            [int(width * 0.01), height - 1]
        ], dtype=np.int32)
    
    # full ROI mask
    full_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(full_mask, [trapezoid_points], 255)
    
    # mask for urgent (larger area URGENT)
    urgent_mask = np.zeros((height, width), dtype=np.uint8)
    
    mid_left = (
        int((trapezoid_points[0][0] * 0.77 + trapezoid_points[3][0] * 0.23)),
        int((trapezoid_points[0][1] * 0.77 + trapezoid_points[3][1] * 0.23))
    )

    mid_right = (
        int((trapezoid_points[1][0] * 0.77 + trapezoid_points[2][0] * 0.23)),
        int((trapezoid_points[1][1] * 0.77 + trapezoid_points[2][1] * 0.23))
    )
    
    # Bottom half trapezoid points
    bottom_trapezoid = np.array([
        mid_left,
        mid_right,
        trapezoid_points[2],
        trapezoid_points[3]
    ], dtype=np.int32)
    
    cv2.fillPoly(urgent_mask, [bottom_trapezoid], 255)
    # determine warning mask (full max - urgent)
    warning_mask = full_mask.copy()
    warning_mask[urgent_mask == 255] = 0
    
    return full_mask, warning_mask, urgent_mask, trapezoid_points, bottom_trapezoid

# determines the zone that the object is in given a box on the screen
def get_zone(box, warning_mask, urgent_mask):
    """
    returns 0, 1, or 2 for each detection:
    0 = outside ROI
    1 = warning zone ("RECOMMENDED")
    2 = urgent zone ("TAKEN")
    """
    x1, y1, x2, y2 = map(int, box)
    
    height, width = warning_mask.shape
    
    # Calculate the center of the bottom edge of the box (where object touches ground)
    bottom_center_x = min((x1 + x2) // 2, width - 1)  # Ensure x is within bounds
    bottom_center_y = min(y2, height - 1)  # Ensure y is within bounds
    
    # Check which zone the bottom center point is in
    if urgent_mask[bottom_center_y, bottom_center_x] > 0: # in urgent zone
        return 2
    elif warning_mask[bottom_center_y, bottom_center_x] > 0: # in warning zone
        return 1
    else: # neither, ignore
        return 0

# draw boundary box for objects
def draw_safety_box(image, box, class_name, conf, zone, frame_count=0):
    x1, y1, x2, y2 = map(int, box)
    
    UNSAFE_COLOR = (0, 0, 255)      # unsafe objects
    SAFE_COLOR = (0, 255, 0)        # safe objects
    IGNORED_COLOR = (128, 128, 128) # outside ROI
    WARNING_COLOR = (0, 165, 255)   # Orange if in warning zone
    URGENT_COLOR = (0, 0, 255)      # urgent zone
    
    # determine color and message based on class and zone
    if zone == 0:  # outside region
        color = IGNORED_COLOR
        safety_text = 'IGNORED - Outside ROI'
        is_unsafe = False
    else:
        # determine if object is safe to drive over
        is_safe = class_name == 'Pothole'
        
        if is_safe:
            color = SAFE_COLOR
            safety_text = 'SAFE - Can Drive Over'
            is_unsafe = False
        else: # warning
            is_unsafe = True
            if zone == 1:
                color = WARNING_COLOR if (frame_count // 10) % 2 == 0 else UNSAFE_COLOR
                safety_text = 'EVASIVE ACTION RECOMMENDED' # display text
            else: # urgent
                bright_red = (0, 0, 255)
                dark_red = (0, 0, 180)
                color = bright_red if (frame_count // 5) % 2 == 0 else dark_red
                safety_text = 'EVASIVE ACTION TAKEN' # display text
    
    # bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # label + confidence
    label = f'{class_name} ({conf:.2f})'
    
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    safety_text_size = cv2.getTextSize(safety_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    max_width = max(text_size[0], safety_text_size[0])
    
    cv2.rectangle(image, (x1, y1 - 45), (x1 + max_width + 10, y1), color, -1)
    
    cv2.putText(image, label, (x1 + 5, y1 - 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(image, safety_text, (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    if is_unsafe:
        if zone == 1:
            warning_text = "!!! EVASIVE ACTION RECOMMENDED !!!"
            warning_color = WARNING_COLOR if (frame_count // 10) % 2 == 0 else UNSAFE_COLOR
        else:
            warning_text = "!!! EVASIVE ACTION TAKEN !!!"
            bright_red = (0, 0, 255)
            dark_red = (0, 0, 180)
            warning_color = bright_red if (frame_count // 5) % 2 == 0 else dark_red
        
        text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = (image.shape[1] - text_size[0]) // 2 
        
        cv2.rectangle(image, (text_x - 10, 10), (text_x + text_size[0] + 10, 10 + text_size[1] + 20), warning_color, -1)
        
        cv2.putText(image, warning_text, (text_x, 10 + text_size[1] + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    return image

def draw_roi(image, trapezoid_points, bottom_trapezoid):
    """Draw the regions of interest on the image with different colors"""
    # Create a copy for overlay
    overlay = image.copy()
    
    # warning zone (yellow-orange)
    warning_points = np.array([
        trapezoid_points[0],
        trapezoid_points[1],
        bottom_trapezoid[1],
        bottom_trapezoid[0]
    ], dtype=np.int32)
    
    cv2.fillPoly(overlay, [warning_points], (0, 165, 255))
    
    # urgent zone (in red)
    cv2.fillPoly(overlay, [bottom_trapezoid], (0, 0, 255))
    
    alpha = 0.3  # transparency
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # draw ROI
    cv2.polylines(image, [trapezoid_points], True, (255, 255, 255), 2)
    
    cv2.line(image, 
             tuple(bottom_trapezoid[0]), 
             tuple(bottom_trapezoid[1]), 
             (255, 255, 255), 2)
    
    return image

def process_frame(frame, model, conf_threshold, masks, trapezoids, frame_count=0):
    """Process a single frame and return the annotated frame"""
    # unpack masks and trapezoids
    full_mask, warning_mask, urgent_mask, full_trapezoid, bottom_trapezoid = masks
    
    # create a visualization copy of the frame
    vis_frame = frame.copy()
    
    # draw ROI on visualization frame
    vis_frame = draw_roi(vis_frame, full_trapezoid, bottom_trapezoid)
    
    # run inference on the whole frame
    results = model(frame, conf=conf_threshold)[0]
    
    # process detections
    for result in results.boxes.data:
        x1, y1, x2, y2, conf, cls = result
        class_name = model.names[int(cls)]
        
        # check detection zone using function
        zone = get_zone((x1, y1, x2, y2), warning_mask, urgent_mask)
        
        # Draw detection
        vis_frame = draw_safety_box(vis_frame, (x1, y1, x2, y2), class_name, conf, zone, frame_count)
    
    return vis_frame

def process_video(source, model, conf_threshold, output_path=None, show=False):
    """logic to process a video if given a video"""
    # video capture
    cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video source: {source}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    masks_and_trapezoids = create_roi_masks((height, width))
    
    # initialize video writer
    writer = None
    if output_path:

        slow_fps = fps / 2 
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            slow_fps,  # slower fps
            (width, height)
        )


    
    # frame counter (used only for flashing effect)
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Frame reading stopped at frame {frame_count}")
                break
            
            processed_frame = process_frame(frame, model, conf_threshold, masks_and_trapezoids, frame_count)
            
            # increment frame counter
            frame_count += 1
            
            # loging
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")
            
            if writer:
                writer.write(processed_frame)
            
            # display
            if show:
                cv2.imshow('Detection', processed_frame)
                delay = 50
                if cv2.waitKey(delay) & 0xFF == ord('q'):
                    break
    
    finally:
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

def process_image(image_path, model, conf_threshold, output_path=None, show=False):
    """logic to process an image if given an image"""

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # create ROI masks based on image dimensions
    masks_and_trapezoids = create_roi_masks(image.shape)
    
    # process image
    processed_image = process_frame(image, model, conf_threshold, masks_and_trapezoids, frame_count=0)
    
    # save
    if output_path:
        cv2.imwrite(str(output_path), processed_image)
    
    # display
    if show:
        cv2.imshow('Detection', processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    args = parse_args()
    
    # load model, source, output, from arguments
    model = YOLO(args.weights)
    source = args.source

    if args.output:
        output_path = Path(args.output)
    else:
        source_path = Path(args.source)
        if source_path.is_file():
            output_path = source_path.parent / f"output_{source_path.name}"
        else:
            output_path = Path(f"output_{int(time.time())}.mp4")
    
    try:
        # check if given webcam instead of video
        if source.isdigit():
            source = int(source)
            print(f"Processing webcam {source}")
            process_video(source, model, args.conf, output_path, args.show)
        
        # if given source file
        elif Path(source).is_file():
            # check for image or video
            if Path(source).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                print(f"Processing image: {source}")
                process_image(source, model, args.conf, output_path, args.show)
            else:
                print(f"Processing video: {source}")
                process_video(source, model, args.conf, output_path, args.show)
        else:
            raise ValueError(f"Invalid source: {source}")
        
        print(f"Processing completed! Output saved to: {output_path}")
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == '__main__':
    main()