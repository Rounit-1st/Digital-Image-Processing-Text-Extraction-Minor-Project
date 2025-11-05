# Document Scanner

A Python-based document scanner that detects document edges and applies perspective correction to get a bird's eye view.

## Installation
```bash
pip install opencv-python numpy
```

## Usage

1. Place your document image in the same directory as the script and name it `sample.jpg`
2. Run the script:
```bash
python script.py
```

3. The application will display:
   - Detected lines on the document
   - Corner points
   - Perspective-corrected output

Press any key to close the windows.

## Documentation

For detailed information about the project's inner workings, please refer to:
```
DIP Project hard.pdf
```

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy