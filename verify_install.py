import numpy
import cv2
import matplotlib
import sklearn
import insightface
import onnxruntime
import tqdm
import sys

def verify_install():
    print(f"Python version: {sys.version}")
    print(f"Numpy version: {numpy.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Matplotlib version: {matplotlib.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")
    print(f"Insightface version: {insightface.__version__}")
    print(f"Onnxruntime version: {onnxruntime.__version__}")
    print(f"Tqdm version: {tqdm.__version__}")
    
    # Check Numpy version explicitly
    if numpy.__version__.startswith("2"):
        print("WARNING: Numpy version is 2.x, which might be incompatible with some older libraries.")
    else:
        print("Numpy version is < 2.0, as requested.")

if __name__ == "__main__":
    try:
        verify_install()
        print("All libraries imported successfully.")
    except Exception as e:
        print(f"Verification failed: {e}")
        sys.exit(1)
