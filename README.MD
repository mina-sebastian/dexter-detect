# **Multi-Task Image Processing with YOLO and AlexNet**

## **Overview**
This project runs two tasks concurrently, utilizing both **YOLO** and **AlexNet** models for image processing. The results are organized into separate folders for clarity and ease of access.
The target is to do face detection on cartoons.

## **Getting Started**

### **Running the Script**
- Execute the main script: **`detector.py`**.
- The script processes two tasks simultaneously, alongside the YOLO model.
- Intermediate and final results are stored in structured directories.

### **Folder Structure**
Upon execution, the following directories are created:

- **Temporary Output (During Execution):**
  - `task1_temp` – Intermediate results for Task 1
  - `task2_temp` – Intermediate results for Task 2
  - `task1_yolo_temp` – YOLO intermediate results for Task 1
  - `task2_yolo_temp` – YOLO intermediate results for Task 2

*These temporary folders allow real-time tracking of the program's progress.*

- **Final Output (After Execution Completes):**
  - Located in: `data/output`
  - Directories created:
    - `task1` – Final results for Task 1
    - `task2` – Final results for Task 2
    - `task1_yolo` – Final YOLO results for Task 1
    - `task2_yolo` – Final YOLO results for Task 2

⚠️ *In case the final directories aren’t created at the end, the complete results can still be found in the temporary folders.*

### **Input Data Path**
The program reads images from the following path: `/validare/validare`


This path is defined at **line 26** in `detector.py`:

```python
test_examples_path = os.path.join(os.getcwd(), 'validare', 'validare')
