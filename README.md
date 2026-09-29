# 🔍 SnapInspect AI

## NPU-Accelerated Intelligent PCB Defect Inspection Using On-Device Computer Vision

SnapInspect AI is an AI-powered automated Printed Circuit Board (PCB) inspection system that uses computer vision and deep learning to detect and localize PCB manufacturing defects.

The project combines **PCB inspection, computer vision, YOLO-based object detection, Python, and Snapdragon edge AI** to create a practical quality-inspection solution. The system captures or receives a PCB image, processes the image, detects defects, displays bounding boxes and confidence scores, and generates an automatic PASS/FAIL inspection result.

The long-term goal is to optimize the trained AI model for **Snapdragon NPU acceleration using Qualcomm AI Hub**, enabling efficient on-device inference without depending entirely on cloud processing.

---

## 🎯 Problem Statement

Printed Circuit Boards contain many components, tracks, solder joints, vias, and fine electrical connections. Manufacturing and assembly defects can affect the reliability and performance of electronic products.

Common PCB defects include:

- Open circuits
- Short circuits
- Mousebites
- Spurs
- Pin holes
- Spurious copper
- Missing or incorrectly placed components
- Soldering defects

Manual PCB inspection can be time-consuming and may produce inconsistent results. Conventional automated optical inspection systems can also require specialized equipment.

SnapInspect AI explores an accessible AI-based approach for automated PCB defect inspection using computer vision and edge AI.

---

## 💡 Proposed Solution

SnapInspect AI uses a camera or uploaded PCB image as input.

The image passes through the following pipeline:

```text
PCB Image / Camera
        ↓
Image Preprocessing
        ↓
YOLO Object Detection Model
        ↓
Snapdragon Edge AI / NPU
        ↓
Defect Classification
        ↓
Bounding Box + Confidence
        ↓
PASS / FAIL Decision
        ↓
Inspection Report
