# GMAD  
Grounded Multilingual Audio Description  
Final Project for COSC 89.30 Topics in Video Understanding (Fall 2025) @ Dartmouth College  

## Abstract  
Current multimodal captioning systems such as GPT4V QwenVL and LLaVA remain constrained by a monolingual core and inherited biases from English pretrained vision encoders like CLIP. They hallucinate freely and flatten rich visual structure into short ungrounded sentences. These systems cannot provide trustworthy culturally adaptive narration for visually impaired users. GMAD challenges this paradigm. Instead of forcing one giant model to both see and speak GMAD decomposes the problem. Every visual signal is grounded through independent detectors trackers and attribute models then compiled into a language neutral Temporal Scene Graph. Only after the world is symbolically understood do we invite powerful text only LLMs to narrate directly in the target language eliminating translation artifacts and bias. GMAD shows that accurate multilingual grounded and culturally aware description does not require multimodal end to end learning but a clean separation of perception and narration with a universal graph at the center.

---

## Overview  
GMAD is a modular pipeline for generating grounded multilingual audio descriptions from raw video. Visual perception is executed entirely through specialized vision models detection tracking and attribute extraction. Narration is produced by text only LLMs ensuring low hallucination and full language flexibility.

Insert pipeline figure here.  
**(pipeline.png)**  
![Pipeline](GMAD-Image.png)

---

## Pipeline  
GMAD proceeds through a transparent sequence of stages.

### Scene Segmentation  
Splits the video into scenes and extracts representative frames.

### Object Detection  
Uses Florence2 YOLO or RTDETR to identify objects in each frame.

### Tracking  
Associates detections across time using a sparse ByteTrackLite tracker.

### Entity Construction  
Chooses a canonical frame per object and crops entities for analysis.

### CLIP Attribute Enrichment  
Extracts colors materials textures and coarse attributes from each crop.

### Temporal Scene Graph  
Builds a symbolic representation of all objects their attributes their relations and all temporal intervals.

### Qwen Salient Event Extraction  
Uses QwenVL to detect high level actions and scene events.

### LLM Fusion and Grounding  
A text only LLM merges the TSG with Qwen cues generating grounded narration directly in the target language without translation.

### Speech and Audio Layering  
Narration audio is synthesized and aligned with the original video.

---

## Directory Structure  
GMAD/
scene_segmentation/
detection_tracking/
clip_enrichment/
tsg_graph_construction/
qwen/
fusion_layer/
Audio_Layering/
tts/
prompts.yaml
config.yml