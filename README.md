LLM Fine-Tuning Using HuggingFace's PEFT

---

#### Project Overview

This project provides a comprehensive solution for fine-tuning large language models (LLMs) using HuggingFace's Parameter-Efficient Fine-Tuning (PEFT) framework. It includes a user-friendly graphical user interface (GUI) for easy interaction with the model training and inference processes. The project is structured into three main modules:

1. **GUI (`gui.py`)**: A Kivy-based interface that allows users to initiate and monitor training and inference tasks.
2. **Inference Module (`inference_module.py`)**: Handles the loading and running of PEFT-based models for inference.
3. **Training Module (`training_module.py`)**: Facilitates the training of language models using the PEFT framework with customisable configuration.

#### Features

- **Easy-to-use GUI**: Simplify the process of training and inference with a graphical interface.
- **PEFT Implementation**: Efficiently fine-tune large language models with reduced computational resources.
- **Customizable Configurations**: Adapt the training and inference settings through configuration files.

#### Requirements

- Python 3.9 minimum
- PyTorch
- Transformers (Hugging Face)
- Kivy (for GUI)
- Other dependencies listed in `requirements.txt` 

#### Setup and Installation

1. **Clone the Repository**:

2. **Install Dependencies**:

```
pip install -r requirements.txt
```

3. **Configuration**:
- Adjust the training and inference parameters in the `config.yaml` file (if used in the project).

#### Usage

1. **Running the GUI**:

```
python gui.py
```

- The GUI allows you to choose between training and inference.
- Follow on-screen instructions to load data, set parameters, and start processes.

2. **Training a Model**:
- Use the GUI to navigate to the training section.
- Set desired parameters and start the training process.

3. **Running Inference**:
- Navigate to the inference section in the GUI.
- Load the trained model and input data for inference.

#### Customization

- Modify `training_module.py` and `inference_module.py` for advanced model configurations and custom training/inference logic.
- Update the GUI in `gui.py` for additional features or changes in the user interface.

---


