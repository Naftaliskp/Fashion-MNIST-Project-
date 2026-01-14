# Fashion-MNIST Image Classification Project

## Project Overview
Convolutional Neural Network implementation with batch normalization for classifying 10 different clothing categories from the Fashion-MNIST dataset. The model achieves 88.74% accuracy on the validation set using a simple yet efficient CNN architecture.

## Key Results
- **Best Model Accuracy**: 88.74% (after 5 epochs)
- **Dataset**: 70,000 images across 10 clothing categories
- **Training Time**: 127.71 seconds
- **Model Parameters**: 18,494 trainable parameters
- **Batch Size**: 100
- **Learning Rate**: 0.1

## Model Architecture
```python
class CNN_batch(nn.Module):
    def __init__(self, out_1=16, out_2=32, number_of_classes=10):
        super(CNN_batch, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, 
                             kernel_size=5, padding=2)
        self.conv1_bn = nn.BatchNorm2d(out_1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, 
                             kernel_size=5, padding=2)
        self.conv2_bn = nn.BatchNorm2d(out_2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(out_2 * 4 * 4, number_of_classes)
        self.bn_fc1 = nn.BatchNorm1d(number_of_classes)
```

## Dataset Classes
The Fashion-MNIST dataset contains 10 different clothing categories:

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | T-shirt/top | Short sleeve t-shirts |
| 1 | Trouser | Pants and trousers |
| 2 | Pullover | Sweaters and pullovers |
| 3 | Dress | Dresses and gowns |
| 4 | Coat | Outerwear coats |
| 5 | Sandal | Open-toe sandals |
| 6 | Shirt | Long sleeve shirts |
| 7 | Sneaker | Athletic shoes |
| 8 | Bag | Handbags and purses |
| 9 | Ankle boot | Boots and ankle-height shoes |

## Training Progress
| Epoch | Training Loss | Validation Accuracy |
|-------|---------------|---------------------|
| 1 | 297.99 | 84.91% |
| 2 | 208.92 | 87.32% |
| 3 | 187.65 | 88.17% |
| 4 | 175.37 | 89.42% |
| 5 | 167.15 | 88.74% |

## Technologies
- Python 3.9+
- PyTorch 2.8.0
- torchvision 0.23.0
- NumPy 1.24.3
- Matplotlib 3.7.1
- scikit-learn 1.3.0
- seaborn 0.12.2
- Jupyter Notebook

## Project Structure
```
Fashion-MNIST-Classification/
├── fashion_mnist_classification.ipynb      # Main notebook
├── requirements.txt                        # Dependencies
├── README.md                               # This file
├── model_weights.pth                       # Trained model weights
├── data/
│   └── FashionMNIST/                       # Dataset directory
└── images/                                # Visualization outputs
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── sample_predictions.png
    └── sample_images.png
```

## Quick Start
Clone repository:
```bash
git clone https://github.com/Naftaliskp/Fashion-MNIST-Classification.git
cd Fashion-MNIST-Classification
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run Jupyter notebook:
```bash
jupyter notebook fashion_mnist_classification.ipynb
```

## Complete Pipeline
1. Data loading and preprocessing
2. CNN with batch normalization implementation
3. Model training and validation
4. Performance evaluation and visualization
5. Confusion matrix analysis
6. Sample prediction visualization

## Visualizations
![Training Curves](images/training_curves.png)
*Training cost and validation accuracy over 5 epochs*

![Confusion Matrix](images/confusion_matrix.png)
*Confusion matrix showing per-class performance*

![Sample Predictions](images/sample_predictions.png)
*Sample predictions with true vs predicted labels*

![Sample Images](images/sample_images.png)
*Example images from Fashion-MNIST dataset*

## Dependencies
```
torch==2.8.0
torchvision==0.23.0
numpy==1.24.3
matplotlib==3.7.1
scikit-learn==1.3.0
seaborn==0.12.2
pandas==2.0.3
jupyter==1.0.0
```

## Key Findings
1. Batch normalization significantly improves training stability
2. Model achieves 88.74% accuracy with just 5 training epochs
3. Highest accuracy on distinct categories: Trouser (98%), Bag (99%), Ankle boot (97%)
4. Most confusion occurs between visually similar classes: Shirt vs T-shirt/top
5. Simple CNN architecture with only 18K parameters yields strong results
6. Training converges quickly with learning rate of 0.1 and SGD optimizer

## Future Enhancements
- Implement data augmentation techniques (rotation, flipping, cropping)
- Experiment with deeper CNN architectures (ResNet, VGG)
- Add dropout layers for better regularization
- Implement learning rate scheduling
- Create web interface for real-time predictions
- Export model to ONNX format for deployment
- Add model interpretability visualizations

---

Convolutional Neural Network for Fashion-MNIST image classification using PyTorch with batch normalization
