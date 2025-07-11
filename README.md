# CNN Model Evaluation Utilities

This project contains utility functions for evaluating the performance of convolutional neural network (CNN) models, with a primary focus on calculating various metrics and generating confusion matrices to assess the model's predictions against true labels.

## Files

- **eval_utils.py**: Contains functions for evaluating model performance, including metrics calculation and confusion matrix generation.
- **heatmap.py**: Contains function that takes in confusion_matrix and prints out a heatmap
- **model.py**: Contains logic for importing models, creating new models, training them etc.
- **main.py** Initialises variables, at the bottom lines can be uncommented to run various functions from other files.

## Usage

1. **Installation**: Ensure you have the required libraries installed. You can install them using pip, e.g.:
   ```
   pip install torch pandas
   ```

2. **Running Evaluations**: To evaluate your model, import the `eval_metrics` function from `eval_utils.py` and pass your model, dataloader, and other necessary parameters.

3. **Metrics**: The evaluation functions will return various metrics, including accuracy, precision, recall, and F1 score, along with a confusion matrix.