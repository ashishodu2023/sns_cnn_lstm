# Anomaly Detector Framework

This project implements a modular framework for an anomaly detector using a CNN+LSTM model. The framework is structured into separate modules for data parsing, data preparation, model building, evaluation (with plots and metrics exported to PDF), and workflow orchestration. The project allows you to run the training or testing workflow through a single main entry point using command-line arguments.

## Project Structure

sns_cnn_lstm/ ├── analysis/ │ └── evaluation.py ├── data_preparation/ │ └── data_prep.py ├── model/ │ └── anomaly_model.py ├── parser/ │ └── paper_parser.py ├── train_flow.py ├── test_flow.py ├── main.py └── requirements.txt


## Installation

   ```bash
   git clone sns_cnn_lstm
   cd sns_cnn_lstm
   pip install -r requirements.txt
   python main.py --train
   python main.py --test
   ```

## Data Parser Modules

- The BPM data configuration is now implemented in `data_parser/bpm_parser.py` (class `BPMDataConfig`).
- The DCM data configuration is now implemented in `data_parser/dcm_config.py` (class `DCMDatConfig`).


## Logs and Reports 

- **Logs are stored in the `logs/` directory:**
  - Overall logs: `logs/app.log`
  - Training logs: `logs/train_flow.log`
  - Testing logs: `logs/test_flow.log`

- **Reports are exported as PDF:**
  - Training report: `train_metrics_report.pdf`
  - Testing report: `test_metrics_report.pdf`

## Customization

- Update `parser/paper_parser.py` with your actual data parsing logic.
- Modify `data_preparation/data_prep.py` for data cleaning, merging, and feature engineering.
- Customize the CNN+LSTM architecture in `model/anomaly_model.py` as needed.
- Adjust evaluation metrics and plots in `analysis/evaluation.py`.