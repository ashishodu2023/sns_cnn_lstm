# Anomaly Detector Framework

This project implements a modular framework for an anomaly detector using a CNN+LSTM model. The framework is structured into separate modules for data parsing, data preparation, model building, evaluation (with plots and metrics exported to PDF), and workflow orchestration. The project allows you to run the training or testing workflow through a single main entry point using command-line arguments.

## Project Structure

```
sns_cnn_lstm
├── analysis
│    └── evaluation.py
├── data_preparation
│    ├── data_loader.py
│    ├── data_scaling.py
│    └── data_transformer.py
├── model
│    └── cnn_lstm_anomaly_model.py
├── parser
│    └── configs.py
├── testing
│    └── test.py
├── training
│    └── train.py
├── utils
│    └── logger.py
├── driver.py
├── submit.sh
└── requirements.txt
```


## Installation

   ```bash
   git clone SNS_Anomaly_Detection/cnn_lstm
   cd SNS_Anomaly_Detection/cnn_lstm
   pip install -r SNS_Anomaly_Detection/requirements.txt
   python ~/SNS_Anomaly_Detection/cnn_lstm/driver.py --train
   python ~/SNS_Anomaly_Detection/cnn_lstm/driver.py --test
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
