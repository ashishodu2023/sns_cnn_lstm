# Anomaly Detector Framework

Lorem Ipsum

## Project Structure

```
sns_cnn_lstm
├── data_preparation
│    ├── data_loader.py
│    ├── data_scaling.py
│    └── data_transformer.py
├── factories
│    └── sns_raw_prep_sep_dnn_factory.py
├── model
│    └── vae_bilstm.py
├── parser
│    └── configs.py
├── utils
│    └── logger.py
├── visualization
│    └── plots.py
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
