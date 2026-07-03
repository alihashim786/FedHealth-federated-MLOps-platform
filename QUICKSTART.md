# FedAIR Quick Start Guide

Get up and running with FedAIR in 5 minutes!

## Prerequisites Check

```bash
python --version  # Should be 3.9+
pip --version
```

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Verify Data Location

Check that your data is at the path specified in `utils/config.py`:
```python
DATA_ROOT = Path(r"C:\Users\zayan\Downloads\kvgwcrbjm3-1\kvgwcrbjm3-1\pollutant\pollutant")
```

You should see 9 CSV files (one per station):
- CCM.csv
- MGQ.csv
- OTZX.csv
- PK.csv
- RJL.csv
- SXL.csv
- XLDXC.csv
- XWH.csv
- ZHM.csv

## Step 3: Test Data Loading

```bash
python -c "from data_ingestion.node_simulator import NodeSimulator; ns = NodeSimulator(); print(f'Loaded {len(ns.get_node_info())} stations')"
```

Expected output: `Loaded 9 stations`

## Step 4: Run Training (Quick Test)

For a quick test with fewer rounds:

1. Edit `utils/config.py`:
```python
FEDERATED_CONFIG = {
    "num_rounds": 10,  # Reduced for quick test
    ...
}
```

2. Run training:
```bash
python training/train_federated.py
```

This will:
- Load and preprocess all station data
- Train federated models for 10 rounds
- Save models to `models/` directory
- Generate training curves in `outputs/`

## Step 5: Start Monitoring API

In a new terminal:
```bash
python -m uvicorn monitoring.monitor_api:app --host 0.0.0.0 --port 8000
```

Test it:
```bash
curl http://localhost:8000/health
```

## Step 6: Launch Dashboards

### Health Authority Dashboard
```bash
streamlit run dashboard/health_dashboard.py
```
Open: http://localhost:8501

### Citizen Dashboard
```bash
streamlit run dashboard/citizen_dashboard.py
```
Open: http://localhost:8501 (or different port if first is running)

## Common Issues

### Issue: "No module named 'utils'"
**Solution**: Make sure you're running from the project root directory.

### Issue: "File not found" for CSV files
**Solution**: Update `DATA_ROOT` in `utils/config.py` to your actual data path.

### Issue: CUDA out of memory
**Solution**: Reduce batch size in `FEDERATED_CONFIG["batch_size"]` or use CPU by setting `TRAINING_CONFIG["device"] = "cpu"`.

### Issue: Import errors
**Solution**: Install all dependencies: `pip install -r requirements.txt`

## Next Steps

1. **Full Training**: Set `num_rounds` back to 100 in config
2. **MLflow Tracking**: Start MLflow UI: `mlflow ui --port 5000`
3. **Production Deployment**: See `DEPLOYMENT.md`
4. **Customization**: Modify `utils/config.py` for your needs

## Verification Checklist

- [ ] All 9 stations loaded successfully
- [ ] Training completes without errors
- [ ] Models saved in `models/` directory
- [ ] Monitoring API responds to health check
- [ ] Dashboards load and display data

## Getting Help

- Check `README.md` for detailed documentation
- Review inline code comments
- Check logs in `logs/` directory
- See `DEPLOYMENT.md` for deployment options

Happy forecasting! 🌬️

