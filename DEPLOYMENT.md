# FedAIR Deployment Guide

This guide covers deployment options for the FedAIR system, from local development to cloud production.

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Monitoring Setup](#monitoring-setup)

## Local Development

### Prerequisites

- Python 3.9+
- pip
- (Optional) CUDA for GPU acceleration

### Setup

1. **Clone and install**
```bash
git clone <repository-url>
cd MlOps_Project
pip install -r requirements.txt
```

2. **Configure data path**
Edit `utils/config.py` to set your data path:
```python
DATA_ROOT = Path(r"path/to/your/data")
```

3. **Start MLflow tracking server**
```bash
mlflow ui --port 5000
```

4. **Run training**
```bash
python training/train_federated.py
```

5. **Start monitoring API**
```bash
python -m uvicorn monitoring.monitor_api:app --host 0.0.0.0 --port 8000
```

6. **Launch dashboards**
```bash
# Health Authority Dashboard
streamlit run dashboard/health_dashboard.py

# Citizen Dashboard (in another terminal)
streamlit run dashboard/citizen_dashboard.py
```

## Docker Deployment

### Build Image

```bash
docker build -t fedair-server:latest -f mlops_pipeline/docker/Dockerfile .
```

### Run Container

```bash
docker run -d \
  --name fedair-server \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/data:/app/data \
  fedair-server:latest
```

### Docker Compose (Recommended)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  mlflow:
    image: python:3.9-slim
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
    command: pip install mlflow && mlflow ui --host 0.0.0.0 --port 5000

  fedair-server:
    build:
      context: .
      dockerfile: mlops_pipeline/docker/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./outputs:/app/outputs
      - ./data:/app/data
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow

  dashboard:
    build:
      context: .
      dockerfile: mlops_pipeline/docker/Dockerfile.dashboard
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    depends_on:
      - fedair-server
```

Run:
```bash
docker-compose up -d
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (v1.20+)
- kubectl configured
- PersistentVolume support

### Deploy

1. **Create namespace and resources**
```bash
kubectl apply -f mlops_pipeline/k8s/deployment.yaml
```

2. **Verify deployment**
```bash
kubectl get pods -n fedair
kubectl get services -n fedair
```

3. **Check logs**
```bash
kubectl logs -f deployment/fedair-server -n fedair
```

4. **Access services**
```bash
# Get service URLs
kubectl get svc -n fedair

# Port forward for local access
kubectl port-forward svc/fedair-server-service 8000:8000 -n fedair
```

### Scaling

```bash
# Scale server replicas
kubectl scale deployment fedair-server --replicas=5 -n fedair

# Scale dashboard
kubectl scale deployment fedair-dashboard --replicas=3 -n fedair
```

### Update Deployment

```bash
# Update image
kubectl set image deployment/fedair-server fedair-server=fedair-server:v2.0 -n fedair

# Rolling update
kubectl rollout status deployment/fedair-server -n fedair
```

## Cloud Deployment

### AWS EKS

1. **Create EKS cluster**
```bash
eksctl create cluster --name fedair-cluster --region us-east-1
```

2. **Deploy to EKS**
```bash
kubectl apply -f mlops_pipeline/k8s/deployment.yaml
```

3. **Configure LoadBalancer**
Update service type to LoadBalancer in deployment.yaml

### Google Cloud GKE

1. **Create GKE cluster**
```bash
gcloud container clusters create fedair-cluster --zone us-central1-a
```

2. **Deploy**
```bash
gcloud container clusters get-credentials fedair-cluster --zone us-central1-a
kubectl apply -f mlops_pipeline/k8s/deployment.yaml
```

### Azure AKS

1. **Create AKS cluster**
```bash
az aks create --resource-group fedair-rg --name fedair-cluster
```

2. **Deploy**
```bash
az aks get-credentials --resource-group fedair-rg --name fedair-cluster
kubectl apply -f mlops_pipeline/k8s/deployment.yaml
```

## Monitoring Setup

### Prometheus

1. **Install Prometheus**
```bash
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/bundle.yaml
```

2. **Configure ServiceMonitor**
Create `monitoring/service-monitor.yaml`:
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: fedair-metrics
  namespace: fedair
spec:
  selector:
    matchLabels:
      app: fedair-server
  endpoints:
  - port: http
    path: /prometheus/metrics
```

3. **Apply**
```bash
kubectl apply -f monitoring/service-monitor.yaml
```

### Grafana

1. **Install Grafana**
```bash
helm repo add grafana https://grafana.github.io/helm-charts
helm install grafana grafana/grafana -n fedair
```

2. **Access Grafana**
```bash
kubectl get secret grafana -n fedair -o jsonpath="{.data.admin-password}" | base64 -d
kubectl port-forward svc/grafana 3000:80 -n fedair
```

3. **Import dashboards**
- Use Prometheus as data source
- Import pre-built dashboards from `monitoring/grafana/`

## Environment Variables

Configure via ConfigMap:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fedair-config
  namespace: fedair
data:
  MLFLOW_TRACKING_URI: "http://mlflow-service:5000"
  DATA_ROOT: "/app/data"
  MODEL_DIR: "/app/models"
```

Apply:
```bash
kubectl apply -f monitoring/configmap.yaml
```

## Troubleshooting

### Pods not starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n fedair

# Check logs
kubectl logs <pod-name> -n fedair
```

### Service not accessible

```bash
# Check service endpoints
kubectl get endpoints fedair-server-service -n fedair

# Test connectivity
kubectl run -it --rm debug --image=busybox --restart=Never -- wget -O- http://fedair-server-service:8000/health
```

### Storage issues

```bash
# Check PVCs
kubectl get pvc -n fedair

# Check PVs
kubectl get pv
```

## Security Considerations

1. **Use secrets for sensitive data**
```bash
kubectl create secret generic fedair-secrets \
  --from-literal=api-key=your-key \
  -n fedair
```

2. **Enable TLS/SSL**
- Use ingress with TLS certificates
- Configure HTTPS in services

3. **Network policies**
- Restrict pod-to-pod communication
- Use service mesh (Istio/Linkerd) for advanced security

## Backup and Recovery

### Backup models

```bash
# Backup to S3
kubectl exec -it <pod-name> -n fedair -- \
  aws s3 sync /app/models s3://fedair-backups/models/
```

### Restore

```bash
# Restore from backup
kubectl exec -it <pod-name> -n fedair -- \
  aws s3 sync s3://fedair-backups/models/ /app/models/
```

## Performance Tuning

1. **Resource limits**: Adjust CPU/memory in deployment.yaml
2. **Replicas**: Scale based on load
3. **Caching**: Enable model caching in monitoring API
4. **Database**: Use external database for MLflow tracking

## Next Steps

- Set up CI/CD pipeline (see `.github/workflows/mlops_ci_cd.yaml`)
- Configure alerting (Prometheus Alertmanager)
- Set up log aggregation (ELK stack)
- Implement auto-scaling (HPA)

