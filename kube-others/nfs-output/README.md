# NFS Output Job

Job configuration with NFS-backed results storage.

## Setup (once)

```bash
kubectl apply -f nfs-pv-pvc.yaml
```

## Create a run

```bash
# Set unique suffix and deploy
SUFFIX="-$(date +%s)" && \
  sed -i.bak "/^nameSuffix:/d" kustomization.yaml && \
  echo "nameSuffix: $SUFFIX" >> kustomization.yaml && \
  kubectl apply -k .
```

This creates:
- `evo-burgers-config-<suffix>` (ConfigMap)
- `evo-burgers-job-<suffix>` (Job)

## List/Delete

```bash
# List all runs
kubectl get jobs,configmaps -l app=evo-burgers

# Delete a specific run by name
kubectl delete job evo-burgers-job-1702400000
kubectl delete configmap evo-burgers-config-1702400000

# Delete ALL runs (careful!)
kubectl delete jobs,configmaps -l app=evo-burgers
```

## Output

Results are written to NFS `10.3.10.253:/bai/<pod-name>/`
