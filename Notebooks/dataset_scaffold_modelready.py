# Python
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
.ipynb_checkpoints/

# Model bundles / weights (distribute via GitHub Releases)
*.pt
*.pth
*.ckpt

# Outputs
output*/
explain_out/
predictions.csv
*.log

# Large data (deposit on Zenodo; keep out of git)
data/*.csv
data/*.parquet
fold_graph_cache/

# OS / editor
.DS_Store
.vscode/
.idea/
