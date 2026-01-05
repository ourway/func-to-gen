# Ollama Offline Setup for macOS (User-Level)

This guide explains how to load the pre-downloaded embedding models into Ollama on macOS without root access and without network connectivity.

## Included Models

- `nomic-embed-text` - Text embedding model (274 MB)

## Prerequisites

1. Install Ollama for macOS from https://ollama.ai (download on a machine with internet access)
2. Clone this repository with Git LFS support

## Setup Steps

### 1. Clone the Repository with LFS

```bash
# Install Git LFS if not already installed
brew install git-lfs
git lfs install

# Clone the repo (LFS files will download automatically)
git clone https://github.com/ourway/func-to-gen.git
cd func-to-gen

# If LFS files weren't pulled automatically
git lfs pull
```

### 2. Copy Models to Ollama Directory

On macOS, Ollama stores models in `~/.ollama/models/`. Copy the pre-downloaded models:

```bash
# Create the Ollama models directory if it doesn't exist
mkdir -p ~/.ollama/models/blobs
mkdir -p ~/.ollama/models/manifests/registry.ollama.ai/library

# Copy the model blobs
cp -r .ollama/models/blobs/* ~/.ollama/models/blobs/

# Copy the model manifests
cp -r .ollama/models/manifests/registry.ollama.ai/library/* ~/.ollama/models/manifests/registry.ollama.ai/library/
```

### 3. Start Ollama (User-Level)

Run Ollama without requiring root:

```bash
# Start the Ollama server in the background
ollama serve &

# Or run it in foreground to see logs
ollama serve
```

By default, Ollama runs on `http://localhost:11434`.

### 4. Verify the Models

```bash
# List available models
ollama list

# You should see:
# NAME                    ID              SIZE      MODIFIED
# nomic-embed-text:latest ...             274 MB    ...
```

### 5. Test the Embedding Model

```bash
# Test embedding generation
curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "Hello, world!"
}'
```

## Using with Python

```python
import requests

def get_embedding(text: str) -> list[float]:
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text}
    )
    return response.json()["embedding"]

# Example usage
embedding = get_embedding("Sample text to embed")
print(f"Embedding dimension: {len(embedding)}")  # 768 dimensions
```

## Environment Variables (Optional)

You can customize Ollama's behavior with environment variables:

```bash
# Custom models directory (if you don't want to use ~/.ollama)
export OLLAMA_MODELS=/path/to/custom/models

# Custom host/port
export OLLAMA_HOST=127.0.0.1:11434

# Then start Ollama
ollama serve
```

## Troubleshooting

### "Model not found" error

Ensure the directory structure matches exactly:
```
~/.ollama/models/
├── blobs/
│   ├── sha256-13014eef637dda50e7a73e6a29e182ec87a39a690be4c8d673b522d04434e144
│   ├── sha256-31df23ea7daa448f9ccdbbcecce6c14689c8552222b80defd3830707c0139d4f
│   ├── sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6
│   ├── sha256-c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4
│   └── sha256-ce4a164fc04605703b485251fe9f1a181688ba0eb6badb80cc6335c0de17ca0d
└── manifests/
    └── registry.ollama.ai/
        └── library/
            └── nomic-embed-text/
                └── latest
```

### Port already in use

```bash
# Find and kill existing Ollama process
pkill ollama

# Or use a different port
OLLAMA_HOST=127.0.0.1:11435 ollama serve
```

### Permission issues

All files should be readable by your user:
```bash
chmod -R u+rw ~/.ollama/models/
```

## Adding More Models for Offline Use

To add more models for offline use:

1. On a machine with internet access:
   ```bash
   ollama pull <model-name>
   ```

2. Copy the new blobs and manifests from `~/.ollama/models/` to this repo's `.ollama/models/`

3. Commit with Git LFS:
   ```bash
   git add .ollama/
   git commit -m "Add <model-name> for offline use"
   git push
   ```
