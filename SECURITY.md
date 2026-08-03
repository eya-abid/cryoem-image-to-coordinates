# Security

Do not commit access tokens, credentials, controlled particle data, private checkpoints, or identifiable workstation paths. Use environment variables and the placeholders documented in `.env.example`.

PyTorch checkpoints use Python serialization. Load only checkpoints produced by a trusted source. A malicious checkpoint may execute code during deserialization.
