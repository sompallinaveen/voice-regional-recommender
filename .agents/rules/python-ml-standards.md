---
trigger: always_on
---

- Make sure all Python code adheres to PEP 8.
- Use explicit type hints for all function arguments and return types.
- When generating Hugging Face model code, always use device logic to map to 'cuda' or 'mps' if available, otherwise fallback to 'cpu'.
- Never write monolithic files. Always split logic into separate files (e.g. models, database, api routes).