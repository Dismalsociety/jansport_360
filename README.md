# Version 13 Characteristics
- Parameters: ~50,000
- Layers: Transformer + BiLSTM (overkill)
- Regularization: Minimal
- Data splitting: Has leakage
- Result: Training loss ↓ Validation loss ↑ (overfitting!)

# GRU Model Characteristics  
- Parameters: ~8,000
- Layers: Single GRU (simple)
- Regularization: Multiple techniques
- Data splitting: Proper (no leakage)
- Expected: Training and validation loss stay close
