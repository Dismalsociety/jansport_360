1. Made Model MUCH Simpler

Before: Transformer + Bidirectional LSTM (~50K parameters)
After: Single GRU (~8K parameters)
Result: Can't memorize training data as easily

2. Fixed Data Leakage

Before: Created sequences → then split (sequences overlap!)
After: Split data → then create sequences (no overlap!)
Result: Validation metrics are now trustworthy

3. Added Triple Regularization

Noise: Augments training data (prevents exact memorization)
Dropout: Randomly disables neurons (forces redundancy)
Weight Decay: Penalizes large weights (keeps model simple)
Result: Model learns patterns, not specific examples

Version 13:
Complex Model + Data Leakage + No Regularization = Overfitting

New GRU Model:
Simple Model + Clean Splits + Triple Regularization = Good Generalization

to summarize, we made the model simpler, we made sure there was no data leakage,  we included regularlization by adding noise, drop, and weight decay

I had diverging loss curves. Want converging ones where validation loss closely tracks training loss.

