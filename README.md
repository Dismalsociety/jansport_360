"We're using an LSTM because it has memory that helps it understand how pressure responds to flow changes over time. I discovered data leakage because our validation loss started at zero - a red flag that the model was 'cheating.' When we randomly split our data, the model was learning from future data to predict the past, essentially memorizing answers instead of learning patterns. By fixing this to train only on past data to predict future data, the model now actually learns the pressure dynamics instead of just memorizing."

	1.	Increase GAP to 100 (line with GAP = 50)
	2.	Reduce HIDDEN_SIZE to 16
	3.	Increase DROPOUT to 0.6
	4.	Reduce sequence length to 30
