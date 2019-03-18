all:
	(cd train_and_predict; cargo run --release --bin train_and_predict)

clean:
	(cd train_and_predict; cargo clean)
