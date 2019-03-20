all:
	(cd svmlight; cargo run --release --bin svmlight)

clean:
	(cd svmlight; cargo clean)
