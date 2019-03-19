#![feature(asm)]
extern crate rustlearn;

use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;

use rustlearn::feature_extraction::DictVectorizer;
use rustlearn::linear_models::sgdclassifier::Hyperparameters;
use rustlearn::linear_models::sgdclassifier::SGDClassifier;
use rustlearn::metrics;
use rustlearn::prelude::*;

// Return a 64-bit timestamp using the rdtsc instruction.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn rdtsc() -> u64 {
    unsafe {
        let lo: u32;
        let hi: u32;
        asm!("rdtsc" : "={eax}"(lo), "={edx}"(hi) : : : "volatile");
        (((hi as u64) << 32) | lo as u64)
    }
}

fn read_data(filename: &str) -> Vec<Vec<String>> {
    // Data reading and pre-processing
    let training_file = File::open(filename).expect("Something went wrong reading the file");
    let training_buf_reader = BufReader::new(training_file);

    let mut training_data: Vec<Vec<String>> = Vec::new();
    for mut line in training_buf_reader.lines() {
        match line {
            Ok(mut line) => {
                let mut vector: Vec<String> = Vec::new();
                for word in line.split_whitespace() {
                    vector.push(word.to_string());
                }
                training_data.push(vector);
            }

            Err(_) => {
                println!("Error in reading the data");
            }
        }
    }
    training_data
}

fn vectorize_data(training_data: Vec<Vec<String>>) -> SparseRowArray {
    let mut vectorizer = DictVectorizer::new();
    for index in 0..training_data.len() {
        let review = &training_data[index];
        for word in review.iter() {
            if index < 12500 {
                vectorizer.partial_fit(index, word, 1.0);
            } else {
                vectorizer.partial_fit(index, word, 0.0);
            }
        }
    }
    vectorizer.transform()
}

fn train_model(train: &SparseRowArray, target: &Array) -> SGDClassifier {
    let mut model = Hyperparameters::new(train.cols())
        .learning_rate(0.5)
        .l2_penalty(0.000001)
        .build();
    let num_epochs = 20;
    for _ in 0..num_epochs {
        model.fit(train, target).unwrap();
    }
    model
}

fn main() {
    let training_data = read_data("../../movie_review/movie_data/full_train.txt");
    let train = vectorize_data(training_data);
    let mut y = vec![0.0; 25000];
    for i in 0..12500 {
        y[i] = 1.0;
    }
    let target = Array::from(y);
    let model = train_model(&train, &target);

    // Predict
    let start = rdtsc();
    let predictions = model.predict(&train).unwrap();
    let stop = rdtsc() - start;
    let accuracy = metrics::accuracy_score(&target, &predictions);
    println!("SGDClassifier accuracy: {}% \nCPU Cycles {}", accuracy*100.0, stop);
}
