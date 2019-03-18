#![feature(asm)]
extern crate rusty_machine;
extern crate vectorizer;

use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;

use rusty_machine::learning::logistic_reg::LogisticRegressor;
use rusty_machine::learning::SupModel;
use rusty_machine::linalg::Matrix;
use rusty_machine::linalg::Vector;

use vectorizer::countvectorizer::CountVectorizer;

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

fn main() {
    // Data reading and pre-processing
    let training_file = File::open("/home/ankit/workspace/movie_review/movie_data/full_train.txt")
        .expect("Something went wrong reading the file");
    let test_file = File::open("/home/ankit/workspace/movie_review/movie_data/full_test.txt")
        .expect("Something went wrong reading the file");

    let training_buf_reader = BufReader::new(training_file);
    let test_buf_reader = BufReader::new(test_file);

    let mut training_data: Vec<String> = Vec::new();
    let mut test_data: Vec<String> = Vec::new();

    for mut line in training_buf_reader.lines() {
        match line {
            Ok(mut line) => {
                training_data.push(line);
            }

            Err(_) => {
                println!("Error in reading the data");
            }
        }
    }

    for line in test_buf_reader.lines() {
        match line {
            Ok(mut line) => {
                test_data.push(line);
            }

            Err(_) => {
                println!("Error in reading the data");
            }
        }
    }

    let mut vectorizer = CountVectorizer::new((1, 2), "lower");
    let train = vectorizer.fit_transform(training_data.iter().map(|s| s.as_str()).collect());
    let test = vectorizer.fit_transform(test_data.iter().map(|s| s.as_str()).collect());
    println!("{} {}", train.rows(), train.cols());
    println!("{} {}", test.rows(), test.cols());

    // Training
    let mut log_mod = LogisticRegressor::default();
    let mut vector = vec![0.0; 25000];
    for i in 0..12500 {
        vector[i] = 1.0;
    }

    let inputs = Matrix::new(4, 1, vec![1.0, 3.0, 5.0, 7.0]);
    let targets = Vector::new(vec![0., 0., 1., 1.]);
    log_mod.train(&inputs, &targets).unwrap();

    // Predict
    let new_point = Matrix::new(1, 1, vec![10.]);
    let start = rdtsc();
    let output = log_mod.predict(&new_point).unwrap();
    println!("{}, CPU Cycles {}", output, rdtsc() - start);
}
