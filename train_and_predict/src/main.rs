#![feature(asm)]
extern crate rusty_machine;

use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;

use rusty_machine::learning::logistic_reg::LogisticRegressor;
use rusty_machine::learning::SupModel;
use rusty_machine::linalg::Vector;

use rusty_machine::data::tokenizers::NaiveTokenizer;
use rusty_machine::data::vectorizers::text::FreqVectorizer;
use rusty_machine::data::vectorizers::Vectorizer;

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
    let training_file = File::open("../../movie_review/movie_data/full_train.txt")
        .expect("Something went wrong reading the file");
    let training_buf_reader = BufReader::new(training_file);

    let mut training_data: Vec<String> = Vec::new();
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

    let mut freq_vectorizer = FreqVectorizer::<f64, NaiveTokenizer>::new(NaiveTokenizer::new());
    let mut training_str: Vec<&str> = Vec::new();
    for string in training_data.iter() {
        training_str.push(string);
    }
    freq_vectorizer.fit(&training_str).unwrap();
    let vectorized = freq_vectorizer.vectorize(&training_str).unwrap();

    println!("Start Training");

    // Training
    let mut log_mod = LogisticRegressor::default();
    let mut vector = vec![0.0; 25000];
    for i in 0..12500 {
        vector[i] = 1.0;
    }

    let targets = Vector::new(vector);
    log_mod.train(&vectorized, &targets).unwrap();
    println!("Finish Training");

    // Predict
    let test_file = File::open("../../movie_review/negative.txt")
        .expect("Something went wrong reading the file");
    let test_buf_reader = BufReader::new(test_file);
    let mut test_data: Vec<String> = Vec::new();
    for mut line in test_buf_reader.lines() {
        match line {
            Ok(mut line) => {
                test_data.push(line);
            }

            Err(_) => {
                println!("Error in reading the data");
            }
        }
    }

    let mut freq_vectorizer = FreqVectorizer::<f64, NaiveTokenizer>::new(NaiveTokenizer::new());
    let mut test_str: Vec<&str> = Vec::new();
    for string in test_data.iter() {
        test_str.push(string);
    }
    freq_vectorizer.fit(&test_str).unwrap();
    let vectorized = freq_vectorizer.vectorize(&test_str).unwrap();
    let start = rdtsc();
    let output = log_mod.predict(&vectorized).unwrap();
    println!("{}, CPU Cycles {}", output, rdtsc() - start);
}
