#![feature(asm)]
extern crate rusty_machine;

use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::str::FromStr;

use rusty_machine::analysis::score::accuracy;
use rusty_machine::learning::logistic_reg::LogisticRegressor;
use rusty_machine::learning::SupModel;
use rusty_machine::linalg::Matrix;
use rusty_machine::linalg::Vector;

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

fn load_svmlight_file(filename: &str) -> (Vec<f64>, Vec<f64>, usize) {
    // Data reading and pre-processing
    let training_file = File::open(filename).expect("Something went wrong reading the file");
    let training_buf_reader = BufReader::new(training_file);

    let mut y_train: Vec<f64> = Vec::new();
    let mut x_train: Vec<f64> = Vec::new();
    let mut num_lines = 0;
    for mut line in training_buf_reader.lines() {
        match line {
            Ok(mut line) => {
                let mut index = 0;
                for word in line.split_whitespace() {
                    if index == 0 {
                        y_train.push(f64::from_str(word).unwrap());
                    } else {
                        let split: Vec<&str> = word.split(":").collect();
                        x_train.push(f64::from_str(split[1]).unwrap());
                    }
                    index += 1;
                }
                num_lines += 1;
            }

            Err(_) => {
                println!("Error in reading the data");
            }
        }
    }
    (x_train, y_train, num_lines)
}

fn main() {
    let (x_train, y_train, num_lines) = load_svmlight_file("poker");
    let rows = num_lines;
    let cols = x_train.len() / num_lines;
    let inputs = Matrix::new(rows, cols, x_train);
    let targets = Vector::new(y_train);

    let mut log_mod = LogisticRegressor::default();

    // Train the model
    log_mod.train(&inputs, &targets).unwrap();

    // Now we'll predict a new point
    let new_point = Matrix::new(
        1,
        cols,
        vec![1.0, 9.0, 1.0, 12.0, 1.0, 10.0, 1.0, 11.0, 1.0, 13.0],
    );
    let start = rdtsc();
    let output = log_mod.predict(&new_point).unwrap();
    let stop = rdtsc() - start;

    println!(
        "{:#?} CPU Cycles {} Accuracy {}",
        output[0],
        stop,
        accuracy(log_mod.predict(&inputs).unwrap().iter(), targets.iter())
    );
}
