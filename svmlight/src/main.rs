#![feature(asm)]
#![allow(non_snake_case)]
extern crate bincode;
extern crate rustlearn;

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::str::FromStr;

use rustlearn::ensemble::random_forest;
use rustlearn::linear_models::sgdclassifier;
use rustlearn::metrics;
use rustlearn::prelude::*;
use rustlearn::trees::decision_tree;

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

fn get_raw_data(filename: &str) -> String {
    let path = Path::new(filename);

    let raw_data = match File::open(&path) {
        Err(_) => {
            panic!("Error in opening file {}", filename);
        }
        Ok(mut file) => {
            let mut file_data = String::new();
            file.read_to_string(&mut file_data).unwrap();
            file_data
        }
    };

    raw_data
}

fn build_x_matrix(data: &str, rows: usize, cols: usize) -> SparseRowArray {
    let mut array = SparseRowArray::zeros(rows, cols);
    let mut row_num = 0;

    for (_row, line) in data.lines().enumerate() {
        let mut col_num = 0;
        for col_str in line.split_whitespace() {
            if col_num > 0 {
                let split: Vec<&str> = col_str.split(":").collect();
                array.set(
                    row_num,
                    usize::from_str(split[0]).unwrap(),
                    f32::from_str(split[1]).unwrap(),
                );
            }
            col_num += 1;
        }
        row_num += 1;
    }

    array
}

fn build_col_matrix(data: &str, rows: usize, cols: usize) -> SparseColumnArray {
    let mut array = SparseColumnArray::zeros(rows, cols);
    let mut row_num = 0;

    for (_row, line) in data.lines().enumerate() {
        let mut col_num = 0;
        for col_str in line.split_whitespace() {
            if col_num > 0 {
                let split: Vec<&str> = col_str.split(":").collect();
                array.set(
                    row_num,
                    usize::from_str(split[0]).unwrap(),
                    f32::from_str(split[1]).unwrap(),
                );
            }
            col_num += 1;
        }
        row_num += 1;
    }

    array
}

fn build_y_array(data: &str) -> Array {
    let mut y = Vec::new();

    for line in data.lines() {
        for datum_str in line.split_whitespace() {
            let datum = datum_str.parse::<i32>().unwrap();
            y.push(datum);
        }
    }

    Array::from(y.iter().map(|&x| x as f32).collect::<Vec<f32>>())
}

fn get_train_data() -> (SparseRowArray, SparseRowArray) {
    let X_train = build_x_matrix(&get_raw_data("./../data/train.feat"), 25000, 89527);
    let X_test = build_x_matrix(&get_raw_data("./../data/test.feat"), 25000, 89527);

    (X_train, X_test)
}

fn get_target_data() -> (Array, Array) {
    let y_train = build_y_array(&get_raw_data("./../data/target"));
    let y_test = build_y_array(&get_raw_data("./../data/target"));

    (y_train, y_test)
}

fn run_sgdclassifier(
    X_train: &SparseRowArray,
    X_test: &SparseRowArray,
    y_train: &Array,
    y_test: &Array,
) -> Vec<u8> {
    println!("Running SGDClassifier...");

    let num_epochs = 200;

    let mut model = sgdclassifier::Hyperparameters::new(X_train.cols()).build();

    for _ in 0..num_epochs {
        model.fit(X_train, y_train).unwrap();
    }

    let predictions = model.predict(X_test).unwrap();
    let accuracy = metrics::accuracy_score(y_test, &predictions);

    println!("SGDClassifier accuracy: {}%", accuracy * 100.0);

    let serialized = bincode::serialize(&model).unwrap();
    println!("{}", serialized.len());
    let start = rdtsc();
    let model: sgdclassifier::SGDClassifier = bincode::deserialize(&serialized).unwrap();
    let diff1 = rdtsc() - start;

    let X = build_x_matrix(&get_raw_data("./../data/positive.feat"), 1, 89527);
    let Y = build_x_matrix(&get_raw_data("./../data/negative.feat"), 1, 89527);

    let start = rdtsc();
    let pos = model.predict(&X).unwrap().data()[0];
    let neg = model.predict(&Y).unwrap().data()[0];
    let diff2 = rdtsc() - start;

    println!("Positive {:#?}, Negative {:#?}", pos, neg);
    println!("CPU cycle for deserialization {}", diff1);
    println!("CPU cycle for 2 prediction {}\n", diff2);

    serialized
}

fn run_decision_tree(
    X_train: &SparseRowArray,
    X_test: &SparseRowArray,
    y_train: &Array,
    y_test: &Array,
) -> Vec<u8> {
    println!("Running DecisionTree...");

    let X_train = SparseColumnArray::from(X_train);
    let X_test = SparseColumnArray::from(X_test);

    let mut model = decision_tree::Hyperparameters::new(X_train.cols()).build();

    model.fit(&X_train, y_train).unwrap();

    let predictions = model.predict(&X_test).unwrap();
    let accuracy = metrics::accuracy_score(y_test, &predictions);

    println!("DecisionTree accuracy: {}%", accuracy * 100.0);

    let serialized = bincode::serialize(&model).unwrap();
    println!("{}", serialized.len());
    let start = rdtsc();
    let model: decision_tree::DecisionTree = bincode::deserialize(&serialized).unwrap();
    let diff1 = rdtsc() - start;

    let X = build_col_matrix(&get_raw_data("./../data/positive.feat"), 1, 89527);
    let Y = build_col_matrix(&get_raw_data("./../data/negative.feat"), 1, 89527);
    let start = rdtsc();
    let pos = model.predict(&X).unwrap().data()[0];
    let neg = model.predict(&Y).unwrap().data()[0];
    let diff2 = rdtsc() - start;

    println!("Positive {:#?}, Negative {:#?}", pos, neg);
    println!("CPU cycle for deserialization {}", diff1);
    println!("CPU cycle for 2 prediction {}\n", diff2);

    serialized
}

fn run_random_forest(
    X_train: &SparseRowArray,
    X_test: &SparseRowArray,
    y_train: &Array,
    y_test: &Array,
) -> Vec<u8> {
    println!("Running RandomForest...");

    let num_trees = 20;

    let tree_params = decision_tree::Hyperparameters::new(X_train.cols());
    let mut model = random_forest::Hyperparameters::new(tree_params, num_trees).build();

    model.fit(X_train, y_train).unwrap();

    let predictions = model.predict(X_test).unwrap();
    let accuracy = metrics::accuracy_score(y_test, &predictions);

    println!("RandomForest accuracy: {}%", accuracy * 100.0);
    let serialized = bincode::serialize(&model).unwrap();
    println!("{}", serialized.len());
    let start = rdtsc();
    let model: random_forest::RandomForest = bincode::deserialize(&serialized).unwrap();
    let diff1 = rdtsc() - start;

    let X = build_x_matrix(&get_raw_data("./../data/positive.feat"), 1, 89527);
    let Y = build_x_matrix(&get_raw_data("./../data/negative.feat"), 1, 89527);

    let start = rdtsc();
    let pos = model.predict(&X).unwrap().data()[0];
    let neg = model.predict(&Y).unwrap().data()[0];
    let diff2 = rdtsc() - start;

    println!("Positive {:#?}, Negative {:#?}", pos, neg);
    println!("CPU cycle for deserialization {}", diff1);
    println!("CPU cycle for 2 prediction {}\n", diff2);

    serialized
}

pub fn run_ml_application() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (X_train, X_test) = get_train_data();
    let (y_train, y_test) = get_target_data();

    println!(
        "Training data: {} by {} matrix with {} nonzero entries",
        X_train.rows(),
        X_train.cols(),
        X_train.nnz()
    );
    println!(
        "Test data: {} by {} matrix with {} nonzero entries\n",
        X_test.rows(),
        X_test.cols(),
        X_test.nnz()
    );

    let sgd = run_sgdclassifier(&X_train, &X_test, &y_train, &y_test);
    let d_tree = run_decision_tree(&X_train, &X_test, &y_train, &y_test);
    let r_forest = run_random_forest(&X_train, &X_test, &y_train, &y_test);

    (sgd, d_tree, r_forest)
}

fn main() {
    run_ml_application();
}
