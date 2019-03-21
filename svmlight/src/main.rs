#![feature(asm)]
#![allow(non_snake_case)]
extern crate rustlearn;

use std::fs::File;
use std::io::Read;
use std::path::Path;

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
            panic!("Error in opening file");
        }
        Ok(mut file) => {
            println!("Reading data for {}", filename);
            let mut file_data = String::new();
            file.read_to_string(&mut file_data).unwrap();
            file_data
        }
    };

    raw_data
}

fn build_x_matrix(data: &str, rows: usize) -> SparseRowArray {
    let mut array = SparseRowArray::zeros(rows, 10);
    let mut row_num = 0;

    for (_row, line) in data.lines().enumerate() {
        let mut col_num = 0;
        for col_str in line.split_whitespace() {
            let data = col_str.parse::<usize>().unwrap();
            array.set(row_num, col_num, data as f32);
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

    Array::from(
        y.iter()
            .map(|&x| match x {
                0 => 0.0,
                _ => 1.0,
            }).collect::<Vec<f32>>(),
    )
}

fn get_train_data() -> (SparseRowArray, SparseRowArray) {
    let X_train = build_x_matrix(&get_raw_data("./../data/train.data"), 25010);
    let X_test = build_x_matrix(&get_raw_data("./../data/test.data"), 1999999);

    (X_train, X_test)
}

fn get_target_data() -> (Array, Array) {
    let y_train = build_y_array(&get_raw_data("./../data/train.target"));
    let y_test = build_y_array(&get_raw_data("./../data/test.target"));

    (y_train, y_test)
}

fn run_sgdclassifier(
    X_train: &SparseRowArray,
    X_test: &SparseRowArray,
    y_train: &Array,
    y_test: &Array,
) {
    println!("Running SGDClassifier...");

    let num_epochs = 10;

    let mut model = sgdclassifier::Hyperparameters::new(X_train.cols())
        .learning_rate(0.5)
        .l2_penalty(0.000001)
        .build();

    for _ in 0..num_epochs {
        model.fit(X_train, y_train).unwrap();
    }

    let predictions = model.predict(X_test).unwrap();
    let accuracy = metrics::accuracy_score(y_test, &predictions);

    println!("SGDClassifier accuracy: {}", accuracy);
}

fn run_decision_tree(
    X_train: &SparseRowArray,
    X_test: &SparseRowArray,
    y_train: &Array,
    y_test: &Array,
) {
    println!("Running DecisionTree...");

    let X_train = SparseColumnArray::from(X_train);
    let X_test = SparseColumnArray::from(X_test);

    let mut model = decision_tree::Hyperparameters::new(X_train.cols()).build();

    model.fit(&X_train, y_train).unwrap();

    let predictions = model.predict(&X_test).unwrap();
    let accuracy = metrics::accuracy_score(y_test, &predictions);

    println!("DecisionTree accuracy: {}", accuracy);
}

fn run_random_forest(
    X_train: &SparseRowArray,
    X_test: &SparseRowArray,
    y_train: &Array,
    y_test: &Array,
) {
    println!("Running RandomForest...");

    let num_trees = 10;

    let tree_params = decision_tree::Hyperparameters::new(X_train.cols());
    let mut model = random_forest::Hyperparameters::new(tree_params, num_trees).build();

    model.fit(X_train, y_train).unwrap();

    let predictions = model.predict(X_test).unwrap();
    let accuracy = metrics::accuracy_score(y_test, &predictions);

    println!("RandomForest accuracy: {}", accuracy);
}

fn main() {
    let (X_train, X_test) = get_train_data();
    let (y_train, y_test) = get_target_data();

    println!(
        "Training data: {} by {} matrix with {} nonzero entries",
        X_train.rows(),
        X_train.cols(),
        X_train.nnz()
    );
    println!(
        "Test data: {} by {} matrix with {} nonzero entries",
        X_test.rows(),
        X_test.cols(),
        X_test.nnz()
    );

    run_sgdclassifier(&X_train, &X_test, &y_train, &y_test);
    run_decision_tree(&X_train, &X_test, &y_train, &y_test);
    run_random_forest(&X_train, &X_test, &y_train, &y_test);
}
