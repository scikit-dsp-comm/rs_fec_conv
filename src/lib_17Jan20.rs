extern crate pyo3;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::PyType;
/* use rayon::prelude::*;
use std::fs;
use std::path::PathBuf;
use pyo3::types::{PyBytes, PyString}; */

#[pyclass(module = "rs_fec_conv")]
#[text_signature = "(c, d, /)"]
struct MyClass {}

#[pymethods]

impl MyClass {
    // the signature for the constructor is attached
    // to the struct definition instead.
    #[new]
    fn new(c: i32, d: &str) -> Self {
        Self {}
    }
    // the self argument should be written $self
    #[text_signature = "($self, e, f)"]
    fn my_method(&self, e: i32, f: i32) -> i32 {
        e + f
    }
    #[classmethod]
    #[text_signature = "(cls, e, f)"]
    fn my_class_method(cls: &PyType, e: i32, f: i32) -> i32 {
        e + f
    }
    #[staticmethod]
    #[text_signature = "(e, f)"]
    fn my_static_method(e: i32, f: i32) -> i32 {
        e + f
    }
}
/*
#[pyclass(module = "rs_fec_conv")]
struct FecConv {
    path: PathBuf,
}
*/

/*
#[pymethods]
impl FecConv {
    #[new]
    fn new(obj: &PyRawObject, path: String) {
        obj.init(FecConv {
            path: PathBuf::from(path),
        });
    }

    /// Searches for the word, parallelized by rayon
    fn search(&self, py: Python<'_>, search: String) -> PyResult<usize> {
        let contents = fs::read_to_string(&self.path)?;

        let count = py.allow_threads(move || {
            contents
                .par_lines()
                .map(|line| count_line(line, &search))
                .sum()
        });
        Ok(count)
    }

    /// Searches for a word in a classic sequential fashion
    fn search_sequential(&self, needle: String) -> PyResult<usize> {
        let contents = fs::read_to_string(&self.path)?;

        let result = contents.lines().map(|line| count_line(line, &needle)).sum();

        Ok(result)
    }
} 

*/
fn matches(word: &str, needle: &str) -> bool {
    let mut needle = needle.chars();
    for ch in word.chars().skip_while(|ch| !ch.is_alphabetic()) {
        match needle.next() {
            None => {
                return !ch.is_alphabetic();
            }
            Some(expect) => {
                if ch.to_lowercase().next() != Some(expect) {
                    return false;
                }
            }
        }
    }
    return needle.next().is_none();
}

/// Count the occurences of needle in line, case insensitive
#[pyfunction]
fn count_line(line: &str, needle: &str) -> usize {
    let mut total = 0;
    for word in line.split(' ') {
        if matches(word, needle) {
            total += 1;
        }
    }
    total
}
 
#[pyfunction]
fn char_count(word: &str) -> usize {

	let mut total = 0;
	total = word.chars().count();
	total
}

#[pymodule]
fn rs_fec_conv(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(char_count))?;
	m.add_wrapped(wrap_pyfunction!(count_line))?;
	//m.add_class::<FecConv>()?;
	m.add_class::<MyClass>()?;
	
    Ok(())
}