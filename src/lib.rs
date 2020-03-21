extern crate pyo3;
extern crate array2d;
extern crate transpose;

use ndarray::ArrayViewD;
use numpy::*;
use pyo3::prelude::*;
use array2d::Array2D;

#[pymodule]
fn rs_fec_conv(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
	
	/// Format an float of int to binary
	fn binaryf(num: f64) -> String {
		// Format an float of int to binary
		let val = num as i32;
		format!("{:b}", val)
	}
	
	/// Format an int to binary
	fn binaryi(num: i32, bin_len: usize) -> String {
		// Format an integer to binary with leading zeros of length bin_len
		format!("{:01$b}", num, bin_len)
	}

	/// Convolutionally encode a signal with 1/2 or 1/3 rate encoding
	///
	/// Rate 1/2
    /// K=3 ('111','101')
    /// K=4 ('1111','1101')
    /// K=5 ('11101','10011')
    /// K=6 ('111101','101011')
    /// K=7 ('1111001','1011011')
    /// K=8 ('11111001','10100111')
    /// K=9 ('111101011','101110001')
	///
    /// Rate 1/3
    /// K=3 ('111','111','101')
    /// K=4 ('1111','1101','1011')
    /// K=5 ('11111','11011','10101')
    /// K=6 ('111101','101011','100111')
    /// K=7 ('1111001','1100101','1011011')
    /// K=8 ('11110111','11011001','10010101')
	///
	/// Benjamin Roepken, Mark Wickert, and Andrew Smit February 2020
    ///
    /// Parameters
    /// ----------
    /// input: Incoming signal to encode (np.array([float, float, ...]))
	/// state: Starting state of length G[0] - 1 (String)
	/// G: G1, G2, (G3) values (np.array([String, String, ...]))
	/// output: Initialize output array of size len(input) * len(G) (np.array([float, float, ...]))
	/// Depth: Decision Depth (int)
	///
	/// Return
	/// ------
	/// output: Encoded signal of length len(input) * len(G) (np.array([float, float, ...]))
	/// state: Output state (String)
	///
	fn conv_encoder(input: ArrayViewD<f64>, state: String, g: ArrayViewD<f64>) -> (Vec<f64>, String) {	
		
		// Convolutionally encode a signal with 1/2 or 1/3 rate encoding
		let mut output = vec![];
		let mut state_out = state;
		let state_len = state_out.len();
		let input1 = &input * 1.0;
		let g1 = &g * 1.0;
		
		
		// Initialize values
		let constraint_length = binaryf(g1[0]).len();
		let _n_states = 2_i32.pow(constraint_length as u32 - 1);
		
		// 1/2 rate encoding
		if g1.len() == 2 {
			for (_i, elem_i) in input1.iter().enumerate() {
				let mut u1 = *elem_i as i32;
				let mut u2 = *elem_i as i32;
				
				// XOR G1 and G2
				for j in 1..constraint_length {
					// XOR G1 with current state
					let g10_s = &binaryf(g1[0])[j..(j+1)];
					let g10: i32 = g10_s.trim().parse().unwrap();
					if g10 == 1 {
						let u1_temp_s = &state_out[(j-1)..j];
						let u1_temp: i32 = u1_temp_s.trim().parse().unwrap();
						u1 = u1 ^ u1_temp;
					}
					
					// XOR G2 with current state
					let g11_s= &binaryf(g1[1])[j..(j+1)];
					let g11: i32 = g11_s.trim().parse().unwrap();
					if g11 == 1 {						
						let u2_temp_s = &state_out[(j-1)..j];
						let u2_temp: i32 = u2_temp_s.trim().parse().unwrap();
						u2 = u2 ^ u2_temp;
					}
				}
				
				// G1 first, G2 second
				output.push(u1.into());
				output.push(u2.into());
				state_out = binaryf(*elem_i) + &state_out[..(state_len-1)];	
			}
		}
		
		// 1/3 rate encoding
		else if g1.len() == 3 {
			for (_i, elem_i) in input1.iter().enumerate() {
				let mut u1 = *elem_i as i32;
				let temp_g0_s = &binaryf(g1[0])[0..1];
				let temp_g0: i32 = temp_g0_s.trim().parse().unwrap();
				if temp_g0 != 1 {
					u1 = 0;
				}
				
				let mut u2 = *elem_i as i32;
				let temp_g1_s = &binaryf(g1[1])[0..1];
				let temp_g1: i32 = temp_g1_s.trim().parse().unwrap();
				if temp_g1 != 1 {
					u2 = 0;
				}
				
				let mut u3 = *elem_i as i32;
				let temp_g2_s = &binaryf(g1[2])[0..1];
				let temp_g2: i32 = temp_g2_s.trim().parse().unwrap();
				if temp_g2 != 1 {
					u3 = 0;
				}
				
				// XOR G1 and G2 and G3
				for j in 1..constraint_length {
					// XOR G1 with current state
					let g10_s = &binaryf(g1[0])[j..(j+1)];
					let g10: i32 = g10_s.trim().parse().unwrap();
					if g10 == 1 {
						let u1_temp_s = &state_out[(j-1)..j];
						let u1_temp: i32 = u1_temp_s.trim().parse().unwrap();
						u1 = u1 ^ u1_temp;
					}
					
					// XOR G2 with current state
					let g11_s = &binaryf(g1[1])[j..(j+1)];
					let g11: i32 = g11_s.trim().parse().unwrap();
					if g11 == 1 {						
						let u2_temp_s = &state_out[(j-1)..j];
						let u2_temp: i32 = u2_temp_s.trim().parse().unwrap();
						u2 = u2 ^ u2_temp;
					}
					
					// XOR G3 with current state
					let g12_s= &binaryf(g1[2])[j..(j+1)];
					let g12: i32 = g12_s.trim().parse().unwrap();
					if g12 == 1 {						
						let u3_temp_s = &state_out[(j-1)..j];
						let u3_temp: i32 = u3_temp_s.trim().parse().unwrap();
						u3 = u3 ^ u3_temp;
					}
				}
				
				// G1 first, G2 second, G3 third
				output.push(u1.into());
				output.push(u2.into());
				output.push(u3.into());
				state_out = binaryf(*elem_i) + &state_out[..(state_len-1)];	
			}
		}
		// Return Error
		else {
			//panic!("Error: Use either rate 1/2 or 1/3");
			//Err(From::from("No matching cities with a population were found."))
			state_out = "Error: Use either rate 1/2 or 1/3 (len(G))".to_string();
		}

		//return state_out
		return (output, state_out)
			
	}
	
	/// Convolutionally encode a signal with 1/2 or 1/3 rate encoding
	///
	/// Rate 1/2
    /// K=3 ('111','101')
    /// K=4 ('1111','1101')
    /// K=5 ('11101','10011')
    /// K=6 ('111101','101011')
    /// K=7 ('1111001','1011011')
    /// K=8 ('11111001','10100111')
    /// K=9 ('111101011','101110001')
	///
    /// Rate 1/3
    /// K=3 ('111','111','101')
    /// K=4 ('1111','1101','1011')
    /// K=5 ('11111','11011','10101')
    /// K=6 ('111101','101011','100111')
    /// K=7 ('1111001','1100101','1011011')
    /// K=8 ('11110111','11011001','10010101')
	///
	/// Benjamin Roepken, Mark Wickert, and Andrew Smit February 2020
    ///
    /// Parameters
    /// ----------
    /// input: Incoming signal to encode (np.array([float, float, ...]))
	/// state: Starting state of length G[0] - 1 (String)
	/// G: G1, G2, (G3) values (np.array([String, String, ...]))
	/// output: Initialize output array of size len(input) * len(G) (np.array([float, float, ...]))
	/// Depth: Decision Depth (int)
	///
	/// Return
	/// ------
	/// output: Encoded signal of length len(input) * len(G) (np.array([float, float, ...]))
	/// state: Output state (String)
	///
	#[pyfn(m, "conv_encoder")]	
	fn conv_encoder_py(_py: Python, input: &PyArrayDyn<f64>, state: String, g: &PyArrayDyn<f64>) -> (Vec<f64>, String) {
		
		// Convolutionally encode a signal with 1/2 or 1/3 rate encoding
		let input = input.as_array();
		let g = g.as_array();

		// Pass parameters to conv_encoder
		let (output, state_out) = conv_encoder(input, state, g);
		
		return (output, state_out)
	}
	
	fn conv_encoder2(input: &Vec<f64>, state: String, g: &ArrayViewD<f64>) -> (Vec<f64>, String) {	
		
		// Convolutionally encode a signal with 1/2 or 1/3 rate encoding
		let mut state_out = state;
		let state_len = state_out.len();
		let g1 = g;
		let mut output_out = vec![0.0; input.len() * g1.len()];
		
		// Initialize values
		let constraint_length = binaryf(g1[0]).len();
		let _n_states = 2_i32.pow(constraint_length as u32 - 1);
		
		// 1/2 rate encoding
		if g1.len() == 2 {
			let mut counter = 0;
			for (_i, elem_i) in input.iter().enumerate() {
				let mut u1 = *elem_i as i32;
				let mut u2 = *elem_i as i32;
				
				// XOR G1 and G2
				for j in 1..constraint_length {
				
					// XOR G1 with current state
					let g10_s= &binaryf(g1[0])[j..(j+1)];
					let g10: i32 = g10_s.trim().parse().unwrap();
					if g10 == 1 {
						let u1_temp_s = &state_out[(j-1)..j];
						let u1_temp: i32 = u1_temp_s.trim().parse().unwrap();
						u1 = u1 ^ u1_temp;
					}
					
					// XOR G2 with current state
					let g11_s= &binaryf(g1[1])[j..(j+1)];
					let g11: i32 = g11_s.trim().parse().unwrap();
					if g11 == 1 {						
						let u2_temp_s = &state_out[(j-1)..j];
						let u2_temp: i32 = u2_temp_s.trim().parse().unwrap();
						u2 = u2 ^ u2_temp;
					}
				}
				
				// G1 first, G2 second
				output_out[counter] = u1.into();
				counter += 1;
				output_out[counter] = u2.into();
				counter += 1;
				state_out = binaryf(*elem_i) + &state_out[..(state_len-1)];	
			}
		}
		
		// 1/3 rate encoding
		else if g1.len() == 3 {
			let mut counter = 0;
			for (_i, elem_i) in input.iter().enumerate() {
				let mut u1 = *elem_i as i32;
				let temp_g0_s = &binaryf(g1[0])[0..1];
				let temp_g0: i32 = temp_g0_s.trim().parse().unwrap();
				if temp_g0 != 1 {
					u1 = 0;
				}
				
				let mut u2 = *elem_i as i32;
				let temp_g1_s = &binaryf(g1[1])[0..1];
				let temp_g1: i32 = temp_g1_s.trim().parse().unwrap();
				if temp_g1 != 1 {
					u2 = 0;
				}
				
				let mut u3 = *elem_i as i32;
				let temp_g2_s = &binaryf(g1[2])[0..1];
				let temp_g2: i32 = temp_g2_s.trim().parse().unwrap();
				if temp_g2 != 1 {
					u3 = 0;
				}
				
				// XOR G1 and G2 and G3
				for j in 1..constraint_length {
					// XOR G1 with current state
					let g10_s = &binaryf(g1[0])[j..(j+1)];
					let g10: i32 = g10_s.trim().parse().unwrap();
					if g10 == 1 {
						let u1_temp_s = &state_out[(j-1)..j];
						let u1_temp: i32 = u1_temp_s.trim().parse().unwrap();
						u1 = u1 ^ u1_temp;
					}
					
					// XOR G2 with current state
					let g11_s = &binaryf(g1[1])[j..(j+1)];
					let g11: i32 = g11_s.trim().parse().unwrap();
					if g11 == 1 {						
						let u2_temp_s = &state_out[(j-1)..j];
						let u2_temp: i32 = u2_temp_s.trim().parse().unwrap();
						u2 = u2 ^ u2_temp;
					}
					
					// XOR G3 with current state
					let g12_s= &binaryf(g1[2])[j..(j+1)];
					let g12: i32 = g12_s.trim().parse().unwrap();
					if g12 == 1 {						
						let u3_temp_s = &state_out[(j-1)..j];
						let u3_temp: i32 = u3_temp_s.trim().parse().unwrap();
						u3 = u3 ^ u3_temp;
					}
				}
				
				// G1 first, G2 second, G3 third
				output_out[counter] = u1.into();
				counter += 1;
				output_out[counter] = u2.into();
				counter += 1;
				output_out[counter] = u3.into();
				counter += 1;
				state_out = binaryf(*elem_i) + &state_out[..(state_len-1)];	
			}
		}
		
		// Return Error
		else {
			state_out = "Error: Use either rate 1/2 or 1/3 (len(G))".to_string();
		}

		//return state_out
		return (output_out, state_out)
			
	}
	
	
	/// distance = bm_calc(ref_code_bits, rec_code_bits, metric_type)
    /// Branch metrics calculation
	///
	/// Benjamin Roepken, Mark Wickert, and Andrew Smit February 2020
	/// Parameters
    /// ----------
	/// ref_code_bits: Reference Code Bits (array<f64>)
	/// rec_code_bits: Received Code Bits (array<f64>)
	/// metric_type: 
    ///    'hard' - Hard decision metric. Expects binary or 0/1 input values.
    ///    'nquant' - unquantized soft decision decoding. Expects +/-1
    ///        input values.
    ///    'soft' - soft decision decoding.
    /// quant_level: The quantization level for soft decoding. Expected 
    /// input values between 0 and 2^quant_level-1. 0 represents the most 
    /// confident 0 and 2^quant_level-1 represents the most confident 1. 
    /// Only used for 'soft' metric type.
	/// 
	/// Returns
    /// -------
    /// distance: Distance between ref_code_bits and rec_code_bits (float)
	///
	fn bm_calc(ref_code_bits: f64, rec_code_bits: &Vec<f64>, metric_type: &String, quant_level: usize, rate: usize) -> f64 {
		
		// distance = bm_calc(ref_code_bits, rec_code_bits, metric_type)
        // Branch metrics calculation
		let mut distance: f64 = 0.0;
		
		// Squared distance metric
		if metric_type == "soft" {
			let bits = binaryi(ref_code_bits as i32, rate);
			for k in 0..bits.len() {
				let bits_k: i32 = bits[k..k+1].parse().unwrap();
				let ref_bit = 2_i32.pow(quant_level as u32 - 1) * bits_k;
				distance += (rec_code_bits[k] - ref_bit as f64).powf(2.0);
			}
		}
		
		// Hard decisions
		else if metric_type == "hard" {
			let bits = binaryi(ref_code_bits as i32, rate);
			for k in 0..rec_code_bits.len() {
				let bits_k: i32 = bits[k..k+1].parse().unwrap();
				distance += (rec_code_bits[k] - bits_k as f64).abs();
			}
		}
		
		// Unquantized
		else if metric_type == "unquant" {
			let bits = binaryi(ref_code_bits as i32, rate);
			for k in 0..bits.len() {
				let bits_k: i32 = bits[k..k+1].parse().unwrap();
				distance += (rec_code_bits[k] - bits_k as f64).powf(2.0);
			}
		}
		
		//Error statement
		else {
		}
		return distance
	}
	
	/// A method which performs Viterbi decoding of noisy bit stream,
    /// taking as input soft bit values centered on +/-1 and returning 
    /// hard decision 0/1 bits.
	///
	/// Benjamin Roepken and Mark Wickert February 2020
    ///
    /// Parameters
    /// ----------
    /// x: Received noisy bit values centered on +/-1 at one sample per bit
    /// metric_type: 
    ///    'hard' - Hard decision metric. Expects binary or 0/1 input values.
    ///    'nquant' - unquantized soft decision decoding. Expects +/-1
    ///        input values.
    ///    'soft' - soft decision decoding.
    /// quant_level: The quantization level for soft decoding. Expected 
    /// input values between 0 and 2^quant_level-1. 0 represents the most 
    /// confident 0 and 2^quant_level-1 represents the most confident 1. 
    /// Only used for 'soft' metric type.
	///
    /// Returns
    /// -------
    /// y: Decoded 0/1 bit stream
	///
	fn viterbi_decoder(input: ArrayViewD<f64>, metric_type: String, quant_level: usize, g: ArrayViewD<f64>, depth: usize) -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {	
	// Viterbi Decode a signal with 1/2 or 1/3 rate encoding
		
		// Initialize values
		let g1 = &g * 1.0;
		let constraint_length = binaryf(g1[0]).len();
		let n_states = 2_i32.pow(constraint_length as u32 - 1);

		// Number of channel symbols to process
		// Even for rate 1/2
		// Multiple of 3 for rate 1/3
		let input1 = &input * 1.0;
		let ns = input1.len();
		
		// Decoded bit sequence
		let mut output = vec![];
		let symbol_l = g1.len();
		
		// Initialize trellis paths
		let mut paths_cum_metrics = vec![vec![0.0; depth]; n_states as usize];
		let mut paths_traceback_states = vec![vec![0.0; depth]; n_states as usize];
		let mut paths_traceback_bits = vec![vec![0.0; depth]; n_states as usize];

		// Initialize trellis nodes
		let mut fn_0 = vec![0.0; n_states as usize];
		let mut tn_0 = vec![0.0; n_states as usize];
		let mut out_bits_0 = vec![0.0; n_states as usize];
		let vec0 = vec![0.0];
		
		let mut fn_1 = vec![0.0; n_states as usize];
		let mut tn_1 = vec![0.0; n_states as usize];
		let mut out_bits_1 = vec![0.0; n_states as usize];
		let vec1 = vec![1.0];
		

		for i in 0..n_states {
			fn_0[i as usize] = i as f64;
			fn_1[i as usize] = i as f64;
			
			// Start labeling with LSB on right (more common)
			let (output0, state0) = conv_encoder2(&vec0, binaryi(i, constraint_length - 1), &g);
			let (output1, state1) = conv_encoder2(&vec1, binaryi(i, constraint_length - 1), &g);
			tn_0[i as usize] = isize::from_str_radix(&state0, 2).unwrap() as f64;
			tn_1[i as usize] = isize::from_str_radix(&state1, 2).unwrap() as f64;
			
			// 1/2 rate
			if g1.len() == 2 {
				out_bits_0[i as usize] = 2.0 * output0[0] + output0[1];
				out_bits_1[i as usize] = 2.0 * output1[0] + output1[1];
			}
			
			// 1/3 rate
			if g1.len() == 3 {
				out_bits_0[i as usize] = 4.0 * output0[0] + 2.0 * output0[1] + output0[2];
				out_bits_1[i as usize] = 4.0 * output1[0] + 2.0 * output1[1] + output1[2];
			}
		}
		
		// Initialize trellis branches
		let mut branches_states1 = vec![0.0; n_states as usize];
		let mut branches_states2 = vec![0.0; n_states as usize];
		let mut branches_bits1 = vec![0.0; n_states as usize];
		let mut branches_bits2 = vec![0.0; n_states as usize];
		let mut branches_input1 = vec![0.0; n_states as usize];
		let mut branches_input2 = vec![0.0; n_states as usize];
		
		for i in 0..n_states {
			let mut match_zero_idx = Vec::new();
			let mut match_one_idx = Vec::new();
			for j in 0..tn_0.len() {
				if tn_0[j as usize] == i as f64 {
					match_zero_idx.push(j);
				}
				if tn_1[j as usize] == i as f64 {
					match_one_idx.push(j);
				}
			}
			
			// Zero index
			if match_zero_idx.len() > 0 {
				branches_states1[i as usize] = fn_0[match_zero_idx[0]];
				branches_states2[i as usize] = fn_0[match_zero_idx[1]];
				branches_bits1[i as usize] = out_bits_0[match_zero_idx[0]];
				branches_bits2[i as usize] = out_bits_0[match_zero_idx[1]];
				branches_input1[i as usize] = 0.0;
				branches_input2[i as usize] = 0.0;
			}
			
			// One index
			else if match_one_idx.len() > 0 {
				branches_states1[i as usize] = fn_1[match_one_idx[0]];
				branches_states2[i as usize] = fn_1[match_one_idx[1]];
				branches_bits1[i as usize] = out_bits_1[match_one_idx[0]];
				branches_bits2[i as usize] = out_bits_1[match_one_idx[1]];
				branches_input1[i as usize] = 1.0;
				branches_input2[i as usize] = 1.0;
			}
			else {
				// error statement
			}
		}
		
		// Calculate branch metrics and update traceback state and bits
		let mut cm_past = vec![0.0; n_states as usize];
		let mut tb_states_temp = vec![vec![0.0; depth - 1]; n_states as usize];
		let mut tb_bits_temp = vec![vec![0.0; depth - 1]; n_states as usize];
	
		for i in (0..ns).step_by(symbol_l) {
			let mut cm_present = vec![0.0; n_states as usize];
			
			// Get column vectors
			let paths_cum_metrics_array = Array2D::from_rows(&paths_cum_metrics);
			for (j, elem_j) in paths_cum_metrics_array.column_iter(0).enumerate() {
				cm_past[j as usize] = *elem_j ;
			}
			
			let paths_traceback_states_array = Array2D::from_rows(&paths_traceback_states);
			for j in 0..(depth - 1) {
				for (k, elem_k) in paths_traceback_states_array.column_iter(j).enumerate() {
					tb_states_temp[k as usize][j as usize] = *elem_k;
				}
			}
			
			let paths_traceback_bits_array = Array2D::from_rows(&paths_traceback_bits);
			for j in 0..(depth - 1) {
				for (k, elem_k) in paths_traceback_bits_array.column_iter(j).enumerate() {
					tb_bits_temp[k as usize][j as usize] = *elem_k;
				}
			}
			
			for j in 0..n_states {
				let mut input_arr = vec![0.0; symbol_l];
				let mut k_counter = 0;
				for k in i..(i+symbol_l)  {
					input_arr[k_counter] = input1[k];
					k_counter += 1;
				}
				
				// Calculate branch metrics
				let mut d1 = bm_calc(branches_bits1[j as usize], &input_arr, &metric_type, quant_level, g1.len());
				d1 += cm_past[branches_states1[j as usize] as usize];
				let mut d2 = bm_calc(branches_bits2[j as usize], &input_arr, &metric_type, quant_level, g1.len());
				d2 += cm_past[branches_states2[j as usize] as usize];
				
				// Find the survivor assuming minimum distance wins
				if d1 <= d2 {
					cm_present[j as usize] = d1;

					paths_traceback_states[j as usize][0] = branches_states1[j as usize];
					for k in 1..depth {
						let x = branches_states1[j as usize];
						paths_traceback_states[j as usize][k as usize] = tb_states_temp[x as usize][(k - 1) as usize];
					}

					paths_traceback_bits[j as usize][0] = branches_input1[j as usize];
					for k in 1..depth {
						let x = branches_states1[j as usize];
						paths_traceback_bits[j as usize][k as usize] = tb_bits_temp[x as usize][(k - 1) as usize];
					}	
				}

				// d2 < d1
				else {
					cm_present[j as usize] = d2;
					
					paths_traceback_states[j as usize][0] = branches_states2[j as usize];
					for k in 1..depth {
						let x = branches_states2[j as usize];
						paths_traceback_states[j as usize][k as usize] = tb_states_temp[x as usize][(k - 1) as usize];
					}
					
					paths_traceback_bits[j as usize][0] = branches_input2[j as usize];
					for k in 1..depth {
						let x = branches_states2[j as usize];
						paths_traceback_bits[j as usize][k as usize] = tb_bits_temp[x as usize][(k - 1) as usize];
					}
				}
			}
			
			// Update cumulative metric history
			for j in 0..n_states {
				for k in (1..depth).rev() {
					paths_cum_metrics[j as usize][k as usize] = paths_cum_metrics[j as usize][(k - 1) as usize];
				}
				paths_cum_metrics[j as usize][0] = cm_present[j as usize];
			}
			
			// Obtain estimate of input bit sequency from the oldest bit in
			// the traceback having the smallest (most likely) cumulative metric
			let mut min_metric_vec = vec![0; n_states as usize];
			
			let min_metric_array = Array2D::from_rows(&paths_cum_metrics);
			for (j, elem_j) in min_metric_array.column_iter(0).enumerate() {
				min_metric_vec[j as usize] = *elem_j as i32;
			}
			let min_metric = min_metric_vec.iter().min().unwrap();
			
			// Get first occurrence of min metric
			let mut min_idx = 0;
			for j in 0..n_states {
			
				if min_metric_vec[j as usize] == *min_metric {
					min_idx = j;
					break;
				}	
			}
			
			// Output data
			if i >= (symbol_l * depth - symbol_l) {
				output.push(paths_traceback_bits[min_idx as usize][(depth - 1) as usize]);
			}
		}
		
		return (output, paths_cum_metrics, paths_traceback_states, paths_traceback_bits)
		
	}
	
	/// A method which performs Viterbi decoding of noisy bit stream,
    /// taking as input soft bit values centered on +/-1 and returning 
    /// hard decision 0/1 bits.
	///
	/// Benjamin Roepken and Mark Wickert February 2020
    ///
    /// Parameters
    /// ----------
    /// x: Received noisy bit values centered on +/-1 at one sample per bit
    /// metric_type: 
    ///    'hard' - Hard decision metric. Expects binary or 0/1 input values.
    ///    'nquant' - unquantized soft decision decoding. Expects +/-1
    ///        input values.
    ///    'soft' - soft decision decoding.
    /// quant_level: The quantization level for soft decoding. Expected 
    /// input values between 0 and 2^quant_level-1. 0 represents the most 
    /// confident 0 and 2^quant_level-1 represents the most confident 1. 
    /// Only used for 'soft' metric type.
	///
    /// Returns
    /// -------
    /// y: Decoded 0/1 bit stream
	///
	#[pyfn(m, "viterbi_decoder")]
	fn viterbi_decoder_py(_py: Python, input: &PyArrayDyn<f64>, metric_type: String, quant_level: usize, g: &PyArrayDyn<f64>, depth: usize) -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {		
	
		// Viterbi Decode a signal with 1/2 or 1/3 rate encoding
		let input = input.as_array();
		let g = g.as_array();

		// Pass parameters to viterbi decoder
		let (output, paths_cum_metrics, paths_traceback_states, paths_traceback_bits) = 
			viterbi_decoder(input, metric_type, quant_level, g, depth);
		
		//return output_out
		return (output, paths_cum_metrics, paths_traceback_states, paths_traceback_bits)
	}

    Ok(())
}




