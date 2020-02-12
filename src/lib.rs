extern crate pyo3;
extern crate special_fun;
extern crate factorial;

use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
//use numpy::{IntoPyArray, PyArrayDyn};
use numpy::*;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
//use pyo3::types::PyType;
use special_fun::cephes_single::erfc;
use factorial::Factorial;

//#[pyfunction]
//fn bm_calc(ref_code_bits: &str, rec_code_bits: &arr, metric_type: &str, quant_level: i32) -> usize {
//}

#[pyfunction]
	fn binary(num: i32) -> i32 {
		// Format an integer to binary without the leading '0b'

		let mut length = 8;
		length = length + num;
        //return format(num, '0{}b'.format(length))
		return length;
}


#[pymodule]
fn rs_fec_conv(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    //m.add_wrapped(wrap_pyfunction!(char_count))?;
	m.add_wrapped(wrap_pyfunction!(binary))?;
	//m.add_wrapped(wrap_pyfunction!(conv_Pb_bound))?;
	//m.add_wrapped(wrap_pyfunction!(axpy2))?;
	//add_wrapped(wrap_pyfunction!(count_line))?;
	//m.add_class::<FecConv>()?;
	//m.add_class::<MyClass>()?;
	
	fn axpy(a: f64, x: ArrayViewD<f64>, y: ArrayViewD<f64>) -> ArrayD<f64> {
        a * &x + &y
    }
	
	 // wrapper of `axpy`
	#[pyfn(m, "axpy")]
	fn axpy_py(
		py: Python,
		a: f64,
		x: &PyArrayDyn<f64>,
		y: &PyArrayDyn<f64>,
	) -> Py<PyArrayDyn<f64>> {
		let x = x.as_array();
		let y = y.as_array();
		axpy(a, x, y).into_pyarray(py).to_owned()
	}
	
	fn mult(a: f64, mut x: ArrayViewMutD<'_, f64>) {

        x *= a;

    }
	
	 // wrapper of `mult`

    #[pyfn(m, "mult")]

    fn mult_py(_py: Python<'_>, a: f64, x: &PyArrayDyn<f64>) -> PyResult<()> {

        let x = x.as_array_mut();

        mult(a, x);

        Ok(())

    }
	
	fn axpy2(a: f64, x: ArrayViewD<f64>) -> ArrayD<f64> {
        a * &x
    }
	
	// wrapper of `axpy`
	#[pyfn(m, "axpy2")]
	fn axpy2_py(
		py: Python,
		a: f64,
		x: &PyArrayDyn<f64>,
	) -> Py<PyArrayDyn<f64>> {
		let x = x.as_array();
		axpy2(a, x).into_pyarray(py).to_owned()
	}
	
	/// Gaussian Q-function
	///
	/// Benjamin Roepken February 2020
	///
	/// Parameters
    /// ----------
	/// x: SNR value (f64)
	///
	pub fn q_fctn(x: f64) -> f64 {
		// Gaussian Q-function	
		let y = x / 2.0f64.sqrt();
		(1.0 / 2.0 * erfc(y as f32)).into()
	}
	
	
	/// pk = hard_pk(k,r,snr)
    ///
    /// Calculates Pk as found in Ziemer & Peterson eq. 7-12, p.505
    ///
    /// Benjamin Roepken and Mark Wickert and Andrew Smit February 2020
    ///
	/// Parameters
    /// ----------
	/// k: Free distance of code (usize)
    /// r: Code rate (float)
	/// snr: Signal-Noise Ratio (f64)
	/// m: M-ary (usize)
	///
	fn hard_pk(k: usize, r: f64, snr: f64, m: usize) -> f64 {
		// Calculate Pk as a hard decision
		
		// calculate p
		let mut p: f64 = 0.0;
		if m == 2 {
			p = q_fctn((2.0 * r * snr).sqrt());
		}
		else {
			let m64: f64 = m as f64;
			p = 4.0 / m64.log2() * (1.0 - 1.0 / m64.sqrt()) * 
				q_fctn((3.0 * r * m64.log2() / (m64 - 1.0) * snr).sqrt());
		}
		
		// calculate Pk
		let mut pk: f64 = 0.0;
		let k64: f64 = k as f64;
		let k_fac: f64 = k.factorial() as f64;
		if k % 2 == 0 { 
			for e in (k / 2 + 1)..(k + 1) {
				let e64: f64 = e as f64; 
				let e_fac: f64 = e.factorial() as f64;
				let ke_fac: f64 = (k - e).factorial() as f64;
				pk += k_fac / (e_fac * ke_fac) * 
					p.powf(e64) * (1.0 - p).powf(k64 - e64);
			}
			let k2_fac: f64 = (k / 2).factorial() as f64;
			let kk2_fac: f64 = (k - k / 2).factorial() as f64;
			pk += 1.0 / 2.0 * k_fac / (k2_fac * kk2_fac) * 
				p.powf(k64 / 2.0) * (1.0 - p).powf(k64 / 2.0);
		}
		else if k % 2 == 1 {
			for e in ((k + 1) / 2)..(k + 1) {
				let e64: f64 = e as f64; 
				let e_fac: f64 = e.factorial() as f64;
				let ke_fac: f64 = (k - e).factorial() as f64;
				pk += k_fac / (e_fac * ke_fac) * p.powf(e64) * (1.0 - p).powf(k64 - e64);
			}
		}
		else {
			pk = 0.0
		}
		
		// return Pk
		pk
	}
	
	/// pk = soft_pk(k,r,snr)
    ///
    /// Calculates Pk as found in Ziemer & Peterson eq. 7-13, p.505
    ///
    /// Benjamin Roepken and Mark Wickert February 2020
    ///
	/// Parameters
    /// ----------
	/// k: Free distance of code (usize)
    /// r: Code rate (float)
	/// snr: Signal-Noise Ratio (f64)
	/// m: M-ary (usize)
	///
	fn soft_pk(k: usize, r: f64, snr: f64, m: usize) -> f64 {
		// Calculate Pk as a soft decision
		
		// calculate Pk
		let mut pk: f64 = 0.0;
		let k64: f64 = k as f64;
		
		// 2-ary
		if m == 2 {
			pk = q_fctn((2.0 * k64 * r * snr).sqrt());
		}
		// M-ary
		else {
			let m64: f64 = m as f64;
			pk = 4.0 / m64.log2() * (1.0 - 1.0 / m64.sqrt()) * 
				q_fctn((3.0 * k64 * r * m64.log2() / (m64 - 1.0) * snr).sqrt());
		}
		
		// return Pk
		pk
	}
	
	/// Coded bit error probabilty
	///
    /// Convolution coding bit error probability upper bound
    /// according to Ziemer & Peterson 7-16, p. 507
    ///
    /// Benjamin Roepken and Mark Wickert February 2020
    ///
    /// Parameters
    /// ----------
    /// r: Code rate (f64)
    /// dfree: Free distance of the code (usize)
    /// ck: Weight coefficient (np.array([float, float, ...]))
    /// snrdb: Signal to noise ratio in dB (np.array([float, float, ...]))
    /// hard_soft: 0 hard, 1 soft, 2 uncoded (usize)
    /// m: M-ary (usize)
	///
	fn conv_pb_bound(r: f64, dfree: usize, ck: ArrayViewD<f64>, 
		snrdb: ArrayViewD<f64>, hard_soft: usize, m: usize) -> ArrayD<f64> {
		
		// get exponent for each element
		let mut snr = &snrdb / 10.0;
		
		// 10 ^ (SNRdB / 10)
		for (_i, elem_i) in snr.iter_mut().enumerate() {
			*elem_i =10.0f64.powf(*elem_i) as f64
		}
		
		// initialize coded bit error probability
		let mut pb = &snrdb * 0.0;
		
		for (i, elem_i) in pb.iter_mut().enumerate() {
			for j in dfree..(dfree + ck.len()) {

				// Evaluate hard decision bound
				if hard_soft == 0 {
					*elem_i += ck[j - dfree] * hard_pk(j, r, snr[i], m);
				}
				
				// Evaluate soft decision bound
				else if hard_soft == 1 {
					*elem_i += ck[j - dfree] * soft_pk(j, r, snr[i], m);
				}
				
				// Compute Uncoded Pe
				else {
					// 2-ary
					if m == 2 {
						*elem_i = q_fctn((2.0 * snr[i]).sqrt());
					}
					// M-ary
					else {
						let m64: f64 = m as f64;
						*elem_i = 4.0 / m64.log2() * (1.0 - 1.0 / m64.sqrt()) * 
							q_fctn((3.0 * r * m64.log2() / (m64 - 1.0) * snr[i]).sqrt());
					}
				}
			}
		}
		
		// return coded bit error probability
		pb
	}
	
	
	/// Coded bit error probabilty
	///
    /// Convolution coding bit error probability upper bound
    /// according to Ziemer & Peterson 7-16, p. 507
    ///
    /// Benjamin Roepken and Mark Wickert February 2020
    ///
    /// Parameters
    /// ----------
    /// R: Code rate (float)
    /// dfree: Free distance of the code (int)
    /// Ck: Weight coefficient (np.array([float, float, ...]))
    /// SNRdB: Signal to noise ratio in dB (np.array([float, float, ...]))
    /// hard_soft: 0 hard, 1 soft, 2 uncoded (int)
    /// M: M-ary (int)
	///
	/// Examples
	/// --------
	/// import numpy as np
	/// import matplotlib.pyplot as plt
	/// import rs_fec_conv
	/// SNRdB = np.arange(0.,12.,.1)
	/// Pb_uc_rs = rs_fec_conv.conv_Pb_bound(1./3,7,np.array([4., 12., 20., 72., 225.]),SNRdB,2,2)
	/// Pb_s_third_3_hard_rs = rs_fec_conv.conv_Pb_bound(1./3,8,np.array([3., 0., 15., 0., 58., 0., 201., 0.]),SNRdB,0,2)
	/// Pb_s_third_5_hard_rs = rs_fec_conv.conv_Pb_bound(1./3,12,np.array([12., 0., 12., 0., 56., 0., 320., 0.]),SNRdB,0,2)
	/// plt.semilogy(SNRdB,Pb_uc_rs)
    /// plt.semilogy(SNRdB,Pb_s_third_3_hard_rs)
    /// plt.semilogy(SNRdB,Pb_s_third_5_hard_rs)
    /// plt.axis([2,12,1e-7,1e0])
	/// plt.xlabel(r'$E_b/N_0$ (dB)')
	/// plt.ylabel(r'Symbol Error Probability')
	/// plt.legend(('Uncoded BPSK','R=1/3, K=3, Hard','R=1/3, K=5, Hard',),loc='upper right')
	/// plt.grid();
	/// plt.show()
	///
	#[pyfn(m, "conv_Pb_bound")]
	fn conv_pb_bound_py(py: Python, r: f64, dfree: usize, ck: &PyArrayDyn<f64>, 
		snrdb: &PyArrayDyn<f64>, hard_soft: usize, m: usize) -> Py<PyArrayDyn<f64>> {
		
		// Wrapper for conv_Pb_bound
		
		// Convert SNR in dB to SNR in linear units
		let ck = ck.as_array();
		let snrdb = snrdb.as_array();
		let pb = conv_pb_bound(r, dfree, ck, snrdb, hard_soft, m).into_pyarray(py).to_owned();
		
		// return coded bit error probability
		pb
	}
	
    Ok(())
}