.. _getting_started:


***************
Getting started
***************

.. _Tutorial:

Tutorial
========

The function conv_Pb_bound can be implemented by:::

  import numpy as np
  import matplotlib.pyplot as plt
  import rs_fec_conv
  SNRdB = np.arange(0.,12.,.1)
  Pb_uc_rs = rs_fec_conv.conv_Pb_bound(1./3,7,np.array([4., 12., 20., 72., 225.]),SNRdB,2,2)
  Pb_s_third_3_hard_rs = rs_fec_conv.conv_Pb_bound(1./3,8,np.array([3., 0., 15., 0., 58., 0., 201., 0.]),SNRdB,0,2)
  Pb_s_third_5_hard_rs = rs_fec_conv.conv_Pb_bound(1./3,12,np.array([12., 0., 12., 0., 56., 0., 320., 0.]),SNRdB,0,2)
  plt.semilogy(SNRdB,Pb_uc_rs)
  plt.semilogy(SNRdB,Pb_s_third_3_hard_rs)
  plt.semilogy(SNRdB,Pb_s_third_5_hard_rs)
  plt.axis([2,12,1e-7,1e0])
  plt.xlabel(r'$E_b/N_0$ (dB)')
  plt.ylabel(r'Symbol Error Probability')
  plt.legend(('Uncoded BPSK','R=1/3, K=3, Hard','R=1/3, K=5, Hard',),loc='upper right')
  plt.grid();
  plt.show()

