// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <string.h>

#include "quantized_face_detection_fast.h"
#include "quantized_utils.h"
#include "quantized_fastgrnn.h"
#include "quantized_rnnpool.h"
#include "quantized_mbconv.h"

#include "q_scut_head_b_face3_model/conv2D.h"
#include "q_scut_head_b_face3_model/rnn1.h"
#include "q_scut_head_b_face3_model/rnn2.h"
#include "q_scut_head_b_face3_model/mbconv1.h"
#include "q_scut_head_b_face3_model/detection1.h"
#include "q_scut_head_b_face3_model/mbconv2.h"
#include "q_scut_head_b_face3_model/detection2.h"
#include "q_scut_head_b_face3_model/mbconv3.h"
#include "q_scut_head_b_face3_model/mbconv4.h"
#include "q_scut_head_b_face3_model/detection3.h"

void q_face_detection_fast(char* const mem_buf) {
  // Conv2D Sub-Pipeline
  q7xq15_q7_convolution((Q7_T*)mem_buf, CBR1F, (Q7_T*)(mem_buf + 76800),
    CONV2D_N, CBR1F_H, CBR1F_W, CBR1F_CIN, CBR1F_HF, CBR1F_WF, CBR1F_CF,
    CONV2D_COUT, CONV2D_HOUT, CONV2D_WOUT, CBR1F_G, CBR1F_HPADL, CBR1F_HPADR,
    CBR1F_WPADL, CBR1F_WPADR, CBR1F_HSTRIDE, CBR1F_WSTRIDE, CBR1F_HDILATION,
    CBR1F_WDILATION, CBR1F_Scinput, CBR1F_Scoutput, CBR1F_Demote);

  q7xq15_q7_t_add_vec((Q7_T*)(mem_buf + 76800), CBR1B, CONV2D_N, CONV2D_HOUT,
    CONV2D_WOUT, CONV2D_COUT, (Q7_T*)mem_buf, CBR1B_Scten, CBR1B_Scvec,
    CBR1B_Scret);

  q7xq15_q7_convolution((Q7_T*)mem_buf, CBR1W, (Q7_T*)mem_buf, CONV2D_N,
    CONV2D_HOUT, CONV2D_WOUT, CONV2D_COUT, CBR1W_HF, CBR1W_WF, CBR1W_CF,
    CBR1W_COUT, CONV2D_HOUT, CONV2D_WOUT, CBR1W_G, CBR1W_HPADL, CBR1W_HPADR,
    CBR1W_WPADL, CBR1W_WPADR, CBR1W_HSTRIDE, CBR1W_WSTRIDE, CBR1W_HDILATION,
    CBR1W_WDILATION, CBR1W_Scinput, CBR1W_Scoutput, CBR1W_Demote);

  q7_t_relu((Q7_T*)mem_buf, CONV2D_N, CONV2D_HOUT, CONV2D_WOUT, CONV2D_COUT,
    (Q7_T*)(mem_buf + 76800), CONV2D_Limit, CONV2D_Div);

  Q7_T* mem_buf_offset_q7 = (Q7_T*)mem_buf;
  Q15_T* mem_buf_offset_q15 = (Q15_T*)mem_buf;

  // RNNPool Sub-Pipeline
  memset(mem_buf, 0, sizeof(Q7_T) * 19200);
  memset((mem_buf + 19200), 0, sizeof(Q15_T));
  memset((mem_buf + 19202), 0, sizeof(Q15_T));

  for (ITER_T patchX = 0; patchX < 14; patchX++) {
    for (ITER_T patchY = 0; patchY < 19; patchY++) {
      q7xq15_q15_rnnpool_block((Q7_T*)(mem_buf + 76800 + ((5120 * patchX) + (32 * patchY))),
        INPUT_CHANNELS, PATCH_DIM, CONV2D_WOUT, q7xq15_q15_fastgrnn, HIDDEN_DIM1,
        (const void*)(&RNN1_PARAMS), (void*)(&RNN1_BUFFERS),
        (const void*)(&RNN1_SCALES), q15_fastgrnn, HIDDEN_DIM2,
        (const void*)(&RNN2_PARAMS), (void*)(&RNN2_BUFFERS),
        (const void*)(&RNN2_SCALES), (Q15_T*)(mem_buf + 19204),
        (Q15_T*)(mem_buf + 21252), ShR1, ShL1, ShR2, ShL2);

      for (ITER_T i = 0; i < 64; i++) {
        mem_buf_offset_q7[patchX * 1280 + patchY * 64 + i] = (Q7_T)(mem_buf_offset_q15[9602 + i]);
      }
    }
  }

  memcpy(&mem_buf_offset_q7[14 * 1280], &mem_buf_offset_q7[13 * 1280],
         19 * 64 * sizeof(Q7_T));
  for (ITER_T i = 0; i < 15; i++) {
    memcpy(&mem_buf_offset_q7[19 * 64 + i * 1280],
           &mem_buf_offset_q7[18 * 64 + i * 1280], 64 * sizeof(Q7_T));
  }

  // MBConv Sub-Pipeline
  // MBConv Layer 1
  q7xq15_q15_mbconv_block((Q7_T*)mem_buf, L1_F1, L1_W1, L1_B1, L1_F2, L1_W2,
    L1_B2, L1_F3, L1_W3, L1_B3, (Q15_T*)(mem_buf + 19200),
    (Q15_T*)(mem_buf + 38400), (Q15_T*)(mem_buf + 54272), L1_N, L1_H, L1_W,
    L1_CIN, L1_CTEMP, L1_HF, L1_WF, L1_COUT, L1_HOUT, L1_WOUT, L1_HPADL,
    L1_HPADR, L1_WPADL, L1_WPADR, L1_HSTRIDE, L1_WSTRIDE, L1_Limit1, L1_Limit2,
    L1_ShRU1, L1_ShRX1, L1_ShRU2, L1_ShRX2, L1_ShRU3, L1_ShRW3, L1_ShLU1,
    L1_ShLX1, L1_ShLU2, L1_ShLX2, L1_ShLU3, L1_ShLW3);

  // Detection Layer 1 Sub-Pipeline
  q15_t_l2_norm((Q15_T*)(mem_buf + 19200), L1_N, L1_HOUT, L1_WOUT, L1_COUT,
    (Q15_T*)(mem_buf + 38400), D1_ScaleIn, D1_ScaleOut);

  q15_convolution((Q15_T*)(mem_buf + 38400), D1NW, (Q15_T*)(mem_buf + 38400),
    L1_N, L1_HOUT, L1_WOUT, L1_COUT, D1NW_HF, D1NW_WF, D1NW_CF, D1NW_COUT,
    L1_HOUT, L1_WOUT, D1NW_G, D1NW_HPADL, D1NW_HPADR, D1NW_WPADL, D1NW_WPADR,
    D1NW_HSTRIDE, D1NW_WSTRIDE, D1NW_HDILATION, D1NW_WDILATION, D1NW_Scinput,
    D1NW_Scoutput, D1NW_Demote);

  q15_convolution((Q15_T*)(mem_buf + 38400), D1CW, (Q15_T*)mem_buf, L1_N,
    L1_HOUT, L1_WOUT, D1NW_COUT * D1NW_G, D1CW_HF, D1CW_WF, D1CW_CF, D1CW_COUT,
    L1_HOUT, L1_WOUT, D1CW_G, D1CW_HPADL, D1CW_HPADR, D1CW_WPADL, D1CW_WPADR,
    D1CW_HSTRIDE, D1CW_WSTRIDE, D1CW_HDILATION, D1CW_WDILATION, D1CW_Scinput,
    D1CW_Scoutput, D1CW_Demote);

  q15_t_add_vec((Q15_T*)mem_buf, D1CB, L1_N, L1_HOUT, L1_WOUT,
    D1CW_COUT, (Q15_T*)mem_buf, D1CB_Scten, D1CB_Scvec, D1CB_Scret);

  q15_convolution((Q15_T*)(mem_buf + 38400), D1LW, (Q15_T*)(mem_buf + 2400),
    L1_N, L1_HOUT, L1_WOUT, D1NW_COUT * D1NW_G, D1LW_HF, D1LW_WF, D1LW_CF,
    D1LW_COUT, L1_HOUT, L1_WOUT, D1LW_G, D1LW_HPADL, D1LW_HPADR, D1LW_WPADL,
    D1LW_WPADR, D1LW_HSTRIDE, D1LW_WSTRIDE, D1LW_HDILATION, D1LW_WDILATION,
    D1LW_Scinput, D1LW_Scoutput, D1LW_Demote);

  q15_t_add_vec((Q15_T*)(mem_buf + 2400), D1LB, L1_N, L1_HOUT, L1_WOUT,
    D1LW_COUT, (Q15_T*)(mem_buf + 2400), D1LB_Scten, D1LB_Scvec, D1LB_Scret);

  memset((mem_buf_offset_q15 + 2400), 0, sizeof(Q15_T) * 600);
  memset(mem_buf_offset_q15 + 3000, 0, sizeof(Q15_T) * 1);
  memset((mem_buf_offset_q15 + 3001), 0, sizeof(Q15_T) * 1);

  for (ITER_T i = 0; i < 15; i++) {
    for (ITER_T j = 0; j < 20; j++) {
      for (ITER_T k = 0; k < 3; k++) {
        mem_buf_offset_q15[3005 + k] = mem_buf_offset_q15[(i * 80 + j * 4 + k)];
      }

      ITER_T index;
      q15_v_argmax(&mem_buf_offset_q15[3005], 3, &index);

      mem_buf_offset_q15[2400 + (i * 40 + j * 2)] = mem_buf_offset_q15[(i * 80 + j * 4 + index)];
      mem_buf_offset_q15[2400 + (i * 40 + j * 2 + 1)] = mem_buf_offset_q15[(i * 80 + j * 4 + 3)];
    }
  }

  // MBConv Layer 2
  q15_mbconv_block((Q15_T*)(mem_buf + 19200), L2_F1, L2_W1, L2_B1, L2_F2,
    L2_W2, L2_B2, L2_F3, L2_W3, L2_B3, (Q15_T*)(mem_buf + 38400),
    (Q15_T*)(mem_buf + 6000), (Q15_T*)(mem_buf + 256), L2_N, L2_H, L2_W,
    L2_CIN, L2_CTEMP, L2_HF, L2_WF, L2_COUT, L2_HOUT, L2_WOUT, L2_HPADL,
    L2_HPADR, L2_WPADL, L2_WPADR, L2_HSTRIDE, L2_WSTRIDE, L2_Limit1, L2_Limit2,
    L2_ShRU1, L2_ShRX1, L2_ShRU2, L2_ShRX2, L2_ShRU3, L2_ShRW3, L2_ShLU1,
    L2_ShLX1, L2_ShLU2, L2_ShLX2, L2_ShLU3, L2_ShLW3);

  // Detection Layer 2 Sub-Pipeline
  q15_t_l2_norm((Q15_T*)(mem_buf + 38400), L2_N, L2_HOUT, L2_WOUT, L2_COUT,
    (Q15_T*)(mem_buf + 96000), D2_ScaleIn, D2_ScaleOut);

  q15_convolution((Q15_T*)(mem_buf + 96000), D2NW, (Q15_T*)(mem_buf + 96000),
    L2_N, L2_HOUT, L2_WOUT, L2_COUT, D2NW_HF, D2NW_WF, D2NW_CF, D2NW_COUT,
    L2_HOUT, L2_WOUT, D2NW_G, D2NW_HPADL, D2NW_HPADR, D2NW_WPADL, D2NW_WPADR,
    D2NW_HSTRIDE, D2NW_WSTRIDE, D2NW_HDILATION, D2NW_WDILATION, D2NW_Scinput,
    D2NW_Scoutput, D2NW_Demote);

  q15_convolution((Q15_T*)(mem_buf + 96000), D2CW, (Q15_T*)mem_buf, L2_N,
    L2_HOUT, L2_WOUT, D2NW_COUT * D2NW_G, D2CW_HF, D2CW_WF, D2CW_CF, D2CW_COUT,
    L2_HOUT, L2_WOUT, D2CW_G, D2CW_HPADL, D2CW_HPADR, D2CW_WPADL, D2CW_WPADR,
    D2CW_HSTRIDE, D2CW_WSTRIDE, D2CW_HDILATION, D2CW_WDILATION, D2CW_Scinput,
    D2CW_Scoutput, D2CW_Demote);

  q15_t_add_vec((Q15_T*)mem_buf, D2CB, L2_N, L2_HOUT, L2_WOUT,
    D2CW_COUT, (Q15_T*)mem_buf, D2CB_Scten, D2CB_Scvec, D2CB_Scret);

  q15_convolution((Q15_T*)(mem_buf + 96000), D2LW, (Q15_T*)(mem_buf + 9456),
    L2_N, L2_HOUT, L2_WOUT, D2NW_COUT * D2NW_G, D2LW_HF, D2LW_WF, D2LW_CF,
    D2LW_COUT, L2_HOUT, L2_WOUT, D2LW_G, D2LW_HPADL, D2LW_HPADR, D2LW_WPADL,
    D2LW_WPADR, D2LW_HSTRIDE, D2LW_WSTRIDE, D2LW_HDILATION, D2LW_WDILATION,
    D2LW_Scinput, D2LW_Scoutput, D2LW_Demote);

  q15_t_add_vec((Q15_T*)(mem_buf + 9456), D2LB, L2_N, L2_HOUT, L2_WOUT,
    D2LW_COUT, (Q15_T*)(mem_buf + 6000), D2LB_Scten, D2LB_Scvec, D2LB_Scret);

  // MBConv Layer 3
  q15xq7_q15_mbconv_block((Q15_T*)(mem_buf + 38400), L3_F1, L3_W1, L3_B1,
    L3_F2, L3_W2, L3_B2, L3_F3, L3_W3, L3_B3, (Q15_T*)(mem_buf + 96000),
    (Q15_T*)(mem_buf + 8400), (Q15_T*)(mem_buf + 1968), L3_N, L3_H, L3_W,
    L3_CIN, L3_CTEMP, L3_HF, L3_WF, L3_COUT, L3_HOUT, L3_WOUT, L3_HPADL,
    L3_HPADR, L3_WPADL, L3_WPADR, L3_HSTRIDE, L3_WSTRIDE, L3_Limit1, L3_Limit2,
    L3_ShRU1, L3_ShRX1, L3_ShRU2, L3_ShRX2, L3_ShRU3, L3_ShRW3, L3_ShLU1,
    L3_ShLX1, L3_ShLU2, L3_ShLX2, L3_ShLU3, L3_ShLW3);

  // MBConv2 + MBConv3
  q15_t_add((Q15_T*)(mem_buf + 38400), (Q15_T*)(mem_buf + 96000), L3_N, L3_HOUT,
    L3_WOUT, L3_COUT, (Q15_T*)(mem_buf + 8400), L3_Scten1, L3_Scten2, L3_Scret);

  // MBConv Layer 4
  q15xq7_q15_mbconv_block((Q15_T*)(mem_buf + 8400), L4_F1, L4_W1, L4_B1, L4_F2,
    L4_W2, L4_B2, L4_F3, L4_W3, L4_B3, (Q15_T*)(mem_buf + 66000),
    (Q15_T*)(mem_buf + 142800), (Q15_T*)(mem_buf + 1968), L4_N, L4_H, L4_W,
    L4_CIN, L4_CTEMP, L4_HF, L4_WF, L4_COUT, L4_HOUT, L4_WOUT, L4_HPADL,
    L4_HPADR, L4_WPADL, L4_WPADR, L4_HSTRIDE, L4_WSTRIDE, L4_Limit1, L4_Limit2,
    L4_ShRU1, L4_ShRX1, L4_ShRU2, L4_ShRX2, L4_ShRU3, L4_ShRW3, L4_ShLU1,
    L4_ShLX1, L4_ShLU2, L4_ShLX2, L4_ShLU3, L4_ShLW3);

  // Detection Layer 3 Sub-Pipeline
  q15_t_l2_norm((Q15_T*)(mem_buf + 66000), L4_N, L4_HOUT, L4_WOUT, L4_COUT,
    (Q15_T*)(mem_buf + 8400), D3_ScaleIn, D3_ScaleOut);

  q15_convolution((Q15_T*)(mem_buf + 8400), D3NW, (Q15_T*)(mem_buf + 8400),
    L4_N, L4_HOUT, L4_WOUT, L4_COUT, D3NW_HF, D3NW_WF, D3NW_CF, D3NW_COUT,
    L4_HOUT, L4_WOUT, D3NW_G, D3NW_HPADL, D3NW_HPADR, D3NW_WPADL, D3NW_WPADR,
    D3NW_HSTRIDE, D3NW_WSTRIDE, D3NW_HDILATION, D3NW_WDILATION, D3NW_Scinput,
    D3NW_Scoutput, D3NW_Demote);

  q15_convolution((Q15_T*)(mem_buf + 8400), D3CW, (Q15_T*)(mem_buf + 89808),
    L4_N, L4_HOUT, L4_WOUT, D3NW_COUT * D3NW_G, D3CW_HF, D3CW_WF, D3CW_CF,
    D3CW_COUT, L4_HOUT, L4_WOUT, D3CW_G, D3CW_HPADL, D3CW_HPADR, D3CW_WPADL,
    D3CW_WPADR, D3CW_HSTRIDE, D3CW_WSTRIDE, D3CW_HDILATION, D3CW_WDILATION,
    D3CW_Scinput, D3CW_Scoutput, D3CW_Demote);

  q15_t_add_vec((Q15_T*)(mem_buf + 89808), D3CB, L4_N, L4_HOUT, L4_WOUT,
    D3CW_COUT, (Q15_T*)(mem_buf + 85200), D3CB_Scten, D3CB_Scvec, D3CB_Scret);

  q15_convolution((Q15_T*)(mem_buf + 8400), D3LW, (Q15_T*)(mem_buf + 91008),
    L4_N, L4_HOUT, L4_WOUT, D3NW_COUT * D3NW_G, D3LW_HF, D3LW_WF, D3LW_CF,
    D3LW_COUT, L4_HOUT, L4_WOUT, D3LW_G, D3LW_HPADL, D3LW_HPADR, D3LW_WPADL,
    D3LW_WPADR, D3LW_HSTRIDE, D3LW_WSTRIDE, D3LW_HDILATION, D3LW_WDILATION,
    D3LW_Scinput, D3LW_Scoutput, D3LW_Demote);

  q15_t_add_vec((Q15_T*)(mem_buf + 91008), D3LB, L4_N, L4_HOUT, L4_WOUT,
    D3LW_COUT, (Q15_T*)(mem_buf + 8400), D3LB_Scten, D3LB_Scvec, D3LB_Scret);

  // Re-ordering the outputs
  memset((mem_buf_offset_q15 + 5400), 0, sizeof(Q15_T) * 5400);
  memcpy(&mem_buf_offset_q15[5400], &mem_buf_offset_q15[2400], 600 * sizeof(Q15_T));

  for (ITER_T i = 0; i < 600; i++) {
    mem_buf_offset_q15[5400 + (i + 600)] = (mem_buf_offset_q15[i] / 2);
  }

  memcpy(&mem_buf_offset_q15[5400 + 1200], &mem_buf_offset_q15[42600], 600 * sizeof(Q15_T));
  memcpy(&mem_buf_offset_q15[5400 + 1800], &mem_buf_offset_q15[1200], 1200 * sizeof(Q15_T));
  memcpy(&mem_buf_offset_q15[5400 + 3000], &mem_buf_offset_q15[3000], 1200 * sizeof(Q15_T));

  for (ITER_T i = 0; i < 1200; i++) {
    mem_buf_offset_q15[5400 + (i + 4200)] = (mem_buf_offset_q15[4200 + i] / 2);
  }
}
