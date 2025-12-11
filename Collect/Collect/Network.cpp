#include "Network.h"
#include "Process.h"
#include "pch.h"
#include <cmath>

float *train_fcn(float *Samples, int numSample, float *targets, int *topology,
                 int layer_count, float **Weights, float **Biases,
                 float learning_rate, float momentum, float Min_Err,
                 int Max_epoch, int &epoch) {

  float total_err;
  float *temp = new float[Max_epoch];

  // Katman sayısı kadar çıkış, türev ve hata dizileri oluştur
  // layer_indices: 0=Input, 1=Hidden1 ... layer_count-1=Output
  // Hesaplamalar layer 1'den başlar (Input layer sadece veri tutar)

  float **net = new float *[layer_count];
  float **fnet = new float *[layer_count];
  float **f_der = new float *[layer_count];
  float **delta = new float *[layer_count];

  // Momentum için önceki değişimleri tutacak diziler
  float **prev_dW = new float *[layer_count - 1]; // Weights L-1 tane
  float **prev_dB = new float *[layer_count - 1]; // Biases L-1 tane

  // Her katman için dizileri ayır (Input katmanı [0] için de fnet ayıralım ki
  // döngü kolay olsun)
  for (int l = 0; l < layer_count; l++) {
    int node_count = topology[l];
    net[l] = new float[node_count];
    fnet[l] = new float[node_count];
    f_der[l] = new float[node_count];
    delta[l] = new float[node_count];
  }

  // Momentum matrislerini başlat (0 ile doldur)
  for (int l = 0; l < layer_count - 1; l++) {
    int src_nodes = topology[l];
    int dest_nodes = topology[l + 1];
    int weight_count = src_nodes * dest_nodes;

    prev_dW[l] = new float[weight_count];
    prev_dB[l] = new float[dest_nodes];

    for (int k = 0; k < weight_count; k++)
      prev_dW[l][k] = 0.0f;
    for (int k = 0; k < dest_nodes; k++)
      prev_dB[l][k] = 0.0f;
  }

  float *desired = new float[topology[layer_count - 1]]; // Sadece output
                                                         // katmanı boyutu kadar
  float *err = new float[topology[layer_count - 1]];

  epoch = 0;

  do {
    total_err = 0.0f;
    for (int step = 0; step < numSample; step++) {

      // --- FeedForward ---

      // Input Layer (Katman 0) verilerini yükle
      for (int i = 0; i < topology[0]; i++) {
        fnet[0][i] = Samples[step * topology[0] + i];
      }

      // Katmanlar arası ilerle (l: hedef katman indeksi, 1'den başlar)
      for (int l = 1; l < layer_count; l++) {
        int prev_nodes = topology[l - 1];
        int curr_nodes = topology[l];

        for (int j = 0; j < curr_nodes; j++) { // Hedef nöron
          net[l][j] = 0.0f;
          // Önceki katmandan gelenler
          for (int i = 0; i < prev_nodes; i++) {
            net[l][j] += Weights[l - 1][j * prev_nodes + i] * fnet[l - 1][i];
          }
          // Bias ekle
          net[l][j] += Biases[l - 1][j];

          // Aktivasyon (Tanh)
          float val = net[l][j];
          if (val > 20.0f)
            val = 20.0f;
          if (val < -20.0f)
            val = -20.0f;
          fnet[l][j] = (2.0f / (1.0f + exp(-val)) - 1.0f);
          // Türev
          f_der[l][j] = 0.5f * (1.0f - fnet[l][j] * fnet[l][j]);
        }
      }

      // --- BackPropagation ---

      int output_layer_idx = layer_count - 1;
      int output_nodes = topology[output_layer_idx];

      // 1. Output Katmanı Hatası ve Delta Hesabı
      for (int k = 0; k < output_nodes; k++) {
        // Desired output hazırla (+1 hedef sınıf, -1 diğerleri)
        if (targets[step] == k)
          desired[k] = 1.0f;
        else
          desired[k] = -1.0f;

        err[k] = desired[k] - fnet[output_layer_idx][k];
        delta[output_layer_idx][k] =
            learning_rate * err[k] * f_der[output_layer_idx][k];

        total_err += (0.5f * (err[k] * err[k]));
      }

      // 2. Gizli Katmanlar Delta Hesabı (Geriye doğru)
      // l: Şu an hesapladığımız gizli katman. (Output'un bir öncesinden
      // input'un sonrasına kadar)
      for (int l = layer_count - 2; l > 0; l--) {
        int curr_nodes = topology[l];
        int next_nodes =
            topology[l + 1]; // İlerideki katman (hata oradan geliyor)

        for (int j = 0; j < curr_nodes; j++) {
          float error_sum = 0.0f;
          for (int k = 0; k < next_nodes; k++) {
            // Weights[l] matrisi: l -> l+1 bağlantısı
            // Weights dizilimi: [next_node_idx * curr_node_count +
            // curr_node_idx] (dikkat!) Ancak bizde genelde [hidden_idx *
            // input_idx] yapısı var. Yukardaki FF'de: Weights[l-1][j *
            // prev_nodes + i] -> j: hedef(curr), i: kaynak(prev) Burada:
            // Weights[l][k * curr_nodes + j] -> k: hedef(next), j: kaynak(curr)
            error_sum += delta[l + 1][k] * Weights[l][k * curr_nodes + j];
          }
          delta[l][j] = learning_rate * error_sum * f_der[l][j];
        }
      }

      // 3. Ağırlık Güncelleme (Momentum Ile)
      // l: bağlantı indeksi (0: Input->L1, 1: L1->L2 ...)
      for (int l = 0; l < layer_count - 1; l++) {
        int src_nodes = topology[l];      // Kaynak katman nöron sayısı
        int dest_nodes = topology[l + 1]; // Hedef katman nöron sayısı

        for (int j = 0; j < dest_nodes; j++) {  // Hedef nöron
          for (int i = 0; i < src_nodes; i++) { // Kaynak nöron
            float current_change = delta[l + 1][j] * fnet[l][i];
            float update =
                current_change + (momentum * prev_dW[l][j * src_nodes + i]);

            Weights[l][j * src_nodes + i] += update;
            prev_dW[l][j * src_nodes + i] = update;
          }

          float current_bias_change = delta[l + 1][j];
          float update_bias = current_bias_change + (momentum * prev_dB[l][j]);

          Biases[l][j] += update_bias;
          prev_dB[l][j] = update_bias;
        }
      }

    } // Sample Loop

    total_err /= float(topology[layer_count - 1] * numSample);

    if (epoch < Max_epoch) {
      temp[epoch] = total_err;
      epoch++;
    }
  } while ((total_err > Min_Err) && (epoch < Max_epoch));

  // Bellek Temizliği
  for (int l = 0; l < layer_count; l++) {
    delete[] net[l];
    delete[] fnet[l];
    delete[] f_der[l];
    delete[] delta[l];
  }
  for (int l = 0; l < layer_count - 1; l++) {
    delete[] prev_dW[l];
    delete[] prev_dB[l];
  }
  delete[] prev_dW;
  delete[] prev_dB;

  delete[] net;
  delete[] fnet;
  delete[] f_der;
  delete[] delta;
  delete[] desired;
  delete[] err;

  return temp;
} // train_fcn

// Test Forward Function
int Test_Forward(float *input, float **Weights, float **Biases, int *topology,
                 int layer_count) {
  // Bellek Allocation (Sadece forward için gerekli)
  float **net = new float *[layer_count];
  float **fnet = new float *[layer_count];

  for (int l = 0; l < layer_count; l++) {
    net[l] = new float[topology[l]];
    fnet[l] = new float[topology[l]];
  }

  // Input Yükle
  for (int i = 0; i < topology[0]; i++) {
    fnet[0][i] = input[i];
  }

  // Forward Propagation
  for (int l = 1; l < layer_count; l++) {
    int prev_nodes = topology[l - 1];
    int curr_nodes = topology[l];

    for (int j = 0; j < curr_nodes; j++) {
      net[l][j] = 0.0f;
      for (int i = 0; i < prev_nodes; i++) {
        net[l][j] += Weights[l - 1][j * prev_nodes + i] * fnet[l - 1][i];
      }
      net[l][j] += Biases[l - 1][j];

      // Aktivasyon (Tanh)
      float val = net[l][j];
      if (val > 20.0f)
        val = 20.0f;
      if (val < -20.0f)
        val = -20.0f;
      fnet[l][j] = (2.0f / (1.0f + exp(-val)) - 1.0f);
    }
  }

  // Winner Takes All (Max output index)
  int output_layer = layer_count - 1;
  int max_idx = 0;
  float max_val = fnet[output_layer][0];

  for (int k = 1; k < topology[output_layer]; k++) {
    if (fnet[output_layer][k] > max_val) {
      max_val = fnet[output_layer][k];
      max_idx = k;
    }
  }

  // Cleanup
  for (int l = 0; l < layer_count; l++) {
    delete[] net[l];
    delete[] fnet[l];
  }
  delete[] net;
  delete[] fnet;

  return max_idx;
}

// MLP Regression Training
float *train_mlp_regression(float *Samples, int numSample, float *targets,
                            int *topology, int layer_count, float **Weights,
                            float **Biases, float learning_rate, float momentum,
                            float Min_Err, int Max_epoch, int &epoch) {

  float total_err;
  float *temp = new float[Max_epoch];

  // Allocation
  float **net = new float *[layer_count];
  float **fnet = new float *[layer_count];
  float **f_der = new float *[layer_count];
  float **delta = new float *[layer_count];

  // Momentum Arrays
  float **prev_dW = new float *[layer_count - 1];
  float **prev_dB = new float *[layer_count - 1];

  for (int l = 0; l < layer_count; l++) {
    int node_count = topology[l];
    net[l] = new float[node_count];
    fnet[l] = new float[node_count];
    f_der[l] = new float[node_count];
    delta[l] = new float[node_count];
  }

  for (int l = 0; l < layer_count - 1; l++) {
    int src_nodes = topology[l];
    int dest_nodes = topology[l + 1];
    int weight_count = src_nodes * dest_nodes;
    prev_dW[l] = new float[weight_count];
    prev_dB[l] = new float[dest_nodes];
    for (int k = 0; k < weight_count; k++)
      prev_dW[l][k] = 0.0f;
    for (int k = 0; k < dest_nodes; k++)
      prev_dB[l][k] = 0.0f;
  }

  // Regression'da tek output vardır ama genel yapı bozulmasın diye topology
  // kullanıyoruz
  float *err = new float[topology[layer_count - 1]];

  epoch = 0;
  do {
    total_err = 0.0f;
    for (int step = 0; step < numSample; step++) {

      // --- FeedForward ---
      // Input Layer
      for (int i = 0; i < topology[0]; i++) {
        fnet[0][i] = Samples[step * topology[0] + i];
      }

      // Hidden & Output Layers
      for (int l = 1; l < layer_count; l++) {
        int prev_nodes = topology[l - 1];
        int curr_nodes = topology[l];
        // bool is_output = (l == layer_count - 1); // Not used directly, but
        // logic is there

        for (int j = 0; j < curr_nodes; j++) {
          net[l][j] = 0.0f;
          for (int i = 0; i < prev_nodes; i++) {
            net[l][j] += Weights[l - 1][j * prev_nodes + i] * fnet[l - 1][i];
          }
          net[l][j] += Biases[l - 1][j];

          if (l == layer_count -
                       1) { // Output Layer: LINEAR Activation for Regression
            fnet[l][j] = net[l][j];
            f_der[l][j] = 1.0f;
          } else {
            // Hidden Layers: TANH Activation
            float val = net[l][j];
            if (val > 20.0f)
              val = 20.0f;
            if (val < -20.0f)
              val = -20.0f;
            fnet[l][j] = (2.0f / (1.0f + exp(-val)) - 1.0f);
            f_der[l][j] = 0.5f * (1.0f - fnet[l][j] * fnet[l][j]);
          }
        }
      }

      // --- BackPropagation ---
      int output_layer_idx = layer_count - 1;
      int output_nodes = topology[output_layer_idx];

      // 1. Output Delta
      for (int k = 0; k < output_nodes; k++) {
        // Regression Error: Target - Output
        // Note: targets dizisi float* type, step ile erişim regression için tek
        // boyutlu ise targets[step]
        float target_val = targets[step * output_nodes + k]; // Genelleme
        err[k] = target_val - fnet[output_layer_idx][k];

        delta[output_layer_idx][k] =
            learning_rate * err[k] * f_der[output_layer_idx][k];

        total_err += (0.5f * (err[k] * err[k]));
      }

      // 2. Hidden Deltas
      for (int l = layer_count - 2; l > 0; l--) {
        int curr_nodes = topology[l];
        int next_nodes = topology[l + 1];

        for (int j = 0; j < curr_nodes; j++) {
          float error_sum = 0.0f;
          for (int k = 0; k < next_nodes; k++) {
            error_sum += delta[l + 1][k] * Weights[l][k * curr_nodes + j];
          }
          delta[l][j] = learning_rate * error_sum * f_der[l][j];
        }
      }

      // 3. Weight Update with Momentum
      for (int l = 0; l < layer_count - 1; l++) {
        int src_nodes = topology[l];
        int dest_nodes = topology[l + 1];

        for (int j = 0; j < dest_nodes; j++) {
          for (int i = 0; i < src_nodes; i++) {
            float current_change = delta[l + 1][j] * fnet[l][i];
            float update =
                current_change + (momentum * prev_dW[l][j * src_nodes + i]);
            Weights[l][j * src_nodes + i] += update;
            prev_dW[l][j * src_nodes + i] = update;
          }
          float current_bias_change = delta[l + 1][j];
          float update_bias = current_bias_change + (momentum * prev_dB[l][j]);
          Biases[l][j] += update_bias;
          prev_dB[l][j] = update_bias;
        }
      }

    } // Sample Loop

    total_err /= float(numSample); // MSE

    if (epoch < Max_epoch) {
      temp[epoch] = total_err;
      epoch++;
    }

  } while ((total_err > Min_Err) && (epoch < Max_epoch));

  // Cleanup
  for (int l = 0; l < layer_count; l++) {
    delete[] net[l];
    delete[] fnet[l];
    delete[] f_der[l];
    delete[] delta[l];
  }
  for (int l = 0; l < layer_count - 1; l++) {
    delete[] prev_dW[l];
    delete[] prev_dB[l];
  }
  delete[] prev_dW;
  delete[] prev_dB;

  delete[] net;
  delete[] fnet;
  delete[] f_der;
  delete[] delta;
  delete[] err;

  return temp;
}

float Evaluate_Regression_Point(float input_val, float **Weights,
                                float **Biases, int *topology,
                                int layer_count) {
  // Bellek Allocation (Sadece forward için gerekli)
  float **net = new float *[layer_count];
  float **fnet = new float *[layer_count];

  for (int l = 0; l < layer_count; l++) {
    net[l] = new float[topology[l]];
    fnet[l] = new float[topology[l]];
  }

  // Input Yükle (Regression için tek input var)
  fnet[0][0] = input_val;

  // Forward Propagation
  for (int l = 1; l < layer_count; l++) {
    int prev_nodes = topology[l - 1];
    int curr_nodes = topology[l];
    bool is_output = (l == layer_count - 1);

    for (int j = 0; j < curr_nodes; j++) {
      net[l][j] = 0.0f;
      for (int i = 0; i < prev_nodes; i++) {
        net[l][j] += Weights[l - 1][j * prev_nodes + i] * fnet[l - 1][i];
      }
      net[l][j] += Biases[l - 1][j];

      if (is_output) {
        // Output Layer: LINEAR Activation for Regression
        fnet[l][j] = net[l][j];
      } else {
        // Hidden Layers: TANH Activation
        float val = net[l][j];
        if (val > 20.0f)
          val = 20.0f;
        if (val < -20.0f)
          val = -20.0f;
        fnet[l][j] = (2.0f / (1.0f + exp(-val)) - 1.0f);
      }
    }
  }

  float result = fnet[layer_count - 1][0];

  // Cleanup
  for (int l = 0; l < layer_count; l++) {
    delete[] net[l];
    delete[] fnet[l];
  }
  delete[] net;
  delete[] fnet;

  return result;
}
