#pragma once
#include "MnistLoader.h"
#include "pch.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace System;
using namespace System::Windows::Forms;

// MNIST Manager - Tüm MNIST işlemlerini kapsayan sınıf
// Mevcut Network.cpp ve Process.cpp'ye dokunmadan bağımsız çalışır
public
ref class MNIST_Manager {
public:
  // Dataset
  std::vector<MnistEntry> *trainData;
  std::vector<MnistEntry> *testData;
  bool dataLoaded;

  // MLP Network için ağırlıklar
  float **mlpWeights;
  float **mlpBias;
  int *mlpTopology;
  int mlpLayerCount;
  bool mlpTrained;

  // Autoencoder için ağırlıklar
  float **aeWeights;
  float **aeBias;
  int *aeTopology;
  int aeLayerCount;
  bool aeTrained;

  // Encoder + Classifier için ağırlıklar
  float **encClassifierWeights;
  float **encClassifierBias;
  int *encClassifierTopology;
  int encClassifierLayerCount;
  bool encClassifierTrained;

  // Constructor
  MNIST_Manager() {
    trainData = new std::vector<MnistEntry>();
    testData = new std::vector<MnistEntry>();
    dataLoaded = false;

    mlpWeights = nullptr;
    mlpBias = nullptr;
    mlpTopology = nullptr;
    mlpLayerCount = 0;
    mlpTrained = false;

    aeWeights = nullptr;
    aeBias = nullptr;
    aeTopology = nullptr;
    aeLayerCount = 0;
    aeTrained = false;

    encClassifierWeights = nullptr;
    encClassifierBias = nullptr;
    encClassifierTopology = nullptr;
    encClassifierLayerCount = 0;
    encClassifierTrained = false;
  }

  // Destructor
  ~MNIST_Manager() {
    Cleanup();
    delete trainData;
    delete testData;
  }

  // Bellek temizliği
  void Cleanup() {
    CleanupMLP();
    CleanupAutoencoder();
    CleanupEncoderClassifier();
  }

  void CleanupMLP() {
    if (mlpWeights) {
      for (int i = 0; i < mlpLayerCount - 1; i++) {
        delete[] mlpWeights[i];
        delete[] mlpBias[i];
      }
      delete[] mlpWeights;
      delete[] mlpBias;
      mlpWeights = nullptr;
      mlpBias = nullptr;
    }
    if (mlpTopology) {
      delete[] mlpTopology;
      mlpTopology = nullptr;
    }
    mlpTrained = false;
  }

  void CleanupAutoencoder() {
    if (aeWeights) {
      for (int i = 0; i < aeLayerCount - 1; i++) {
        delete[] aeWeights[i];
        delete[] aeBias[i];
      }
      delete[] aeWeights;
      delete[] aeBias;
      aeWeights = nullptr;
      aeBias = nullptr;
    }
    if (aeTopology) {
      delete[] aeTopology;
      aeTopology = nullptr;
    }
    aeTrained = false;
  }

  void CleanupEncoderClassifier() {
    if (encClassifierWeights) {
      for (int i = 0; i < encClassifierLayerCount - 1; i++) {
        delete[] encClassifierWeights[i];
        delete[] encClassifierBias[i];
      }
      delete[] encClassifierWeights;
      delete[] encClassifierBias;
      encClassifierWeights = nullptr;
      encClassifierBias = nullptr;
    }
    if (encClassifierTopology) {
      delete[] encClassifierTopology;
      encClassifierTopology = nullptr;
    }
    encClassifierTrained = false;
  }

  // ========== MNIST YÜKLEME ==========
  bool LoadMNIST(String ^ basePath, int trainSamplesPerDigit,
                 int testSamplesPerDigit) {
    trainData->clear();
    testData->clear();

    // Train klasörü
    String ^ trainPath = System::IO::Path::Combine(basePath, "train");
    // Test klasörü
    String ^ testPath = System::IO::Path::Combine(basePath, "test");

    // Load train data
    std::vector<MnistEntry> allTrain = MnistLoader::LoadFromFolder(
        msclr::interop::marshal_as<std::string>(trainPath));

    // Load test data
    std::vector<MnistEntry> allTest = MnistLoader::LoadFromFolder(
        msclr::interop::marshal_as<std::string>(testPath));

    if (allTrain.empty() || allTest.empty()) {
      return false;
    }

    // Her rakamdan belirli sayıda örnek seç
    SelectSamplesPerClass(allTrain, *trainData, trainSamplesPerDigit);
    SelectSamplesPerClass(allTest, *testData, testSamplesPerDigit);

    // Shuffle train data
    ShuffleData(*trainData);

    dataLoaded = true;
    return true;
  }

  // ========== MLP EĞİTİMİ ==========
  // 784 -> 128 -> 64 -> 10 mimarisi
  float *TrainMLP(float learningRate, float momentum, float minError,
                  int maxEpoch, int &finalEpoch) {
    if (!dataLoaded)
      return nullptr;

    CleanupMLP();

    // Topology: Input(784) -> Hidden1(128) -> Hidden2(64) -> Output(10)
    mlpLayerCount = 4;
    mlpTopology = new int[mlpLayerCount];
    mlpTopology[0] = 784;
    mlpTopology[1] = 128;
    mlpTopology[2] = 64;
    mlpTopology[3] = 10;

    // Ağırlık allocation
    AllocateWeightsForMLP();

    // Eğitim
    float *errorHistory = TrainNetwork(
        mlpWeights, mlpBias, mlpTopology, mlpLayerCount, *trainData,
        learningRate, momentum, minError, maxEpoch, finalEpoch, false);

    mlpTrained = true;
    return errorHistory;
  }

  // ========== MLP TEST ==========
  float TestMLP() { return TestMLP(nullptr); }

  float TestMLP(int confusionMatrix[10][10]) {
    if (!mlpTrained || !dataLoaded)
      return 0.0f;

    // Confusion matrix'i sıfırla
    if (confusionMatrix) {
      for (int i = 0; i < 10; i++)
        for (int j = 0; j < 10; j++)
          confusionMatrix[i][j] = 0;
    }

    int correct = 0;
    for (size_t i = 0; i < testData->size(); i++) {
      int predicted = ForwardPass((*testData)[i].pixels.data(), mlpWeights,
                                  mlpBias, mlpTopology, mlpLayerCount);
      int actual = (*testData)[i].label;

      if (confusionMatrix) {
        confusionMatrix[actual][predicted]++;
      }

      if (predicted == actual) {
        correct++;
      }
    }

    return (float)correct / (float)testData->size() * 100.0f;
  }

  // ========== AUTOENCODER EĞİTİMİ ==========
  // 784 -> 256 -> 32 -> 256 -> 784 (simetrik)
  float *TrainAutoencoder(float learningRate, float momentum, float minError,
                          int maxEpoch, int &finalEpoch) {
    if (!dataLoaded)
      return nullptr;

    CleanupAutoencoder();

    // Topology: 784 -> 256 -> 32 (latent) -> 256 -> 784
    aeLayerCount = 5;
    aeTopology = new int[aeLayerCount];
    aeTopology[0] = 784;
    aeTopology[1] = 256;
    aeTopology[2] = 32; // Latent space - artırıldı!
    aeTopology[3] = 256;
    aeTopology[4] = 784;

    // Ağırlık allocation
    AllocateWeightsForAutoencoder();

    // Autoencoder eğitimi (hedef = girdi)
    float *errorHistory = TrainAutoencoder_Internal(
        aeWeights, aeBias, aeTopology, aeLayerCount, *trainData, learningRate,
        momentum, minError, maxEpoch, finalEpoch);

    aeTrained = true;
    return errorHistory;
  }

  // ========== ENCODER + CLASSIFIER EĞİTİMİ ==========
  float *TrainEncoderClassifier(float learningRate, float momentum,
                                float minError, int maxEpoch, int &finalEpoch) {
    if (!aeTrained || !dataLoaded)
      return nullptr;

    CleanupEncoderClassifier();

    // Encoder output (32) -> Hidden(64) -> Output(10)
    encClassifierLayerCount = 3;
    encClassifierTopology = new int[encClassifierLayerCount];
    encClassifierTopology[0] =
        32; // Encoder çıktısı (latent dim) - güncellendi!
    encClassifierTopology[1] = 64;
    encClassifierTopology[2] = 10; // Sınıf sayısı

    // Ağırlık allocation
    AllocateWeightsForEncoderClassifier();

    // Önce tüm train datayı encoder'dan geçir
    std::vector<std::vector<float>> encodedData;
    std::vector<int> labels;

    for (size_t i = 0; i < trainData->size(); i++) {
      std::vector<float> encoded =
          GetEncoderOutput((*trainData)[i].pixels.data());
      encodedData.push_back(encoded);
      labels.push_back((*trainData)[i].label);
    }

    // Eğitim
    float *errorHistory = TrainClassifierOnEncodedData(
        encClassifierWeights, encClassifierBias, encClassifierTopology,
        encClassifierLayerCount, encodedData, labels, learningRate, momentum,
        minError, maxEpoch, finalEpoch);

    encClassifierTrained = true;
    return errorHistory;
  }

  // ========== ENCODER + CLASSIFIER TEST ==========
  float TestEncoderClassifier() { return TestEncoderClassifier(nullptr); }

  float TestEncoderClassifier(int confusionMatrix[10][10]) {
    if (!encClassifierTrained || !aeTrained || !dataLoaded)
      return 0.0f;

    // Confusion matrix'i sıfırla
    if (confusionMatrix) {
      for (int i = 0; i < 10; i++)
        for (int j = 0; j < 10; j++)
          confusionMatrix[i][j] = 0;
    }

    int correct = 0;
    for (size_t i = 0; i < testData->size(); i++) {
      // Önce encoder'dan geçir
      std::vector<float> encoded =
          GetEncoderOutput((*testData)[i].pixels.data());

      // Sonra classifier'dan geçir
      int predicted =
          ForwardPass(encoded.data(), encClassifierWeights, encClassifierBias,
                      encClassifierTopology, encClassifierLayerCount);

      int actual = (*testData)[i].label;

      if (confusionMatrix) {
        confusionMatrix[actual][predicted]++;
      }

      if (predicted == actual) {
        correct++;
      }
    }

    return (float)correct / (float)testData->size() * 100.0f;
  }

  // ========== AUTOENCODER REKONSTRÜKSİYON TESTİ ==========
  float *ReconstructImage(float *input) {
    if (!aeTrained)
      return nullptr;

    // Full autoencoder forward pass
    return AutoencoderForward(input, aeWeights, aeBias, aeTopology,
                              aeLayerCount);
  }

  // Getter'lar
  int GetTrainCount() { return (int)trainData->size(); }
  int GetTestCount() { return (int)testData->size(); }

private:
  // ========== YARDIMCI FONKSİYONLAR ==========

  void SelectSamplesPerClass(std::vector<MnistEntry> &source,
                             std::vector<MnistEntry> &dest,
                             int samplesPerClass) {
    // Her sınıf için ayrı listeler
    std::vector<MnistEntry> byClass[10];

    for (auto &entry : source) {
      if (entry.label >= 0 && entry.label < 10) {
        byClass[entry.label].push_back(entry);
      }
    }

    // Her sınıftan rastgele seç
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int c = 0; c < 10; c++) {
      std::shuffle(byClass[c].begin(), byClass[c].end(), gen);
      int count = (samplesPerClass < (int)byClass[c].size())
                      ? samplesPerClass
                      : (int)byClass[c].size();
      for (int i = 0; i < count; i++) {
        dest.push_back(byClass[c][i]);
      }
    }
  }

  void ShuffleData(std::vector<MnistEntry> &data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(data.begin(), data.end(), gen);
  }

  void AllocateWeightsForMLP() {
    mlpWeights = new float *[mlpLayerCount - 1];
    mlpBias = new float *[mlpLayerCount - 1];

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int l = 0; l < mlpLayerCount - 1; l++) {
      int src = mlpTopology[l];
      int dest = mlpTopology[l + 1];
      int size = src * dest;

      mlpWeights[l] = new float[size];
      mlpBias[l] = new float[dest];

      float limit = sqrt(6.0f / (float)(src + dest));
      std::uniform_real_distribution<float> dist(-limit, limit);

      for (int i = 0; i < size; i++) {
        mlpWeights[l][i] = dist(gen);
      }
      for (int i = 0; i < dest; i++) {
        mlpBias[l][i] = 0.0f;
      }
    }
  }

  void AllocateWeightsForAutoencoder() {
    aeWeights = new float *[aeLayerCount - 1];
    aeBias = new float *[aeLayerCount - 1];

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int l = 0; l < aeLayerCount - 1; l++) {
      int src = aeTopology[l];
      int dest = aeTopology[l + 1];
      int size = src * dest;

      aeWeights[l] = new float[size];
      aeBias[l] = new float[dest];

      float limit = sqrt(6.0f / (float)(src + dest));
      std::uniform_real_distribution<float> dist(-limit, limit);

      for (int i = 0; i < size; i++) {
        aeWeights[l][i] = dist(gen);
      }
      for (int i = 0; i < dest; i++) {
        aeBias[l][i] = 0.0f;
      }
    }
  }

  void AllocateWeightsForEncoderClassifier() {
    encClassifierWeights = new float *[encClassifierLayerCount - 1];
    encClassifierBias = new float *[encClassifierLayerCount - 1];

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int l = 0; l < encClassifierLayerCount - 1; l++) {
      int src = encClassifierTopology[l];
      int dest = encClassifierTopology[l + 1];
      int size = src * dest;

      encClassifierWeights[l] = new float[size];
      encClassifierBias[l] = new float[dest];

      float limit = sqrt(6.0f / (float)(src + dest));
      std::uniform_real_distribution<float> dist(-limit, limit);

      for (int i = 0; i < size; i++) {
        encClassifierWeights[l][i] = dist(gen);
      }
      for (int i = 0; i < dest; i++) {
        encClassifierBias[l][i] = 0.0f;
      }
    }
  }

  // Tanh aktivasyon
  float Tanh(float x) { return 2.0f / (1.0f + exp(-x)) - 1.0f; }

  float TanhDerivative(float output) { return 0.5f * (1.0f - output * output); }

  // Forward Pass - sınıf tahmini döndürür
  int ForwardPass(float *input, float **weights, float **bias, int *topology,
                  int layerCount) {
    float **fnet = new float *[layerCount];
    for (int l = 0; l < layerCount; l++) {
      fnet[l] = new float[topology[l]];
    }

    // Input layer
    for (int i = 0; i < topology[0]; i++) {
      fnet[0][i] = input[i];
    }

    // Forward
    for (int l = 1; l < layerCount; l++) {
      int prevNodes = topology[l - 1];
      int currNodes = topology[l];

      for (int j = 0; j < currNodes; j++) {
        float net = 0.0f;
        for (int i = 0; i < prevNodes; i++) {
          net += weights[l - 1][j * prevNodes + i] * fnet[l - 1][i];
        }
        net += bias[l - 1][j];
        fnet[l][j] = Tanh(net);
      }
    }

    // Argmax
    int outputLayer = layerCount - 1;
    int maxIdx = 0;
    float maxVal = fnet[outputLayer][0];
    for (int k = 1; k < topology[outputLayer]; k++) {
      if (fnet[outputLayer][k] > maxVal) {
        maxVal = fnet[outputLayer][k];
        maxIdx = k;
      }
    }

    // Cleanup
    for (int l = 0; l < layerCount; l++) {
      delete[] fnet[l];
    }
    delete[] fnet;

    return maxIdx;
  }

  // Encoder çıktısını al (latent space)
  std::vector<float> GetEncoderOutput(float *input) {
    std::vector<float> result;

    // Autoencoder'ın ilk yarısını çalıştır (encoder kısmı)
    // aeTopology: 784 -> 256 -> 32 -> 256 -> 784
    // Encoder: 784 -> 256 -> 32 (ilk 3 katman)

    float **fnet = new float *[3];
    fnet[0] = new float[784];
    fnet[1] = new float[256]; // 420'den 256'ya güncellendi
    fnet[2] = new float[32];  // 10'dan 32'ye güncellendi

    // Input
    for (int i = 0; i < 784; i++) {
      fnet[0][i] = input[i];
    }

    // Layer 1: 784 -> 256
    for (int j = 0; j < 256; j++) {
      float net = 0.0f;
      for (int i = 0; i < 784; i++) {
        net += aeWeights[0][j * 784 + i] * fnet[0][i];
      }
      net += aeBias[0][j];
      fnet[1][j] = Tanh(net);
    }

    // Layer 2: 256 -> 32 (latent)
    for (int j = 0; j < 32; j++) {
      float net = 0.0f;
      for (int i = 0; i < 256; i++) {
        net += aeWeights[1][j * 256 + i] * fnet[1][i];
      }
      net += aeBias[1][j];
      fnet[2][j] = Tanh(net);
      result.push_back(fnet[2][j]);
    }

    delete[] fnet[0];
    delete[] fnet[1];
    delete[] fnet[2];
    delete[] fnet;

    return result;
  }

  // Autoencoder full forward (rekonstrüksiyon için)
  float *AutoencoderForward(float *input, float **weights, float **bias,
                            int *topology, int layerCount) {
    float **fnet = new float *[layerCount];
    for (int l = 0; l < layerCount; l++) {
      fnet[l] = new float[topology[l]];
    }

    // Input
    for (int i = 0; i < topology[0]; i++) {
      fnet[0][i] = input[i];
    }

    // Forward through all layers
    for (int l = 1; l < layerCount; l++) {
      int prevNodes = topology[l - 1];
      int currNodes = topology[l];

      for (int j = 0; j < currNodes; j++) {
        float net = 0.0f;
        for (int i = 0; i < prevNodes; i++) {
          net += weights[l - 1][j * prevNodes + i] * fnet[l - 1][i];
        }
        net += bias[l - 1][j];
        fnet[l][j] = Tanh(net);
      }
    }

    // Copy output
    float *output = new float[topology[layerCount - 1]];
    for (int i = 0; i < topology[layerCount - 1]; i++) {
      output[i] = fnet[layerCount - 1][i];
    }

    // Cleanup
    for (int l = 0; l < layerCount; l++) {
      delete[] fnet[l];
    }
    delete[] fnet;

    return output;
  }

  // MLP Eğitim fonksiyonu
  float *TrainNetwork(float **weights, float **bias, int *topology,
                      int layerCount, std::vector<MnistEntry> &data,
                      float learningRate, float momentum, float minError,
                      int maxEpoch, int &epoch, bool isAutoencoder) {

    float *errorHistory = new float[maxEpoch];

    // Allocation
    float **net = new float *[layerCount];
    float **fnet = new float *[layerCount];
    float **fder = new float *[layerCount];
    float **delta = new float *[layerCount];
    float **prevDW = new float *[layerCount - 1];
    float **prevDB = new float *[layerCount - 1];

    for (int l = 0; l < layerCount; l++) {
      net[l] = new float[topology[l]];
      fnet[l] = new float[topology[l]];
      fder[l] = new float[topology[l]];
      delta[l] = new float[topology[l]];
    }

    for (int l = 0; l < layerCount - 1; l++) {
      int size = topology[l] * topology[l + 1];
      prevDW[l] = new float[size]();
      prevDB[l] = new float[topology[l + 1]]();
    }

    float *desired = new float[topology[layerCount - 1]];
    float *err = new float[topology[layerCount - 1]];

    epoch = 0;
    float totalErr;

    do {
      totalErr = 0.0f;

      // Shuffle at each epoch
      ShuffleData(data);

      for (size_t step = 0; step < data.size(); step++) {
        // Load input
        for (int i = 0; i < topology[0]; i++) {
          fnet[0][i] = data[step].pixels[i];
        }

        // Forward pass
        for (int l = 1; l < layerCount; l++) {
          int prevNodes = topology[l - 1];
          int currNodes = topology[l];

          for (int j = 0; j < currNodes; j++) {
            net[l][j] = 0.0f;
            for (int i = 0; i < prevNodes; i++) {
              net[l][j] += weights[l - 1][j * prevNodes + i] * fnet[l - 1][i];
            }
            net[l][j] += bias[l - 1][j];
            fnet[l][j] = Tanh(net[l][j]);
            fder[l][j] = TanhDerivative(fnet[l][j]);
          }
        }

        // Prepare target (one-hot for classification)
        int outputNodes = topology[layerCount - 1];
        for (int k = 0; k < outputNodes; k++) {
          if (data[step].label == k) {
            desired[k] = 1.0f;
          } else {
            desired[k] = -1.0f;
          }
        }

        // Output layer error and delta
        for (int k = 0; k < outputNodes; k++) {
          err[k] = desired[k] - fnet[layerCount - 1][k];

          // Gradient clipping
          float d = learningRate * err[k] * fder[layerCount - 1][k];
          if (d > 10.0f)
            d = 10.0f;
          if (d < -10.0f)
            d = -10.0f;
          delta[layerCount - 1][k] = d;

          totalErr += 0.5f * err[k] * err[k];
        }

        // Hidden layer deltas (backprop)
        for (int l = layerCount - 2; l > 0; l--) {
          int currNodes = topology[l];
          int nextNodes = topology[l + 1];

          for (int j = 0; j < currNodes; j++) {
            float errorSum = 0.0f;
            for (int k = 0; k < nextNodes; k++) {
              errorSum += delta[l + 1][k] * weights[l][k * currNodes + j];
            }
            float d = learningRate * errorSum * fder[l][j];
            if (d > 10.0f)
              d = 10.0f;
            if (d < -10.0f)
              d = -10.0f;
            delta[l][j] = d;
          }
        }

        // Weight update with momentum
        for (int l = 0; l < layerCount - 1; l++) {
          int srcNodes = topology[l];
          int destNodes = topology[l + 1];

          for (int j = 0; j < destNodes; j++) {
            for (int i = 0; i < srcNodes; i++) {
              float change = delta[l + 1][j] * fnet[l][i];
              float update = change + momentum * prevDW[l][j * srcNodes + i];
              weights[l][j * srcNodes + i] += update;
              prevDW[l][j * srcNodes + i] = update;
            }

            float biasChange = delta[l + 1][j];
            float biasUpdate = biasChange + momentum * prevDB[l][j];
            bias[l][j] += biasUpdate;
            prevDB[l][j] = biasUpdate;
          }
        }
      }

      totalErr /= (float)(topology[layerCount - 1] * data.size());
      errorHistory[epoch] = totalErr;
      epoch++;

    } while (totalErr > minError && epoch < maxEpoch);

    // Cleanup
    for (int l = 0; l < layerCount; l++) {
      delete[] net[l];
      delete[] fnet[l];
      delete[] fder[l];
      delete[] delta[l];
    }
    for (int l = 0; l < layerCount - 1; l++) {
      delete[] prevDW[l];
      delete[] prevDB[l];
    }
    delete[] net;
    delete[] fnet;
    delete[] fder;
    delete[] delta;
    delete[] prevDW;
    delete[] prevDB;
    delete[] desired;
    delete[] err;

    return errorHistory;
  }

  // Autoencoder eğitimi (hedef = girdi)
  float *TrainAutoencoder_Internal(float **weights, float **bias, int *topology,
                                   int layerCount,
                                   std::vector<MnistEntry> &data,
                                   float learningRate, float momentum,
                                   float minError, int maxEpoch, int &epoch) {

    float *errorHistory = new float[maxEpoch];

    // Allocation
    float **net = new float *[layerCount];
    float **fnet = new float *[layerCount];
    float **fder = new float *[layerCount];
    float **delta = new float *[layerCount];
    float **prevDW = new float *[layerCount - 1];
    float **prevDB = new float *[layerCount - 1];

    for (int l = 0; l < layerCount; l++) {
      net[l] = new float[topology[l]];
      fnet[l] = new float[topology[l]];
      fder[l] = new float[topology[l]];
      delta[l] = new float[topology[l]];
    }

    for (int l = 0; l < layerCount - 1; l++) {
      int size = topology[l] * topology[l + 1];
      prevDW[l] = new float[size]();
      prevDB[l] = new float[topology[l + 1]]();
    }

    float *err = new float[topology[layerCount - 1]];

    epoch = 0;
    float totalErr;

    do {
      totalErr = 0.0f;
      ShuffleData(data);

      for (size_t step = 0; step < data.size(); step++) {
        // Load input
        for (int i = 0; i < topology[0]; i++) {
          fnet[0][i] = data[step].pixels[i];
        }

        // Forward pass
        for (int l = 1; l < layerCount; l++) {
          int prevNodes = topology[l - 1];
          int currNodes = topology[l];

          for (int j = 0; j < currNodes; j++) {
            net[l][j] = 0.0f;
            for (int i = 0; i < prevNodes; i++) {
              net[l][j] += weights[l - 1][j * prevNodes + i] * fnet[l - 1][i];
            }
            net[l][j] += bias[l - 1][j];
            fnet[l][j] = Tanh(net[l][j]);
            fder[l][j] = TanhDerivative(fnet[l][j]);
          }
        }

        // Autoencoder: target = input
        int outputNodes = topology[layerCount - 1];
        for (int k = 0; k < outputNodes; k++) {
          err[k] = data[step].pixels[k] - fnet[layerCount - 1][k];

          float d = learningRate * err[k] * fder[layerCount - 1][k];
          if (d > 10.0f)
            d = 10.0f;
          if (d < -10.0f)
            d = -10.0f;
          delta[layerCount - 1][k] = d;

          totalErr += 0.5f * err[k] * err[k];
        }

        // Backpropagation
        for (int l = layerCount - 2; l > 0; l--) {
          int currNodes = topology[l];
          int nextNodes = topology[l + 1];

          for (int j = 0; j < currNodes; j++) {
            float errorSum = 0.0f;
            for (int k = 0; k < nextNodes; k++) {
              errorSum += delta[l + 1][k] * weights[l][k * currNodes + j];
            }
            float d = learningRate * errorSum * fder[l][j];
            if (d > 10.0f)
              d = 10.0f;
            if (d < -10.0f)
              d = -10.0f;
            delta[l][j] = d;
          }
        }

        // Weight update
        for (int l = 0; l < layerCount - 1; l++) {
          int srcNodes = topology[l];
          int destNodes = topology[l + 1];

          for (int j = 0; j < destNodes; j++) {
            for (int i = 0; i < srcNodes; i++) {
              float change = delta[l + 1][j] * fnet[l][i];
              float update = change + momentum * prevDW[l][j * srcNodes + i];
              weights[l][j * srcNodes + i] += update;
              prevDW[l][j * srcNodes + i] = update;
            }

            float biasChange = delta[l + 1][j];
            float biasUpdate = biasChange + momentum * prevDB[l][j];
            bias[l][j] += biasUpdate;
            prevDB[l][j] = biasUpdate;
          }
        }
      }

      totalErr /= (float)(topology[layerCount - 1] * data.size());
      errorHistory[epoch] = totalErr;
      epoch++;

    } while (totalErr > minError && epoch < maxEpoch);

    // Cleanup
    for (int l = 0; l < layerCount; l++) {
      delete[] net[l];
      delete[] fnet[l];
      delete[] fder[l];
      delete[] delta[l];
    }
    for (int l = 0; l < layerCount - 1; l++) {
      delete[] prevDW[l];
      delete[] prevDB[l];
    }
    delete[] net;
    delete[] fnet;
    delete[] fder;
    delete[] delta;
    delete[] prevDW;
    delete[] prevDB;
    delete[] err;

    return errorHistory;
  }

  // Encoded data üzerinde classifier eğitimi
  float *TrainClassifierOnEncodedData(
      float **weights, float **bias, int *topology, int layerCount,
      std::vector<std::vector<float>> &encodedData, std::vector<int> &labels,
      float learningRate, float momentum, float minError, int maxEpoch,
      int &epoch) {

    float *errorHistory = new float[maxEpoch];

    // Allocation
    float **net = new float *[layerCount];
    float **fnet = new float *[layerCount];
    float **fder = new float *[layerCount];
    float **delta = new float *[layerCount];
    float **prevDW = new float *[layerCount - 1];
    float **prevDB = new float *[layerCount - 1];

    for (int l = 0; l < layerCount; l++) {
      net[l] = new float[topology[l]];
      fnet[l] = new float[topology[l]];
      fder[l] = new float[topology[l]];
      delta[l] = new float[topology[l]];
    }

    for (int l = 0; l < layerCount - 1; l++) {
      int size = topology[l] * topology[l + 1];
      prevDW[l] = new float[size]();
      prevDB[l] = new float[topology[l + 1]]();
    }

    float *desired = new float[topology[layerCount - 1]];
    float *err = new float[topology[layerCount - 1]];

    // Shuffle indices
    std::vector<int> indices(encodedData.size());
    for (size_t i = 0; i < indices.size(); i++)
      indices[i] = (int)i;

    epoch = 0;
    float totalErr;

    do {
      totalErr = 0.0f;

      // Shuffle
      std::random_device rd;
      std::mt19937 gen(rd());
      std::shuffle(indices.begin(), indices.end(), gen);

      for (size_t s = 0; s < encodedData.size(); s++) {
        int step = indices[s];

        // Load input (encoded data)
        for (int i = 0; i < topology[0]; i++) {
          fnet[0][i] = encodedData[step][i];
        }

        // Forward pass
        for (int l = 1; l < layerCount; l++) {
          int prevNodes = topology[l - 1];
          int currNodes = topology[l];

          for (int j = 0; j < currNodes; j++) {
            net[l][j] = 0.0f;
            for (int i = 0; i < prevNodes; i++) {
              net[l][j] += weights[l - 1][j * prevNodes + i] * fnet[l - 1][i];
            }
            net[l][j] += bias[l - 1][j];
            fnet[l][j] = Tanh(net[l][j]);
            fder[l][j] = TanhDerivative(fnet[l][j]);
          }
        }

        // Target (one-hot)
        int outputNodes = topology[layerCount - 1];
        for (int k = 0; k < outputNodes; k++) {
          if (labels[step] == k) {
            desired[k] = 1.0f;
          } else {
            desired[k] = -1.0f;
          }
        }

        // Error and delta
        for (int k = 0; k < outputNodes; k++) {
          err[k] = desired[k] - fnet[layerCount - 1][k];
          float d = learningRate * err[k] * fder[layerCount - 1][k];
          if (d > 10.0f)
            d = 10.0f;
          if (d < -10.0f)
            d = -10.0f;
          delta[layerCount - 1][k] = d;
          totalErr += 0.5f * err[k] * err[k];
        }

        // Backprop
        for (int l = layerCount - 2; l > 0; l--) {
          int currNodes = topology[l];
          int nextNodes = topology[l + 1];

          for (int j = 0; j < currNodes; j++) {
            float errorSum = 0.0f;
            for (int k = 0; k < nextNodes; k++) {
              errorSum += delta[l + 1][k] * weights[l][k * currNodes + j];
            }
            float d = learningRate * errorSum * fder[l][j];
            if (d > 10.0f)
              d = 10.0f;
            if (d < -10.0f)
              d = -10.0f;
            delta[l][j] = d;
          }
        }

        // Weight update
        for (int l = 0; l < layerCount - 1; l++) {
          int srcNodes = topology[l];
          int destNodes = topology[l + 1];

          for (int j = 0; j < destNodes; j++) {
            for (int i = 0; i < srcNodes; i++) {
              float change = delta[l + 1][j] * fnet[l][i];
              float update = change + momentum * prevDW[l][j * srcNodes + i];
              weights[l][j * srcNodes + i] += update;
              prevDW[l][j * srcNodes + i] = update;
            }

            float biasChange = delta[l + 1][j];
            float biasUpdate = biasChange + momentum * prevDB[l][j];
            bias[l][j] += biasUpdate;
            prevDB[l][j] = biasUpdate;
          }
        }
      }

      totalErr /= (float)(topology[layerCount - 1] * encodedData.size());
      errorHistory[epoch] = totalErr;
      epoch++;

    } while (totalErr > minError && epoch < maxEpoch);

    // Cleanup
    for (int l = 0; l < layerCount; l++) {
      delete[] net[l];
      delete[] fnet[l];
      delete[] fder[l];
      delete[] delta[l];
    }
    for (int l = 0; l < layerCount - 1; l++) {
      delete[] prevDW[l];
      delete[] prevDB[l];
    }
    delete[] net;
    delete[] fnet;
    delete[] fder;
    delete[] delta;
    delete[] prevDW;
    delete[] prevDB;
    delete[] desired;
    delete[] err;

    return errorHistory;
  }
};
