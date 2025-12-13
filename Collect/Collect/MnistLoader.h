#pragma once

// Prevent Windows SDK conflicts with .NET Framework
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <algorithm>
#include <map>
#include <msclr/marshal_cppstd.h>
#include <random>
#include <string>
#include <vector>


using namespace System;
using namespace System::IO;
using namespace System::Drawing;
using namespace System::Collections::Generic;

// Helper struct for a single MNIST sample (Native C++ friendly)
struct MnistEntry {
  std::vector<float> pixels;
  int label;
};

// Managed wrapper or mixed class
public
ref class MnistLoader {
public:
  // Load from folder structure: root/{0-9}/*.png
  // Returns native vector via pointer/copy logic or we can just return managed
  // and convert later. To keep Form1 code minimal, let's return a pointer to
  // std::vector if possible, or use a helper to converting. Actually, simple
  // static method doing the work is cleanest.

  static std::vector<MnistEntry> LoadFromFolder(std::string rootPath) {
    std::vector<MnistEntry> dataset;
    String ^ folderPath = gcnew String(rootPath.c_str());

    if (!Directory::Exists(folderPath)) {
      // Try to find it relative to current execution?
      // Just return empty if not found.
      return dataset;
    }

    // Iterate 0 to 9
    for (int i = 0; i < 10; ++i) {
      String ^ classFolder = Path::Combine(folderPath, i.ToString());
      if (!Directory::Exists(classFolder))
        continue;

      array<String ^> ^ files = Directory::GetFiles(classFolder);
      for each (String ^ file in files) {
        try {
          Bitmap ^ bmp = gcnew Bitmap(file);
          // MNIST is 28x28. If images are different, resize?
          // User said 28x28 matrix so assume they are 28x28.
          // But safeguard:
          if (bmp->Width != 28 || bmp->Height != 28) {
            Bitmap ^ resized = gcnew Bitmap(bmp, 28, 28);
            delete bmp;
            bmp = resized;
          }

          MnistEntry entry;
          entry.label = i;
          entry.pixels.resize(28 * 28);

          for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
              Color c = bmp->GetPixel(x, y);
              // Grayscale conversion: (R+G+B)/3 or brightness
              // MNIST white on black or black on white?
              // MNIST original: White digit on Black background (0=background,
              // 255=digit). If user images are Black digit on White background,
              // we might need to invert. Standard MNIST pngs usually preserve
              // 0=black. Let's assume standard intensity.
              float val = (c.R + c.G + c.B) / 3.0f;

              // Normalize 0-255 -> [-1, 1] for tanh activation
              entry.pixels[y * 28 + x] = (val / 127.5f) - 1.0f;
            }
          }
          delete bmp;
          dataset.push_back(entry);
        } catch (...) {
          // Skip bad file
        }
      }
    }

    // Shuffle dataset? Default usually yes for training.
    // std::shuffle(dataset.begin(), dataset.end(),
    // std::default_random_engine(0)); Let form handle shuffling or do it here.

    return dataset;
  }
};
