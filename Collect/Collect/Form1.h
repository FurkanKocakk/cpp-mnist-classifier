#pragma once
#include "MNIST_Manager.h"
#include "Network.h"
#include "Process.h"
#include <fstream>
#include <iostream>
#include <string>

namespace CppCLRWinformsProjekt {

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;
using namespace System::IO;

/// <summary>
/// Zusammenfassung f�r Form1
/// </summary>
public
ref class Form1 : public System::Windows::Forms::Form {
public:
  Form1(void) {
    InitializeComponent();
    // MNIST Manager (Designer uyumlulugu icin burada)
    mnistManager = gcnew MNIST_Manager();
    //
    // TODO: Konstruktorcode hier hinzuf�gen.
    //
  }

protected:
  /// <summary>
  /// Verwendete Ressourcen bereinigen.
  /// </summary>
  ~Form1() {
    if (components) {
      delete components;
    }
    // Dinamik dizileri güvenli şekilde temizle
    if (Samples) {
      delete[] Samples;
      Samples = nullptr;
    }
    if (targets) {
      delete[] targets;
      targets = nullptr;
    }
    if (Weights) {
      for (int i = 0; i < layer_count - 1; i++)
        delete[] Weights[i];
      delete[] Weights;
      Weights = nullptr;
    }
    if (bias) {
      for (int i = 0; i < layer_count - 1; i++)
        delete[] bias[i];
      delete[] bias;
      bias = nullptr;
    }
    if (topology) {
      delete[] topology;
      topology = nullptr;
    }
  }

private:
  System::Windows::Forms::PictureBox ^ pictureBox1;

  // Header of Form1.h
private:
  System::Windows::Forms::ComboBox ^ cbLayerCount;
  System::Windows::Forms::FlowLayoutPanel ^ pnlLayers;
  System::Windows::Forms::Label ^ lblLayerCount;

private:
  System::Windows::Forms::GroupBox ^ groupBox1;
  System::Windows::Forms::Button ^ Set_Net;
  System::Windows::Forms::Label ^ label1;
  System::Windows::Forms::ComboBox ^ ClassCountBox;
  System::Windows::Forms::Label ^ lblHiddenLayers;
  // txtHiddenLayers removed

private:
  System::Windows::Forms::GroupBox ^ groupBox2;
  System::Windows::Forms::Label ^ label2;
  System::Windows::Forms::ComboBox ^ ClassNoBox;
  System::Windows::Forms::Label ^ label3;

  // ... (rest of controls)

  // In InitializeComponent:

private:
private:
  /// <summary>
  /// User Defined Variables
  int numSample = 0, inputDim = 2;
  // MLP Variables
  int *topology = nullptr;
  int layer_count = 0;
  int class_count = 0; // Output layer size

  float *Samples, *targets;
  float **Weights, **bias;

private:
  System::Windows::Forms::MenuStrip ^ menuStrip1;

private:
  System::Windows::Forms::ToolStripMenuItem ^ fileToolStripMenuItem;

private:
  System::Windows::Forms::ToolStripMenuItem ^ readDataToolStripMenuItem;

private:
  System::Windows::Forms::OpenFileDialog ^ openFileDialog1;

private:
  System::Windows::Forms::TextBox ^ textBox1;

private:
  System::Windows::Forms::ToolStripMenuItem ^ saveDataToolStripMenuItem;

private:
  System::Windows::Forms::SaveFileDialog ^ saveFileDialog1;

private:
  System::Windows::Forms::ToolStripMenuItem ^ processToolStripMenuItem;

private:
  System::Windows::Forms::ToolStripMenuItem ^ trainingToolStripMenuItem;

private:
  System::Windows::Forms::ToolStripMenuItem ^ testingToolStripMenuItem;

private:
  System::Windows::Forms::ToolStripMenuItem ^ regressionToolStripMenuItem;

  // MNIST Menu Items
private:
  System::Windows::Forms::ToolStripMenuItem ^ mnistToolStripMenuItem;
  System::Windows::Forms::ToolStripMenuItem ^ loadMNISTToolStripMenuItem;
  System::Windows::Forms::ToolStripMenuItem ^ trainMNISTToolStripMenuItem;
  System::Windows::Forms::ToolStripMenuItem ^ testMNISTToolStripMenuItem;
  System::Windows::Forms::ToolStripMenuItem ^ trainAutoencoderToolStripMenuItem;
  System::Windows::Forms::ToolStripMenuItem ^
      testEncoderClassifierToolStripMenuItem;
  System::Windows::Forms::FolderBrowserDialog ^ folderBrowserDialog1;

private:
  System::Windows::Forms::DataVisualization::Charting::Chart ^ chart1;

  /// </summary>
  System::ComponentModel::Container ^ components;

#pragma region Windows Form Designer generated code
  /// <summary>
  /// Erforderliche Methode f�r die Designerunterst�tzung.
  /// Der Inhalt der Methode darf nicht mit dem Code-Editor ge�ndert werden.
  /// </summary>
  void InitializeComponent(void) {
    System::Windows::Forms::DataVisualization::Charting::ChartArea ^
        chartArea2 =
        (gcnew
             System::Windows::Forms::DataVisualization::Charting::ChartArea());
    System::Windows::Forms::DataVisualization::Charting::Legend ^ legend2 =
        (gcnew System::Windows::Forms::DataVisualization::Charting::Legend());
    System::Windows::Forms::DataVisualization::Charting::Series ^ series2 =
        (gcnew System::Windows::Forms::DataVisualization::Charting::Series());
    this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
    this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
    this->pnlLayers = (gcnew System::Windows::Forms::FlowLayoutPanel());
    this->lblLayerCount = (gcnew System::Windows::Forms::Label());
    this->cbLayerCount = (gcnew System::Windows::Forms::ComboBox());
    this->Set_Net = (gcnew System::Windows::Forms::Button());
    this->label1 = (gcnew System::Windows::Forms::Label());
    this->ClassCountBox = (gcnew System::Windows::Forms::ComboBox());
    this->groupBox2 = (gcnew System::Windows::Forms::GroupBox());
    this->label2 = (gcnew System::Windows::Forms::Label());
    this->ClassNoBox = (gcnew System::Windows::Forms::ComboBox());
    this->label3 = (gcnew System::Windows::Forms::Label());
    this->menuStrip1 = (gcnew System::Windows::Forms::MenuStrip());
    this->fileToolStripMenuItem =
        (gcnew System::Windows::Forms::ToolStripMenuItem());
    this->readDataToolStripMenuItem =
        (gcnew System::Windows::Forms::ToolStripMenuItem());
    this->saveDataToolStripMenuItem =
        (gcnew System::Windows::Forms::ToolStripMenuItem());
    this->processToolStripMenuItem =
        (gcnew System::Windows::Forms::ToolStripMenuItem());
    this->trainingToolStripMenuItem =
        (gcnew System::Windows::Forms::ToolStripMenuItem());
    this->testingToolStripMenuItem =
        (gcnew System::Windows::Forms::ToolStripMenuItem());
    this->regressionToolStripMenuItem =
        (gcnew System::Windows::Forms::ToolStripMenuItem());
    this->mnistToolStripMenuItem =
        (gcnew System::Windows::Forms::ToolStripMenuItem());
    this->loadMNISTToolStripMenuItem =
        (gcnew System::Windows::Forms::ToolStripMenuItem());
    this->trainMNISTToolStripMenuItem =
        (gcnew System::Windows::Forms::ToolStripMenuItem());
    this->testMNISTToolStripMenuItem =
        (gcnew System::Windows::Forms::ToolStripMenuItem());
    this->trainAutoencoderToolStripMenuItem =
        (gcnew System::Windows::Forms::ToolStripMenuItem());
    this->testEncoderClassifierToolStripMenuItem =
        (gcnew System::Windows::Forms::ToolStripMenuItem());
    this->openFileDialog1 = (gcnew System::Windows::Forms::OpenFileDialog());
    this->textBox1 = (gcnew System::Windows::Forms::TextBox());
    this->saveFileDialog1 = (gcnew System::Windows::Forms::SaveFileDialog());
    this->chart1 =
        (gcnew System::Windows::Forms::DataVisualization::Charting::Chart());
    this->lblMomentum = (gcnew System::Windows::Forms::Label());
    this->txtMomentum = (gcnew System::Windows::Forms::TextBox());
    this->btnSetMomentum = (gcnew System::Windows::Forms::Button());
    this->folderBrowserDialog1 =
        (gcnew System::Windows::Forms::FolderBrowserDialog());
    (cli::safe_cast<System::ComponentModel::ISupportInitialize ^>(
         this->pictureBox1))
        ->BeginInit();
    this->groupBox1->SuspendLayout();
    this->groupBox2->SuspendLayout();
    this->menuStrip1->SuspendLayout();
    (cli::safe_cast<System::ComponentModel::ISupportInitialize ^>(this->chart1))
        ->BeginInit();
    this->SuspendLayout();
    //
    // pictureBox1
    //
    this->pictureBox1->BackColor =
        System::Drawing::SystemColors::ButtonHighlight;
    this->pictureBox1->Location = System::Drawing::Point(13, 35);
    this->pictureBox1->Name = L"pictureBox1";
    this->pictureBox1->Size = System::Drawing::Size(802, 578);
    this->pictureBox1->TabIndex = 0;
    this->pictureBox1->TabStop = false;
    this->pictureBox1->Paint += gcnew System::Windows::Forms::PaintEventHandler(
        this, &Form1::pictureBox1_Paint);
    this->pictureBox1->MouseClick +=
        gcnew System::Windows::Forms::MouseEventHandler(
            this, &Form1::pictureBox1_MouseClick);
    //
    // groupBox1
    //
    this->groupBox1->Controls->Add(this->pnlLayers);
    this->groupBox1->Controls->Add(this->lblLayerCount);
    this->groupBox1->Controls->Add(this->cbLayerCount);
    this->groupBox1->Controls->Add(this->Set_Net);
    this->groupBox1->Controls->Add(this->label1);
    this->groupBox1->Controls->Add(this->ClassCountBox);
    this->groupBox1->Font = (gcnew System::Drawing::Font(
        L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Bold,
        System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(162)));
    this->groupBox1->Location = System::Drawing::Point(850, 35);
    this->groupBox1->Name = L"groupBox1";
    this->groupBox1->Size = System::Drawing::Size(220, 300);
    this->groupBox1->TabIndex = 1;
    this->groupBox1->TabStop = false;
    this->groupBox1->Text = L"Network Architecture";
    //
    // pnlLayers
    //
    this->pnlLayers->AutoScroll = true;
    this->pnlLayers->Location = System::Drawing::Point(10, 100);
    this->pnlLayers->Name = L"pnlLayers";
    this->pnlLayers->Size = System::Drawing::Size(200, 150);
    this->pnlLayers->TabIndex = 5;
    //
    // lblLayerCount
    //
    this->lblLayerCount->AutoSize = true;
    this->lblLayerCount->Location = System::Drawing::Point(100, 70);
    this->lblLayerCount->Name = L"lblLayerCount";
    this->lblLayerCount->Size = System::Drawing::Size(86, 13);
    this->lblLayerCount->TabIndex = 3;
    this->lblLayerCount->Text = L"Katman Sayisi";
    //
    // cbLayerCount
    //
    this->cbLayerCount->FormattingEnabled = true;
    this->cbLayerCount->Items->AddRange(gcnew cli::array<System::Object ^>(6){
        L"0", L"1", L"2", L"3", L"4", L"5"});
    this->cbLayerCount->Location = System::Drawing::Point(10, 67);
    this->cbLayerCount->Name = L"cbLayerCount";
    this->cbLayerCount->Size = System::Drawing::Size(82, 21);
    this->cbLayerCount->TabIndex = 4;
    this->cbLayerCount->Text = L"1";
    this->cbLayerCount->SelectedIndexChanged += gcnew System::EventHandler(
        this, &Form1::cbLayerCount_SelectedIndexChanged);
    //
    // Set_Net
    //
    this->Set_Net->Location = System::Drawing::Point(10, 260);
    this->Set_Net->Name = L"Set_Net";
    this->Set_Net->Size = System::Drawing::Size(131, 33);
    this->Set_Net->TabIndex = 6;
    this->Set_Net->Text = L"Network Setting";
    this->Set_Net->UseVisualStyleBackColor = true;
    this->Set_Net->Click +=
        gcnew System::EventHandler(this, &Form1::Set_Net_Click);
    //
    // label1
    //
    this->label1->AutoSize = true;
    this->label1->Location = System::Drawing::Point(100, 38);
    this->label1->Name = L"label1";
    this->label1->Size = System::Drawing::Size(69, 13);
    this->label1->TabIndex = 1;
    this->label1->Text = L"Sinif Sayisi";
    //
    // ClassCountBox
    //
    this->ClassCountBox->FormattingEnabled = true;
    this->ClassCountBox->Items->AddRange(gcnew cli::array<System::Object ^>(6){
        L"2", L"3", L"4", L"5", L"6", L"7"});
    this->ClassCountBox->Location = System::Drawing::Point(10, 35);
    this->ClassCountBox->Name = L"ClassCountBox";
    this->ClassCountBox->Size = System::Drawing::Size(82, 21);
    this->ClassCountBox->TabIndex = 0;
    this->ClassCountBox->Text = L"2";
    //
    // groupBox2
    //
    this->groupBox2->Controls->Add(this->label2);
    this->groupBox2->Controls->Add(this->ClassNoBox);
    this->groupBox2->Font = (gcnew System::Drawing::Font(
        L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Bold,
        System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(162)));
    this->groupBox2->Location = System::Drawing::Point(1103, 43);
    this->groupBox2->Name = L"groupBox2";
    this->groupBox2->Size = System::Drawing::Size(220, 80);
    this->groupBox2->TabIndex = 2;
    this->groupBox2->TabStop = false;
    this->groupBox2->Text = L"Data Collection";
    //
    // label2
    //
    this->label2->AutoSize = true;
    this->label2->Location = System::Drawing::Point(98, 23);
    this->label2->Name = L"label2";
    this->label2->Size = System::Drawing::Size(81, 13);
    this->label2->TabIndex = 1;
    this->label2->Text = L"Ornek Etiketi";
    //
    // ClassNoBox
    //
    this->ClassNoBox->FormattingEnabled = true;
    this->ClassNoBox->Items->AddRange(gcnew cli::array<System::Object ^>(9){
        L"1", L"2", L"3", L"4", L"5", L"6", L"7", L"8", L"9"});
    this->ClassNoBox->Location = System::Drawing::Point(7, 20);
    this->ClassNoBox->Name = L"ClassNoBox";
    this->ClassNoBox->Size = System::Drawing::Size(75, 21);
    this->ClassNoBox->TabIndex = 0;
    this->ClassNoBox->Text = L"1";
    //
    // label3
    //
    this->label3->AutoSize = true;
    this->label3->Location = System::Drawing::Point(1348, 63);
    this->label3->Name = L"label3";
    this->label3->Size = System::Drawing::Size(59, 13);
    this->label3->TabIndex = 3;
    this->label3->Text = L"Samples: 0";
    //
    // menuStrip1
    //
    this->menuStrip1->Items->AddRange(
        gcnew cli::array<System::Windows::Forms::ToolStripItem ^>(3){
            this->fileToolStripMenuItem, this->processToolStripMenuItem,
            this->mnistToolStripMenuItem});
    this->menuStrip1->Location = System::Drawing::Point(0, 0);
    this->menuStrip1->Name = L"menuStrip1";
    this->menuStrip1->Size = System::Drawing::Size(1505, 24);
    this->menuStrip1->TabIndex = 4;
    this->menuStrip1->Text = L"menuStrip1";
    //
    // fileToolStripMenuItem
    //
    this->fileToolStripMenuItem->DropDownItems->AddRange(
        gcnew cli::array<System::Windows::Forms::ToolStripItem ^>(2){
            this->readDataToolStripMenuItem, this->saveDataToolStripMenuItem});
    this->fileToolStripMenuItem->Name = L"fileToolStripMenuItem";
    this->fileToolStripMenuItem->Size = System::Drawing::Size(37, 20);
    this->fileToolStripMenuItem->Text = L"File";
    //
    // readDataToolStripMenuItem
    //
    this->readDataToolStripMenuItem->Name = L"readDataToolStripMenuItem";
    this->readDataToolStripMenuItem->Size = System::Drawing::Size(129, 22);
    this->readDataToolStripMenuItem->Text = L"Read_Data";
    this->readDataToolStripMenuItem->Click += gcnew System::EventHandler(
        this, &Form1::readDataToolStripMenuItem_Click);
    //
    // saveDataToolStripMenuItem
    //
    this->saveDataToolStripMenuItem->Name = L"saveDataToolStripMenuItem";
    this->saveDataToolStripMenuItem->Size = System::Drawing::Size(129, 22);
    this->saveDataToolStripMenuItem->Text = L"Save_Data";
    this->saveDataToolStripMenuItem->Click += gcnew System::EventHandler(
        this, &Form1::saveDataToolStripMenuItem_Click);
    //
    // processToolStripMenuItem
    //
    this->processToolStripMenuItem->DropDownItems->AddRange(
        gcnew cli::array<System::Windows::Forms::ToolStripItem ^>(3){
            this->trainingToolStripMenuItem, this->testingToolStripMenuItem,
            this->regressionToolStripMenuItem});
    this->processToolStripMenuItem->Name = L"processToolStripMenuItem";
    this->processToolStripMenuItem->Size = System::Drawing::Size(59, 20);
    this->processToolStripMenuItem->Text = L"Process";
    //
    // trainingToolStripMenuItem
    //
    this->trainingToolStripMenuItem->Name = L"trainingToolStripMenuItem";
    this->trainingToolStripMenuItem->Size = System::Drawing::Size(131, 22);
    this->trainingToolStripMenuItem->Text = L"Training";
    this->trainingToolStripMenuItem->Click += gcnew System::EventHandler(
        this, &Form1::trainingToolStripMenuItem_Click);
    //
    // testingToolStripMenuItem
    //
    this->testingToolStripMenuItem->Name = L"testingToolStripMenuItem";
    this->testingToolStripMenuItem->Size = System::Drawing::Size(131, 22);
    this->testingToolStripMenuItem->Text = L"Testing";
    this->testingToolStripMenuItem->Click += gcnew System::EventHandler(
        this, &Form1::testingToolStripMenuItem_Click);
    //
    // regressionToolStripMenuItem
    //
    this->regressionToolStripMenuItem->Name = L"regressionToolStripMenuItem";
    this->regressionToolStripMenuItem->Size = System::Drawing::Size(131, 22);
    this->regressionToolStripMenuItem->Text = L"Regression";
    this->regressionToolStripMenuItem->Click += gcnew System::EventHandler(
        this, &Form1::regressionToolStripMenuItem_Click);
    //
    // mnistToolStripMenuItem
    //
    this->mnistToolStripMenuItem->DropDownItems->AddRange(
        gcnew cli::array<System::Windows::Forms::ToolStripItem ^>(5){
            this->loadMNISTToolStripMenuItem, this->trainMNISTToolStripMenuItem,
            this->testMNISTToolStripMenuItem,
            this->trainAutoencoderToolStripMenuItem,
            this->testEncoderClassifierToolStripMenuItem});
    this->mnistToolStripMenuItem->Name = L"mnistToolStripMenuItem";
    this->mnistToolStripMenuItem->Size = System::Drawing::Size(55, 20);
    this->mnistToolStripMenuItem->Text = L"MNIST";
    //
    // loadMNISTToolStripMenuItem
    //
    this->loadMNISTToolStripMenuItem->Name = L"loadMNISTToolStripMenuItem";
    this->loadMNISTToolStripMenuItem->Size = System::Drawing::Size(237, 22);
    this->loadMNISTToolStripMenuItem->Text = L"Load MNIST Dataset";
    this->loadMNISTToolStripMenuItem->Click += gcnew System::EventHandler(
        this, &Form1::loadMNISTToolStripMenuItem_Click);
    //
    // trainMNISTToolStripMenuItem
    //
    this->trainMNISTToolStripMenuItem->Name = L"trainMNISTToolStripMenuItem";
    this->trainMNISTToolStripMenuItem->Size = System::Drawing::Size(237, 22);
    this->trainMNISTToolStripMenuItem->Text = L"Train MLP (784->128->64->10)";
    this->trainMNISTToolStripMenuItem->Click += gcnew System::EventHandler(
        this, &Form1::trainMNISTToolStripMenuItem_Click);
    //
    // testMNISTToolStripMenuItem
    //
    this->testMNISTToolStripMenuItem->Name = L"testMNISTToolStripMenuItem";
    this->testMNISTToolStripMenuItem->Size = System::Drawing::Size(237, 22);
    this->testMNISTToolStripMenuItem->Text = L"Test MLP";
    this->testMNISTToolStripMenuItem->Click += gcnew System::EventHandler(
        this, &Form1::testMNISTToolStripMenuItem_Click);
    //
    // trainAutoencoderToolStripMenuItem
    //
    this->trainAutoencoderToolStripMenuItem->Name =
        L"trainAutoencoderToolStripMenuItem";
    this->trainAutoencoderToolStripMenuItem->Size =
        System::Drawing::Size(237, 22);
    this->trainAutoencoderToolStripMenuItem->Text =
        L"Train Autoencoder + Classifier";
    this->trainAutoencoderToolStripMenuItem->Click +=
        gcnew System::EventHandler(
            this, &Form1::trainAutoencoderToolStripMenuItem_Click);
    //
    // testEncoderClassifierToolStripMenuItem
    //
    this->testEncoderClassifierToolStripMenuItem->Name =
        L"testEncoderClassifierToolStripMenuItem";
    this->testEncoderClassifierToolStripMenuItem->Size =
        System::Drawing::Size(237, 22);
    this->testEncoderClassifierToolStripMenuItem->Text =
        L"Test Encoder + Classifier";
    this->testEncoderClassifierToolStripMenuItem->Click +=
        gcnew System::EventHandler(
            this, &Form1::testEncoderClassifierToolStripMenuItem_Click);
    //
    // openFileDialog1
    //
    this->openFileDialog1->FileName = L"openFileDialog1";
    //
    // textBox1
    //
    this->textBox1->Location = System::Drawing::Point(1076, 135);
    this->textBox1->Multiline = true;
    this->textBox1->Name = L"textBox1";
    this->textBox1->ScrollBars = System::Windows::Forms::ScrollBars::Vertical;
    this->textBox1->Size = System::Drawing::Size(417, 318);
    this->textBox1->TabIndex = 5;
    //
    // saveFileDialog1
    //
    this->saveFileDialog1->FileName = L"saveFileDialog1";
    //
    // chart1
    //
    chartArea2->Name = L"ChartArea1";
    this->chart1->ChartAreas->Add(chartArea2);
    legend2->Name = L"Legend1";
    this->chart1->Legends->Add(legend2);
    this->chart1->Location = System::Drawing::Point(850, 470);
    this->chart1->Name = L"chart1";
    series2->ChartArea = L"ChartArea1";
    series2->Legend = L"Legend1";
    series2->Name = L"Series1";
    this->chart1->Series->Add(series2);
    this->chart1->Size = System::Drawing::Size(500, 300);
    this->chart1->TabIndex = 6;
    this->chart1->Text = L"chart1";
    //
    // lblMomentum
    //
    this->lblMomentum->AutoSize = true;
    this->lblMomentum->Location = System::Drawing::Point(857, 440);
    this->lblMomentum->Name = L"lblMomentum";
    this->lblMomentum->Size = System::Drawing::Size(62, 13);
    this->lblMomentum->TabIndex = 7;
    this->lblMomentum->Text = L"Momentum:";
    //
    // txtMomentum
    //
    this->txtMomentum->Location = System::Drawing::Point(925, 437);
    this->txtMomentum->Name = L"txtMomentum";
    this->txtMomentum->Size = System::Drawing::Size(50, 20);
    this->txtMomentum->TabIndex = 8;
    this->txtMomentum->Text = L"0,9";
    //
    // btnSetMomentum
    //
    this->btnSetMomentum->Location = System::Drawing::Point(985, 435);
    this->btnSetMomentum->Name = L"btnSetMomentum";
    this->btnSetMomentum->Size = System::Drawing::Size(50, 23);
    this->btnSetMomentum->TabIndex = 9;
    this->btnSetMomentum->Text = L"Set";
    this->btnSetMomentum->UseVisualStyleBackColor = true;
    this->btnSetMomentum->Click +=
        gcnew System::EventHandler(this, &Form1::btnSetMomentum_Click);
    //
    // folderBrowserDialog1
    //
    this->folderBrowserDialog1->Description =
        L"MNIST klasorunu secin (train ve test alt klasorleri icermeli)";
    //
    // Form1
    //
    this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
    this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
    this->ClientSize = System::Drawing::Size(1505, 781);
    this->Controls->Add(this->btnSetMomentum);
    this->Controls->Add(this->txtMomentum);
    this->Controls->Add(this->lblMomentum);
    this->Controls->Add(this->chart1);
    this->Controls->Add(this->textBox1);
    this->Controls->Add(this->label3);
    this->Controls->Add(this->groupBox2);
    this->Controls->Add(this->groupBox1);
    this->Controls->Add(this->pictureBox1);
    this->Controls->Add(this->menuStrip1);
    this->MainMenuStrip = this->menuStrip1;
    this->Name = L"Form1";
    this->Text = L"Form1";
    (cli::safe_cast<System::ComponentModel::ISupportInitialize ^>(
         this->pictureBox1))
        ->EndInit();
    this->groupBox1->ResumeLayout(false);
    this->groupBox1->PerformLayout();
    this->groupBox2->ResumeLayout(false);
    this->groupBox2->PerformLayout();
    this->menuStrip1->ResumeLayout(false);
    this->menuStrip1->PerformLayout();
    (cli::safe_cast<System::ComponentModel::ISupportInitialize ^>(this->chart1))
        ->EndInit();
    this->ResumeLayout(false);
    this->PerformLayout();
  }
  // Refactored LineCiz without lambdas
  void LineCiz(float x1, float y1, float x2, float y2, int width, int height,
               System::Drawing::Graphics ^ g) {
    int center_x = width / 2;
    int center_y = height / 2;

    if (Double::IsNaN(x1) || Double::IsNaN(y1) || Double::IsNaN(x2) ||
        Double::IsNaN(y2) || Double::IsInfinity(x1) || Double::IsInfinity(y1) ||
        Double::IsInfinity(x2) || Double::IsInfinity(y2)) {
      return;
    }

    // Inline clamping logic
    int px1 = center_x + (int)x1;
    if (px1 < -10000)
      px1 = -10000;
    if (px1 > width + 10000)
      px1 = width + 10000;

    int py1 = center_y - (int)y1;
    if (py1 < -10000)
      py1 = -10000;
    if (py1 > height + 10000)
      py1 = height + 10000;

    int px2 = center_x + (int)x2;
    if (px2 < -10000)
      px2 = -10000;
    if (px2 > width + 10000)
      px2 = width + 10000;

    int py2 = center_y - (int)y2;
    if (py2 < -10000)
      py2 = -10000;
    if (py2 > height + 10000)
      py2 = height + 10000;

    System::Drawing::Pen ^ pen =
        gcnew System::Drawing::Pen(System::Drawing::Color::Red, 2.0f);
    try {
      g->DrawLine(pen, px1, py1, px2, py2);
    } catch (...) {
    }
    delete pen;
  }

  void draw_sample(int temp_x, int temp_y, int label) {
    Pen ^ pen; // = gcnew Pen(Color::Black, 3.0f);
    switch (label) {
    case 0:
      pen = gcnew Pen(Color::Black, 3.0f);
      break;
    case 1:
      pen = gcnew Pen(Color::Red, 3.0f);
      break;
    case 2:
      pen = gcnew Pen(Color::Blue, 3.0f);
      break;
    case 3:
      pen = gcnew Pen(Color::Green, 3.0f);
      break;
    case 4:
      pen = gcnew Pen(Color::Yellow, 3.0f);
      break;
    case 5:
      pen = gcnew Pen(Color::Orange, 3.0f);
      break;
    default:
      pen = gcnew Pen(Color::YellowGreen, 3.0f);
    } // switch
    pictureBox1->CreateGraphics()->DrawLine(pen, temp_x - 5, temp_y, temp_x + 5,
                                            temp_y);
    pictureBox1->CreateGraphics()->DrawLine(pen, temp_x, temp_y - 5, temp_x,
                                            temp_y + 5);
  } // draw_sample
#pragma endregion
private:
  System::Void pictureBox1_MouseClick(System::Object ^ sender,
                                      System::Windows::Forms::MouseEventArgs ^
                                          e) {
    if (class_count == 0)
      MessageBox::Show("The Network Architeture should be firtly set up");
    else {
      float *x = new float[inputDim];
      int temp_x = (System::Convert::ToInt32(e->X));
      int temp_y = (System::Convert::ToInt32(e->Y));
      x[0] = float(temp_x - (pictureBox1->Width / 2));
      x[1] = float(pictureBox1->Height / 2 - temp_y);
      int label;
      int numLabel = Convert::ToInt32(ClassNoBox->Text);
      if (numLabel > class_count)
        MessageBox::Show("The class label cannot be greater than the maximum "
                         "number of classes.");
      else {
        label = numLabel - 1; // D�g�ler 0 dan ba�lad���ndan, label de�eri 0 dan
                              // ba�lamas� i�in bir eksi�i al�nm��t�r
        if (numSample == 0) { // Dinamik al�nan ilk �rnek i�in sadece
          numSample = 1;
          Samples = new float[numSample * inputDim];
          targets = new float[numSample];
          for (int i = 0; i < inputDim; i++)
            Samples[i] = x[i];
          targets[0] = float(label);
        } else {
          numSample++;
          Samples = Add_Data(Samples, numSample, x, inputDim);
          targets = Add_Labels(targets, numSample, label);
        } // else
        draw_sample(temp_x, temp_y, label);
        label3->Text = "Samples Count: " + System::Convert::ToString(numSample);
        delete[] x;
      } // else of if (Etiket ...
    } // else
  } // pictureMouseClick
private:
  System::Void pictureBox1_Paint(System::Object ^ sender,
                                 System::Windows::Forms::PaintEventArgs ^ e) {
    // Ana eksen do�rularini cizdir
    Pen ^ pen = gcnew Pen(Color::Black, 3.0f);
    int center_width, center_height;
    center_width = (int)(pictureBox1->Width / 2);
    center_height = (int)(pictureBox1->Height / 2);
    e->Graphics->DrawLine(pen, center_width, 0, center_width,
                          pictureBox1->Height);
    e->Graphics->DrawLine(pen, 0, center_height, pictureBox1->Width,
                          center_height);
  }

private:
  System::Void Set_Net_Click(System::Object ^ sender, System::EventArgs ^ e) {
    // Önceki ağ yapısını temizle
    if (Weights) {
      for (int i = 0; i < layer_count - 1; i++)
        delete[] Weights[i];
      delete[] Weights;
      Weights = nullptr;
    }
    if (bias) {
      for (int i = 0; i < layer_count - 1; i++)
        delete[] bias[i];
      delete[] bias;
      bias = nullptr;
    }
    if (topology) {
      delete[] topology;
      topology = nullptr;
    }

    // Yeni yapı
    class_count = Convert::ToInt32(ClassCountBox->Text);

    // Dynamic Layers Reading
    int hiddenCount =
        pnlLayers->Controls->Count / 2; // Each layer has Label + TextBox

    layer_count = 1 + hiddenCount + 1; // Input + Hiddens + Output
    topology = new int[layer_count];

    topology[0] = inputDim; // Input

    int layerIdx = 0;
    for each (Control ^ ctrl in pnlLayers->Controls) {
      if (dynamic_cast<TextBox ^>(ctrl)) { // Find TextBox
        try {
          int neurons = Convert::ToInt32(ctrl->Text);
          topology[layerIdx + 1] = neurons; // +1 because topology[0] is input
          layerIdx++;
        } catch (...) {
          MessageBox::Show("Lütfen tüm katmanlar için geçerli sayı girin.");
          return;
        }
      }
    }

    topology[layer_count - 1] = class_count; // Output

    // Ağırlık ve Bias tahsisi
    Weights = new float *[layer_count - 1];
    bias = new float *[layer_count - 1]; // Input katmanı hariç her katman için
                                         // bias (veya bağlantı sayısı kadar)

    // Random başlatma ve allocation
    for (int l = 0; l < layer_count - 1; l++) {
      int src = topology[l];
      int dest = topology[l + 1];

      Weights[l] = init_array_random(dest * src);
      bias[l] = init_array_random(dest);
    }

    Set_Net->Text =
        " MLP Ready (" + Convert::ToString(layer_count) + " Layers)";
  } // Set_Net

private:
  System::Void cbLayerCount_SelectedIndexChanged(System::Object ^ sender,
                                                 System::EventArgs ^ e) {
    pnlLayers->Controls->Clear();
    int count = Convert::ToInt32(cbLayerCount->Text);

    for (int i = 0; i < count; i++) {
      // Label
      Label ^ lbl = gcnew Label();
      lbl->Text = "Katman " + (i + 1) + " Nöron:";
      lbl->AutoSize = true;
      lbl->Margin = System::Windows::Forms::Padding(3, 5, 3, 0); // Spacing

      // TextBox
      TextBox ^ txt = gcnew TextBox();
      txt->Text = "4"; // Default value
      txt->Width = 50;

      pnlLayers->Controls->Add(lbl);
      pnlLayers->Controls->Add(txt);

      // Add a line break logic for FlowLayoutPanel?
      // FlowLayoutPanel with FlowDirection::TopDown or wrapping handles it if
      // sizes are right. Or simpler: use FlowDirection::LeftToRight (default)
      // but add explicit breaks? Let's rely on FlowDirection::LeftToRight and
      // control widths.
      pnlLayers->SetFlowBreak(txt, true); // Force new line after TextBox
    }
  }

private:
  System::Void readDataToolStripMenuItem_Click(System::Object ^ sender,
                                               System::EventArgs ^ e) {
    MessageBox::Show("Read Data is momentarily disabled for MLP upgrade.");
    /*
  char **c = new char *[2];
  // Veri Kmesini okunacak
  MessageBox::Show("Veri Kmesini Ykleyin");
  c[0] = "../Data/Samples.txt";
  c[1] = "../Data/weights.txt";
  std::ifstream file;
  int num, w, h, Dim, label;
  file.open(c[0]);
  if (file.is_open()) {
    // MessageBox::Show("Dosya acildi");
    file >> Dim >> w >> h >> num;
    textBox1->Text += "Dimension: " + Convert::ToString(Dim) +
                      "- Width: " + Convert::ToString(w) +
                      " - Height: " + Convert::ToString(h) +
                      " - Number of Class: " + Convert::ToString(num) +
                      "\r\n";
    // Set network values
    class_count = num;
    inputDim = Dim;
    Weights = new float[class_count * inputDim];
    bias = new float[class_count];
    numSample = 0;
    float *x = new float[inputDim];
    while (!file.eof()) {
      if (numSample == 0) { // ilk rnek iin sadece
        numSample = 1;
        Samples = new float[inputDim];
        targets = new float[numSample];
        for (int i = 0; i < inputDim; i++)
          file >> Samples[i];
        file >> targets[0];
      } else {

        for (int i = 0; i < inputDim; i++)
          file >> x[i];
        file >> label;
        if (!file.eof()) {
          numSample++;
          Samples = Add_Data(Samples, numSample, x, inputDim);
          targets = Add_Labels(targets, numSample, label);
        }
      } // else
    } // while
    delete[] x;
    file.close();
    for (int i = 0; i < numSample; i++) {
      draw_sample(Samples[i * inputDim] + w, h - Samples[i * inputDim + 1],
                  targets[i]);
      for (int j = 0; j < inputDim; j++)
        textBox1->Text += Convert::ToString(Samples[i * inputDim + j]) + " ";
      textBox1->Text += Convert::ToString(targets[i]) + "\r\n";
    }
    // draw_sample(temp_x, temp_y, label);
    label3->Text = "Samples Count: " + System::Convert::ToString(numSample);
    MessageBox::Show("Dosya basari ile okundu");
  } // file.is_open
  else
    MessageBox::Show("Dosya acilamadi");
  // Get weights
  int Layer;
  file.open(c[1]);
  if (file.is_open()) {
    file >> Layer >> Dim >> num;
    class_count = num;
    inputDim = Dim;
    Weights = new float[class_count * inputDim];
    bias = new float[class_count];
    textBox1->Text += "Layer: " + Convert::ToString(Layer) +
                      " Dimension: " + Convert::ToString(Dim) +
                      " numClass:" + Convert::ToString(num) + "\r\n";
    while (!file.eof()) {
      for (int i = 0; i < class_count; i++)
        for (int j = 0; j < inputDim; j++)
          file >> Weights[i * inputDim + j];
      for (int i = 0; i < class_count; i++)
        file >> bias[i];
    }
    file.close();
  } // file.is_open
  delete[] c;
    */
  } // Read_Data
private:
  System::Void saveDataToolStripMenuItem_Click(System::Object ^ sender,
                                               System::EventArgs ^ e) {
    MessageBox::Show("Save Data is momentarily disabled for MLP upgrade.");
    /*
  if (numSample != 0) {
    char **c = new char *[2];
    // Veri Kmesi yazlacak
    c[0] = "../Data/Samples.txt";
    c[1] = "../Data/weights.txt";
    std::ofstream ofs(c[0]);
    if (!ofs.bad()) {
      // Width,  Height, number of Class, data+label
      ofs << inputDim << " " << pictureBox1->Width / 2 << " "
          << pictureBox1->Height / 2 << " " << class_count << std::endl;
      for (int i = 0; i < numSample; i++) {
        for (int d = 0; d < inputDim; d++)
          ofs << Samples[i * inputDim + d] << " ";
        ofs << targets[i] << std::endl;
      }
      ofs.close();
    } else
      MessageBox::Show("Samples icin dosya acilamadi");
    std::ofstream file(c[1]);
    if (!file.bad()) {
      // #Layer Dimension numClass weights biases
      file << 1 << " " << inputDim << " " << class_count << std::endl;
      for (int k = 0; k < class_count * inputDim; k++)
        file << Weights[k] << " ";
      file << std::endl;
      for (int k = 0; k < class_count; k++)
        file << bias[k] << " ";
      file.close();
    } else
      MessageBox::Show("Weight icin dosya acilamadi");
    delete[] c;
  } else
    MessageBox::Show("At least one sample should be given");
    */
  } // Save_Data
private:
  System::Void testingToolStripMenuItem_Click(System::Object ^ sender,
                                              System::EventArgs ^ e) {
    float *x = new float[2];
    float *mean = new float[2];
    float *std = new float[2];
    // mean ve std tekrar hesaplan�yor, Asl�nda e�itimde bunlar saklanmal�
    // oradan al�nmal�
    Z_Score_Parameters(Samples, numSample, inputDim, mean, std);
    // MessageBox::Show("mean: "+System::Convert::ToString(mean[0])+ " "+
    // System::Convert::ToString(mean[1]));
    int num, temp_x, temp_y;
    Bitmap ^ surface = gcnew Bitmap(pictureBox1->Width, pictureBox1->Height);
    pictureBox1->Image = surface;
    Color c;
    for (int row = 0; row < pictureBox1->Height; row += 2) {
      for (int column = 0; column < pictureBox1->Width; column += 2) {
        x[0] = (float)(column - (pictureBox1->Width / 2));
        x[1] = (float)((pictureBox1->Height / 2) - row);
        x[0] = (x[0] - mean[0]) / std[0];
        x[1] = (x[1] - mean[1]) / std[1];
        // MLP Test Forward
        num = Test_Forward(x, Weights, bias, topology, layer_count);
        // MessageBox::Show("merhaba: class :" +
        // System::Convert::ToString(numClass));
        switch (num) {
        case 0:
          c = Color::FromArgb(0, 0, 0);
          break;
        case 1:
          c = Color::FromArgb(255, 0, 0);
          break;
        case 2:
          c = Color::FromArgb(0, 255, 0);
          break;
        case 3:
          c = Color::FromArgb(0, 0, 255);
          break;
        default:
          c = Color::FromArgb(0, 255, 255);
        } // switch
        surface->SetPixel(column, row, c);
      } // column
      // MessageBox::Show("merhaba2: class :" +
      // System::Convert::ToString(numClass));
    }
    // Samples Draw
    Pen ^ pen; // = gcnew Pen(Color::Black, 3.0f);
    MessageBox::Show("�rnekler cizilecek");
    for (int i = 0; i < numSample; i++) {
      switch (int(targets[i])) {
      case 0:
        pen = gcnew Pen(Color::Black, 3.0f);
        break;
      case 1:
        pen = gcnew Pen(Color::Red, 3.0f);
        break;
      case 2:
        pen = gcnew Pen(Color::Blue, 3.0f);
        break;
      case 3:
        pen = gcnew Pen(Color::Green, 3.0f);
        break;
      case 4:
        pen = gcnew Pen(Color::Yellow, 3.0f);
        break;
      case 5:
        pen = gcnew Pen(Color::Orange, 3.0f);
        break;
      default:
        pen = gcnew Pen(Color::YellowGreen, 3.0f);
      } // switch
      temp_x = int(Samples[i * inputDim]) + pictureBox1->Width / 2;
      temp_y = pictureBox1->Height / 2 - int(Samples[i * inputDim + 1]);
      pictureBox1->CreateGraphics()->DrawLine(pen, temp_x - 5, temp_y,
                                              temp_x + 5, temp_y);
      pictureBox1->CreateGraphics()->DrawLine(pen, temp_x, temp_y - 5, temp_x,
                                              temp_y + 5);
    }
    delete[] x;
    delete[] mean;
    delete[] std;
  } // Testing
private:
  System::Void trainingToolStripMenuItem_Click(System::Object ^ sender,
                                               System::EventArgs ^ e) {
    if (numSample == 0) {
      MessageBox::Show("Önce veri toplamanız gerekiyor!");
      return;
    }
    if (class_count == 0) {
      MessageBox::Show("Önce ağ yapısını ayarlamanız gerekiyor!");
      return;
    }

    int Max_epoch = 10000; // Increased even more
    float Min_Err = 0.001f;
    float learning_rate = 0.01f;
    float momentum = 0.9f; // Momentum Coefficient
    try {
      momentum = (float)System::Convert::ToDouble(txtMomentum->Text);
    } catch (...) {
      momentum = 0.9f;
    }
    int epoch = 0;

    // Verileri normalize et (test fonksiyonu ile tutarlılık için)
    float *normalizedSamples = Z_Score_Norm(Samples, numSample, inputDim);

    float *error_history = train_fcn(
        normalizedSamples, numSample, targets, topology, layer_count, Weights,
        bias, learning_rate, momentum, Min_Err, Max_epoch, epoch);

    // Normalize edilmiş veriyi temizle
    delete[] normalizedSamples;

    if (epoch > 0) {
      int lastEpoch =
          epoch - 1; // epoch zaten artırılmış, son geçerli indeks epoch-1
      System::Diagnostics::Debug::WriteLine(
          "Epoch: " + Convert::ToString(epoch) +
          " Error: " + Convert::ToString(error_history[lastEpoch]));
      textBox1->Text +=
          "Eğitim tamamlandı! Epoch: " + Convert::ToString(epoch) + "\r\n";
      textBox1->Text +=
          "Son hata: " + Convert::ToString(error_history[lastEpoch]) + "\r\n";
    }

    chart1->Series["Series1"]->Points->Clear();
    chart1->Series["Series1"]->ChartType = System::Windows::Forms::
        DataVisualization::Charting::SeriesChartType::Column;
    for (int i = 0; i < epoch; i++) {
      chart1->Series["Series1"]->Points->AddY(error_history[i]);
    }

    delete[] error_history;
  }

private:
  System::Void regressionToolStripMenuItem_Click(System::Object ^ sender,
                                                 System::EventArgs ^ e) {
    if (numSample == 0) {
      MessageBox::Show("Önce veri toplamanız gerekiyor!");
      return;
    }

    // --- 1. Veri Hazırlığı (Regression için) ---
    // x: Input, y: Target
    float *regSamples = new float[numSample]; // Input (x)
    float *regTargets = new float[numSample]; // Target (y)

    // Normalize Parameters
    float meanX = 0, stdX = 0;
    float meanY = 0, stdY = 0;

    for (int i = 0; i < numSample; i++) {
      regSamples[i] = Samples[i * inputDim];     // x
      regTargets[i] = Samples[i * inputDim + 1]; // y
      meanX += regSamples[i];
      meanY += regTargets[i];
    }
    meanX /= numSample;
    meanY /= numSample;

    for (int i = 0; i < numSample; i++) {
      stdX += (regSamples[i] - meanX) * (regSamples[i] - meanX);
      stdY += (regTargets[i] - meanY) * (regTargets[i] - meanY);
    }
    stdX = sqrt(stdX / numSample);
    stdY = sqrt(stdY / numSample);
    if (stdX == 0)
      stdX = 1;
    if (stdY == 0)
      stdY = 1;

    // Normalize Data
    float *normRegSamples = new float[numSample];
    float *normRegTargets = new float[numSample];

    for (int i = 0; i < numSample; i++) {
      normRegSamples[i] = (regSamples[i] - meanX) / stdX;
      normRegTargets[i] = (regTargets[i] - meanY) / stdY;
    }

    // --- 2. MLP Yapılandırması (Topology) ---
    // Topology: [1, Hidden1, Hidden2, ..., 1]

    // UI'dan hidden layer bilgisini al
    // UI'dan hidden layer bilgisini al
    // Dynamic Layers Reading
    int hiddenCount =
        pnlLayers->Controls->Count / 2; // Each layer has Label + TextBox

    int regLayerCount = 1 + hiddenCount + 1; // Input + Hiddens + Output
    int *regTopology = new int[regLayerCount];

    regTopology[0] = 1; // Input Dim for Regression is 1

    int layerIdx = 0;
    for each (Control ^ ctrl in pnlLayers->Controls) {
      if (dynamic_cast<TextBox ^>(ctrl)) { // Find TextBox
        try {
          int neurons = Convert::ToInt32(ctrl->Text);
          regTopology[layerIdx + 1] =
              neurons; // +1 because topology[0] is input
          layerIdx++;
        } catch (...) {
          MessageBox::Show("Lütfen tüm katmanlar için geçerli sayı girin.");
          return;
        }
      }
    }

    // Output Dim for Regression is 1
    // Note: The original code might assume class_count? But for regression we
    // usually do 1 output. Let's assume 1.
    regTopology[regLayerCount - 1] = 1;

    // --- 3. Ağırlık Allocation ---
    float **regWeights = new float *[regLayerCount - 1];
    float **regBiases = new float *[regLayerCount - 1];

    for (int l = 0; l < regLayerCount - 1; l++) {
      int src = regTopology[l];
      int dest = regTopology[l + 1];
      // Random init
      regWeights[l] = new float[src * dest];
      regBiases[l] = new float[dest];
      for (int k = 0; k < src * dest; k++)
        regWeights[l][k] = ((float)rand() / RAND_MAX) - 0.5f;
      for (int k = 0; k < dest; k++)
        regBiases[l][k] = ((float)rand() / RAND_MAX) - 0.5f;
    }

    // --- 4. Eğitim ---
    // --- 4. Eğitim ---
    int Max_epoch = 5000; // Increased to improve accuracy
    float Min_Err = 0.0001f;
    float learning_rate = 0.01f;
    float momentum = 0.9f;
    int epoch = 0;

    float *error_history = train_mlp_regression(
        normRegSamples, numSample, normRegTargets, regTopology, regLayerCount,
        regWeights, regBiases, learning_rate, momentum, Min_Err, Max_epoch,
        epoch);

    // --- 5. Raporlama ---
    if (epoch > 0) {
      int lastEpoch = epoch - 1;
      textBox1->Text +=
          "MLP Regression Tamamlandı! Epoch: " + Convert::ToString(epoch) +
          "\r\n";
      textBox1->Text +=
          "Son Ortalama Hata: " + Convert::ToString(error_history[lastEpoch]) +
          "\r\n";
    }

    // --- 6. Çizim (Curve Fitting) ---
    // Picture box'ı temizle
    Bitmap ^ surface = gcnew Bitmap(pictureBox1->Width, pictureBox1->Height);
    Graphics ^ g = Graphics::FromImage(surface);
    g->Clear(Color::White);

    // Eksenleri çiz
    Pen ^ axisPen = gcnew Pen(Color::Black, 2.0f);
    int center_x = pictureBox1->Width / 2;
    int center_y = pictureBox1->Height / 2;
    g->DrawLine(axisPen, center_x, 0, center_x, pictureBox1->Height);
    g->DrawLine(axisPen, 0, center_y, pictureBox1->Width, center_y);
    delete axisPen;

    // Veri noktalarını çiz (Orijinal)
    for (int i = 0; i < numSample; i++) {
      // RegSamples[i] aslında merkezden uzaklık (x), RegTargets[i] (y)
      // Ancak bizim Samples dizimiz zaten merkezden uzaklık şeklinde
      // kaydediliyor olması lazım pictureBox1_MouseClick fonksiyonuna bakarak:
      // x[0] = temp_x - width/2; x[1] = height/2 - temp_y;
      // Evet, Samples değerleri merkez orijinli.

      int px = center_x + (int)regSamples[i];
      int py = center_y - (int)regTargets[i];
      g->FillEllipse(gcnew SolidBrush(Color::Blue), px - 3, py - 3, 6, 6);
    }

    // Prediction Curve Çiz
    Pen ^ curvePen = gcnew Pen(Color::Red, 2.0f);

    // Ekranın solundan sağına kadar her piksel için tahmin yap
    for (int px = 0; px < pictureBox1->Width - 1; px += 2) {
      float x_real1 = (float)(px - center_x);
      float x_real2 = (float)((px + 2) - center_x);

      // Normalize Input
      float x_norm1 = (x_real1 - meanX) / stdX;
      float x_norm2 = (x_real2 - meanX) / stdX;

      // Predict
      float y_norm1 = Evaluate_Regression_Point(x_norm1, regWeights, regBiases,
                                                regTopology, regLayerCount);
      float y_norm2 = Evaluate_Regression_Point(x_norm2, regWeights, regBiases,
                                                regTopology, regLayerCount);

      // Denormalize Output
      float y_real1 = y_norm1 * stdY + meanY;
      float y_real2 = y_norm2 * stdY + meanY;

      // Screen Coordinates
      int py1 = center_y - (int)y_real1;
      int py2 = center_y - (int)y_real2;

      // Clip to bounds to avoid crash (Graphics handles clip usually but good
      // practice)
      g->DrawLine(curvePen, px, py1, px + 2, py2);
    }
    delete curvePen;
    delete g;
    pictureBox1->Image = surface;

    // Grafik (Error History)
    chart1->Series["Series1"]->Points->Clear();
    chart1->Series["Series1"]->ChartType = System::Windows::Forms::
        DataVisualization::Charting::SeriesChartType::Line;
    for (int i = 0; i < epoch; i++) {
      chart1->Series["Series1"]->Points->AddY(error_history[i]);
    }

    // Cleanup
    delete[] error_history;
    for (int l = 0; l < regLayerCount - 1; l++) {
      delete[] regWeights[l];
      delete[] regBiases[l];
    }
    delete[] regWeights;
    delete[] regBiases;
    delete[] regTopology;

    delete[] regSamples;
    delete[] regTargets;
    delete[] normRegSamples;
    delete[] normRegTargets;

    MessageBox::Show("MLP Regresyon tamamlandı!");
  }

  void btnSetMomentum_Click(System::Object ^ sender, System::EventArgs ^ e) {
    try {
      float val = (float)System::Convert::ToDouble(txtMomentum->Text);
      if (val < 0.0f || val >= 1.0f) {
        MessageBox::Show(
            "Momentum value must be between 0.0 and 1.0 (exclusive).");
        return;
      }
      this->currentMomentum = val;
      MessageBox::Show("Momentum set to: " + val);
    } catch (...) {
      MessageBox::Show("Invalid momentum value!");
    }
  }

private:
  System::Windows::Forms::Label ^ lblMomentum;
  System::Windows::Forms::TextBox ^ txtMomentum;
  System::Windows::Forms::Button ^ btnSetMomentum;
  float currentMomentum;

  // MNIST Manager (designer bölgesinin dışında)
  MNIST_Manager ^ mnistManager;

  // ========== MNIST EVENT HANDLERS ==========
private:
  System::Void loadMNISTToolStripMenuItem_Click(System::Object ^ sender,
                                                System::EventArgs ^ e) {
    if (folderBrowserDialog1->ShowDialog() ==
        System::Windows::Forms::DialogResult::OK) {
      String ^ path = folderBrowserDialog1->SelectedPath;

      // 1000 train (100 per digit), 100 test (10 per digit)
      bool success = mnistManager->LoadMNIST(path, 100, 10);

      if (success) {
        int trainCount = mnistManager->GetTrainCount();
        int testCount = mnistManager->GetTestCount();
        MessageBox::Show("MNIST Loaded!\nTrain: " + trainCount +
                         " samples\nTest: " + testCount + " samples");
        textBox1->Text = "MNIST veri seti yuklendu.\r\nEgitim: " + trainCount +
                         " ornek\r\nTest: " + testCount + " ornek\r\n";
      } else {
        MessageBox::Show(
            "MNIST yukleme basarisiz! Klasor yapisi kontrol edin:\n" + path +
            "\\train\\0-9\n" + path + "\\test\\0-9");
      }
    }
  }

  System::Void trainMNISTToolStripMenuItem_Click(System::Object ^ sender,
                                                 System::EventArgs ^ e) {
    if (!mnistManager->dataLoaded) {
      MessageBox::Show("Once MNIST veri setini yukleyin!");
      return;
    }

    float learningRate = 0.005f; // Düşük LR - daha stabil öğrenme
    float momentum = currentMomentum;
    float minError = 0.01f; // Düşük hata toleransı
    int maxEpoch = 50;      // Daha fazla epoch
    int finalEpoch = 0;

    textBox1->Text += "\r\n========================================\r\n";
    textBox1->Text +=
        "MLP Egitimi basliyor...\r\nLearning Rate: " + learningRate +
        "\r\nMomentum: " + momentum + "\r\nMax Epoch: " + maxEpoch + "\r\n";
    textBox1->Text += "Lutfen bekleyin, bu islem biraz zaman alabilir...\r\n";
    this->Refresh();
    Application::DoEvents();

    // Başlangıç zamanını kaydet
    System::DateTime startTime = System::DateTime::Now;

    float *errorHistory = mnistManager->TrainMLP(
        learningRate, momentum, minError, maxEpoch, finalEpoch);

    // Geçen süreyi hesapla
    System::TimeSpan elapsed = System::DateTime::Now - startTime;

    if (errorHistory) {
      // Error grafiğini çiz
      chart1->Series["Series1"]->Points->Clear();
      chart1->Series["Series1"]->ChartType = System::Windows::Forms::
          DataVisualization::Charting::SeriesChartType::Line;

      for (int i = 0; i < finalEpoch; i++) {
        chart1->Series["Series1"]->Points->AddY(errorHistory[i]);
      }

      textBox1->Text += "Egitim tamamlandi! Epoch: " + finalEpoch + "\r\n";
      textBox1->Text += "Son hata: " + errorHistory[finalEpoch - 1] + "\r\n";
      textBox1->Text +=
          "Sure: " + elapsed.TotalSeconds.ToString("F1") + " saniye\r\n";

      delete[] errorHistory;

      MessageBox::Show("MLP Egitimi tamamlandi!\nEpoch: " + finalEpoch +
                       "\nSure: " + elapsed.TotalSeconds.ToString("F1") +
                       " saniye");
    } else {
      MessageBox::Show("Egitim basarisiz! Veri yuklendiginden emin olun.");
    }
  }

  System::Void testMNISTToolStripMenuItem_Click(System::Object ^ sender,
                                                System::EventArgs ^ e) {
    if (!mnistManager->mlpTrained) {
      MessageBox::Show("Once MLP'yi egitin!");
      return;
    }

    int confMatrix[10][10];
    float accuracy = mnistManager->TestMLP(confMatrix);

    textBox1->Text += "\r\n========================================\r\n";
    textBox1->Text += "=== MLP CONFUSION MATRIX ===\r\n";
    textBox1->Text +=
        "       0   1   2   3   4   5   6   7   8   9  <- Predicted\r\n";
    textBox1->Text += "    --------------------------------------------\r\n";

    for (int i = 0; i < 10; i++) {
      String ^ row = i + " |";
      for (int j = 0; j < 10; j++) {
        row += confMatrix[i][j].ToString()->PadLeft(4);
      }
      textBox1->Text += row + "\r\n";
    }
    textBox1->Text += "^\r\nActual\r\n\r\n";
    textBox1->Text += "MLP Test Accuracy: %" + accuracy.ToString("F2") + "\r\n";

    MessageBox::Show("MLP Test Accuracy: %" + accuracy.ToString("F2"));
  }

  System::Void trainAutoencoderToolStripMenuItem_Click(System::Object ^ sender,
                                                       System::EventArgs ^ e) {
    if (!mnistManager->dataLoaded) {
      MessageBox::Show("Once MNIST veri setini yukleyin!");
      return;
    }

    float learningRate = 0.002f; // Düşük (stabil öğrenme)
    float momentum = currentMomentum;
    float minError = 0.01f; // Düşük hata toleransı
    int maxEpoch = 50;      // 50 epoch
    int finalEpoch = 0;

    textBox1->Text += "\r\n========================================\r\n";
    textBox1->Text +=
        "Autoencoder Egitimi basliyor...\r\nMimari: 784->256->32->256->784\r\n";
    textBox1->Text += "Lutfen bekleyin (bu islem cok uzun surebilir)...\r\n";
    this->Refresh();
    Application::DoEvents();

    System::DateTime startTime = System::DateTime::Now;

    float *errorHistory = mnistManager->TrainAutoencoder(
        learningRate, momentum, minError, maxEpoch, finalEpoch);

    System::TimeSpan elapsed = System::DateTime::Now - startTime;

    if (errorHistory) {
      chart1->Series["Series1"]->Points->Clear();
      chart1->Series["Series1"]->ChartType = System::Windows::Forms::
          DataVisualization::Charting::SeriesChartType::Line;

      for (int i = 0; i < finalEpoch; i++) {
        chart1->Series["Series1"]->Points->AddY(errorHistory[i]);
      }

      textBox1->Text +=
          "Autoencoder egitimi tamamlandi! Epoch: " + finalEpoch + "\r\n";
      textBox1->Text +=
          "Sure: " + elapsed.TotalSeconds.ToString("F1") + " saniye\r\n";
      delete[] errorHistory;
    }

    // Encoder + Classifier eğitimi
    textBox1->Text += "\r\nEncoder + Classifier egitimi basliyor...\r\n";
    this->Refresh();
    Application::DoEvents();

    startTime = System::DateTime::Now;
    learningRate = 0.005f;
    maxEpoch = 50; // Daha fazla epoch

    float *encErrorHistory = mnistManager->TrainEncoderClassifier(
        learningRate, momentum, minError, maxEpoch, finalEpoch);

    elapsed = System::DateTime::Now - startTime;

    if (encErrorHistory) {
      chart1->Series["Series1"]->Points->Clear();

      for (int i = 0; i < finalEpoch; i++) {
        chart1->Series["Series1"]->Points->AddY(encErrorHistory[i]);
      }

      textBox1->Text +=
          "Encoder + Classifier egitimi tamamlandi! Epoch: " + finalEpoch +
          "\r\n";
      textBox1->Text +=
          "Sure: " + elapsed.TotalSeconds.ToString("F1") + " saniye\r\n";
      delete[] encErrorHistory;

      MessageBox::Show("Autoencoder ve Encoder+Classifier egitimi tamamlandi!");
    }
  }

  System::Void
  testEncoderClassifierToolStripMenuItem_Click(System::Object ^ sender,
                                               System::EventArgs ^ e) {
    if (!mnistManager->encClassifierTrained) {
      MessageBox::Show("Once Autoencoder ve Encoder+Classifier'i egitin!");
      return;
    }

    int confMatrix[10][10];
    float accuracy = mnistManager->TestEncoderClassifier(confMatrix);

    textBox1->Text += "\r\n========================================\r\n";
    textBox1->Text += "=== ENCODER+CLASSIFIER CONFUSION MATRIX ===\r\n";
    textBox1->Text +=
        "       0   1   2   3   4   5   6   7   8   9  <- Predicted\r\n";
    textBox1->Text += "    --------------------------------------------\r\n";

    for (int i = 0; i < 10; i++) {
      String ^ row = i + " |";
      for (int j = 0; j < 10; j++) {
        row += confMatrix[i][j].ToString()->PadLeft(4);
      }
      textBox1->Text += row + "\r\n";
    }
    textBox1->Text += "^\r\nActual\r\n\r\n";
    textBox1->Text += "Encoder + Classifier Test Accuracy: %" +
                      accuracy.ToString("F2") + "\r\n";

    MessageBox::Show("Encoder + Classifier Test Accuracy: %" +
                     accuracy.ToString("F2"));
  }
};
} // namespace CppCLRWinformsProjekt
