// Copyright 2026 Maree Carroll
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

static cv::Mat softmax(const cv::Mat& logits) {
    // logits: 1x2
    cv::Mat maxVals;
    cv::reduce(logits, maxVals, 1, cv::REDUCE_MAX);

    cv::Mat shifted = logits - cv::repeat(maxVals, 1, logits.cols);

    cv::Mat expVals;
    cv::exp(shifted, expVals);

    cv::Mat sumExp;
    cv::reduce(expVals, sumExp, 1, cv::REDUCE_SUM);

    cv::Mat probs = expVals / cv::repeat(sumExp, 1, expVals.cols);
    return probs;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cout << "Usage: ./infer model.onnx image.jpg\n";
    return 1;
  }

  std::string modelPath = argv[1];
  std::string imagePath = argv[2];

  // Load ONNX model
  cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
  if (net.empty()) {
    std::cerr << "Error: Could not load model: " << modelPath << "\n";
    return 1;
  }

  // Load image
  cv::Mat img = cv::imread(imagePath);
  if (img.empty()) {
    std::cerr << "Error: Could not load image: " << imagePath << "\n";
    return 1;
  }

  // Preprocessing: resize → normalize → convert to blob
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(224, 224));

  cv::Mat blob = cv::dnn::blobFromImage(
      resized,
      1.0 / 255.0,                              // scale factor
      cv::Size(224, 224), cv::Scalar(0, 0, 0),  // no mean subtraction
      true,                                     // swap RB
      false);                                   // do not crop

  // Run inference
  net.setInput(blob);
  cv::Mat output = net.forward();

  // Apply softmax
  cv::Mat probs = softmax(output);

  float p_not_tawny = probs.at<float>(0, 0);
  float p_is_tawny = probs.at<float>(0, 1);

  std::string predicted = (p_is_tawny > p_not_tawny) ? "is_tawny" : "not_tawny";

  // Print results
  std::cout << "Prediction: " << predicted << "\n";
  std::cout << "  P(not_tawny) = " << p_not_tawny << "\n";
  std::cout << "  P(is_tawny)  = " << p_is_tawny << "\n";

  return 0;
}
