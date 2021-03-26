/* 
 * Copyright (c) 2021 Michael Gruner <michael.gruner@ridgerun.com>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following
 * disclaimer in the documentation and/or other materials provided
 * with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived
 * from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <string>

#include <opencv2/core.hpp>         // Basic OpenCV structures
#include <opencv2/imgproc.hpp>      // Image processing methods for the CPU
#include <opencv2/imgcodecs.hpp>    // Images IO

static void
sobel(const cv::Mat &input, cv::Mat &output) {
  // Lowpass filter to clean noise
  cv::Mat blurred;
  cv::GaussianBlur(input, blurred, cv::Size(7, 7), -1);

  // Compute X and Y derivatives
  cv::Mat x, y;
  cv::Sobel(blurred, x, CV_32F, 1, 0, 3, 1, 0);
  cv::Sobel(blurred, y, CV_32F, 0, 1, 3, 1, 0);

  // Compute X^2 and Y^2
  cv::Mat x2, y2;
  cv::pow(x, 2, x2);
  cv::pow(y, 2, y2);

  // Compute MAG2 = X^2 + Y^2
  cv::Mat mag2;
  cv::addWeighted(x2, 1, y2, 1, 0, mag2);

  // Compute MAG = √ (X^2 + Y^2)
  cv::Mat mag;
  cv::sqrt(mag2, mag);

  // Convert from floating point to char
  cv::convertScaleAbs(mag, output);
}

int
main(int argc, char *argv[]) {
  std::string to_read = "dog.jpg";
  if (argc >= 2) {
    to_read = argv[1];
  }

  std::string to_write = "dog_gradient_cpu.jpg";
  if (argc >= 3) {
    to_write = argv[2];
  }

  cv::Mat input = cv::imread(to_read, cv::IMREAD_GRAYSCALE);

  if (input.empty()) {
    std::cerr << "Unable to find \"" << to_read << "\". Is the path ok?"
              << std::endl;
    return 1;
  }

  // The first call is typically a warmup call so we dont benchmark
  cv::Mat output;
  sobel(input, output);

  int N = 100;
  double time = cv::getTickCount();

  std::cout << "Performing " << N << " iterations..." << std::flush;

  for (int i = 0; i < N; i++) {
    sobel(input, output);
  }

  time = 1000.0*(cv::getTickCount() - time)/cv::getTickFrequency();
  time /= N;

  std::cout << " done!" << std::endl << "Average for " << N << " CPU runs: "
            << time << "ms" << std::endl;

  std::cout << "Resulting image wrote to " << to_write << std::endl;
  cv::imwrite(to_write, output);

  return 0;
}
