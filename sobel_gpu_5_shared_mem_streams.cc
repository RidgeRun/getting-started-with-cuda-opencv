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

#include <opencv2/cudaarithm.hpp>   // CUDA matrix operations
#include <opencv2/cudafilters.hpp>  // CUDA image filters

struct Filters {
  const cv::Ptr<cv::cuda::Filter> gaussian;
  const cv::Ptr<cv::cuda::Filter> sobelx;
  const cv::Ptr<cv::cuda::Filter> sobely;
};

struct GpuMemories {
  cv::cuda::GpuMat input;
  cv::cuda::GpuMat blurred;
  cv::cuda::GpuMat x;
  cv::cuda::GpuMat y;
  cv::cuda::GpuMat x2;
  cv::cuda::GpuMat y2;
  cv::cuda::GpuMat mag2;
  cv::cuda::GpuMat mag;
  cv::cuda::GpuMat output;
};

struct Streams {
  cv::cuda::Stream x;
  cv::cuda::Stream y;
};

static void
sobel(const Filters &filters, Streams &streams, GpuMemories &gpu) {
  // Migrating data from the CPU to the GPU is no longer needed

  // Low pass filter to clean noise
  filters.gaussian->apply(gpu.input, gpu.blurred, streams.x);

  streams.x.waitForCompletion();

  // X and Y derivatives
  filters.sobelx->apply(gpu.blurred, gpu.x, streams.x);
  filters.sobely->apply(gpu.blurred, gpu.y, streams.y);

  // X^2 and Y^2
  cv::cuda::pow(gpu.x, 2, gpu.x2, streams.x);
  cv::cuda::pow(gpu.y, 2, gpu.y2, streams.y);

  streams.y.waitForCompletion();

  // MAG2 = X^2 + Y^2
  cv::cuda::addWeighted(gpu.x2, 1, gpu.y2, 1, 0, gpu.mag2, -1, streams.x);

  // MAG = √(X^2 + Y^2)
  cv::cuda::sqrt(gpu.mag2, gpu.mag, streams.x);

  // Convert from floating point to char
  gpu.mag.convertTo(gpu.output, CV_8UC1, streams.x);

  // Migrating data back from GPU to CPU is no longer needed
}

int
main(int argc, char *argv[]) {
  std::string to_read = "dog.jpg";
  if (argc >= 2) {
    to_read = argv[1];
  }

  std::string to_write = "dog_gradient_gpu_5_shared_mem_streams.jpg";
  if (argc >= 3) {
    to_write = argv[2];
  }

  cv::Mat input = cv::imread(to_read, cv::IMREAD_GRAYSCALE);

  if (input.empty()) {
    std::cerr << "Unable to find \"" << to_read << "\". Is the path ok?"
              << std::endl;
    return 1;
  }

  cv::cuda::HostMem shared_input(input, cv::cuda::HostMem::AllocType::SHARED);
  cv::cuda::HostMem shared_output(input, cv::cuda::HostMem::AllocType::SHARED);

  // Filters in CUDA are created one time
  Filters filters = {
    gaussian : cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1,
                                              cv::Size(7, 7), -1),
    sobelx : cv::cuda::createSobelFilter(CV_8UC1, CV_32FC1, 1, 0, 3, 1),
    sobely : cv::cuda::createSobelFilter(CV_8UC1, CV_32FC1, 0, 1, 3, 1)
  };

  GpuMemories gpu = {
    input : shared_output.createGpuMatHeader(),
    blurred : cv::cuda::GpuMat(input.size(), CV_8UC1),
    x : cv::cuda::GpuMat(input.size(), CV_32FC1),
    y : cv::cuda::GpuMat(input.size(), CV_32FC1),
    x2 : cv::cuda::GpuMat(input.size(), CV_32FC1),
    y2 : cv::cuda::GpuMat(input.size(), CV_32FC1),
    mag2 : cv::cuda::GpuMat(input.size(), CV_32FC1),
    mag : cv::cuda::GpuMat(input.size(), CV_32FC1),
    output : shared_output.createGpuMatHeader()
  };

  Streams streams;

  // The first call is typically a warmup call so we dont benchmark
  sobel(filters, streams, gpu);

  int N = 100;
  double time = cv::getTickCount();

  std::cout << "Performing " << N << " iterations..." << std::flush;

  for (int i = 0; i < N; i++) {
    sobel(filters, streams, gpu);
  }

  time = 1000.0*(cv::getTickCount() - time)/cv::getTickFrequency();
  time /= N;

  std::cout << " done!" << std::endl << "Average for " << N << " CPU runs: "
            << time << "ms" << std::endl;

  std::cout << "Resulting image wrote to " << to_write << std::endl;
  cv::imwrite(to_write, shared_output);

  return 0;
}
