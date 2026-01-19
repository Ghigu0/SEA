#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>

void compute_ahash64_batch(const uint8_t* frames_rgb_dev, int frames, uint64_t* out_hash64_dev);

static void fill_test_frames(std::vector<uint8_t>& h, int frames) {
  // Pattern semplice: frame0 = gradiente, frame1 = gradiente + quadrato bianco
  const int W=1280, H=720;
  for (int f=0; f<frames; f++) {
    uint8_t* img = h.data() + (size_t)f*W*H*3;
    for (int y=0;y<H;y++) for (int x=0;x<W;x++) {
      uint8_t v = (uint8_t)((x*255)/ (W-1));
      img[(y*W+x)*3+0] = v;
      img[(y*W+x)*3+1] = v;
      img[(y*W+x)*3+2] = v;
    }
    if (f==1) {
      // quadrato bianco
      for (int y=200;y<400;y++) for (int x=400;x<700;x++) {
        img[(y*W+x)*3+0]=255;
        img[(y*W+x)*3+1]=255;
        img[(y*W+x)*3+2]=255;
      }
    }
  }
}

int main() {
  const int frames = 2;
  const size_t bytes = (size_t)frames * 1280ull * 720ull * 3ull;

  std::vector<uint8_t> h_frames(bytes);
  fill_test_frames(h_frames, frames);

  uint8_t* d_frames = nullptr;
  uint64_t* d_hashes = nullptr;

  cudaMalloc(&d_frames, bytes);
  cudaMalloc(&d_hashes, (size_t)frames * sizeof(uint64_t));

  cudaMemcpy(d_frames, h_frames.data(), bytes, cudaMemcpyHostToDevice);

  compute_ahash64_batch(d_frames, frames, d_hashes);

  std::vector<uint64_t> h_hashes(frames);
  cudaMemcpy(h_hashes.data(), d_hashes, (size_t)frames*sizeof(uint64_t), cudaMemcpyDeviceToHost);

  for (int i=0;i<frames;i++) {
    printf("frame %d hash = 0x%016llx\n", i, (unsigned long long)h_hashes[i]);
  }

  cudaFree(d_frames);
  cudaFree(d_hashes);
  return 0;
}
