// must run `python layernorm.py` first to generate the reference data
// then compile for example as `gcc layernorm.c -o layernorm -lm`
// and then run as `./layernorm` to see the output

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void layernorm_forward(float *out, float *mean, float *rstd, float *inp,
                       float *weight, float *bias, int B, int T, int C) {
  float eps = 1e-5f;
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      // seek to the input position inp[b,t,:]
      float *x = inp + b * T * C + t * C;
      // calculate the mean
      float m = 0.0f;
      for (int i = 0; i < C; i++) {
        m += x[i];
      }
      m = m / C;
      // calculate the variance (without any bias correction)
      float v = 0.0f;
      for (int i = 0; i < C; i++) {
        float xshift = x[i] - m;
        v += xshift * xshift;
      }
      v = v / C;
      // calculate the rstd
      float s = 1.0f / sqrtf(v + eps);
      // seek to the output position in out[b,t,:]
      float *out_bt = out + b * T * C + t * C;
      for (int i = 0; i < C; i++) {
        float n = (s * (x[i] - m));        // normalized output
        float o = n * weight[i] + bias[i]; // scale and shift it
        out_bt[i] = o;                     // write
      }
      // cache the mean and rstd for the backward pass later
      mean[b * T + t] = m;
      rstd[b * T + t] = s;
    }
  }
}


// poor man's tensor checker
int check_tensor(float *a, float *b, int n, char *label) {
  int ok = 1;
  printf("%s\n", label);
  for (int i = 0; i < n; i++) {
    if (fabs(a[i] - b[i]) <= 1e-5) {
      printf("OK ");
    } else {
      printf("NOT OK ");
      ok = 0;
    }
    printf("%f %f\n", a[i], b[i]);
  }
  return ok;
}

int main() {

  int B = 2; // batch
  int T = 3; // time / sequence length
  int C = 4; // number of channels

  float *x = (float *)malloc(B * T * C * sizeof(float));
  float *w = (float *)malloc(C * sizeof(float));
  float *b = (float *)malloc(C * sizeof(float));
  float *out = (float *)malloc(B * T * C * sizeof(float));
  float *mean = (float *)malloc(B * T * sizeof(float));
  float *rstd = (float *)malloc(B * T * sizeof(float));

  // read reference information from Python
  FILE *file = fopen("ln.bin", "rb");
  if (file == NULL) {
    printf("Error opening file\n");
    return 1;
  }
  fread(x, sizeof(float), B * T * C, file);
  fread(w, sizeof(float), C, file);
  fread(b, sizeof(float), C, file);
  fread(out, sizeof(float), B * T * C, file);
  fread(mean, sizeof(float), B * T, file);
  fread(rstd, sizeof(float), B * T, file);
  fclose(file);

  // now let's calculate everything ourselves

  // forward pass
  float *c_out = (float *)malloc(B * T * C * sizeof(float));
  float *c_mean = (float *)malloc(B * T * sizeof(float));
  float *c_rstd = (float *)malloc(B * T * sizeof(float));
  layernorm_forward(c_out, c_mean, c_rstd, x, w, b, B, T, C);

  // check correctness of forward pass
  check_tensor(out, c_out, B * T * C, "out");
  check_tensor(mean, c_mean, B * T, "mean");
  check_tensor(rstd, c_rstd, B * T, "rstd");

  free(x);
  free(w);
  free(b);
  free(out);
  free(mean);
  free(rstd);
  return 0;
}
