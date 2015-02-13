package rbm

import (
  "fmt"
  "math"
  "math/rand"
)

func uniform(r *rand.Rand) float64 {
  if r == nil {
    return rand.Float64()
  } else {
    return r.Float64()
  }
}
func expit(x float64) float64 {
  return 1.0 / (1.0 + math.Exp(-x))
}
func bernoulli(r *rand.Rand, p float64) int {
  if uniform(r) < p {
    return 1
  } else {
    return 0
  }
}

type RBM struct {
  d int           // visible units
  m int           // hidden units
  w [][]float64   // connection weights (m x d)
  a []float64     // visible unit biases
  b []float64     // hidden unit biases
  cdt int         // number of contrastive divergence samples
  r *rand.Rand
}

func NewRBM(numVisible, numHidden, cdt int, r *rand.Rand) (self *RBM) {
  self = new(RBM)
  self.d, self.m, self.cdt = numVisible, numHidden, cdt
  self.a = make([]float64, self.d)
  self.b = make([]float64, self.m)
  self.w = make([][]float64, self.d)
  for i := 0; i < self.d; i++ {
    self.w[i] = make([]float64, self.m)
  }
  self.r = r
  return
}

func (self *RBM) GetHiddenProbability(j int, v []int) float64 {
  x := self.b[j]
  for i := 0; i < self.d; i++ {
    x += self.w[i][j] * float64(v[i])
  }
  return expit(x)
}
func (self *RBM) GetVisibleProbability(i int, h []int) float64 {
  x := self.a[i]
  for j := 0; j < self.m; j++ {
    x += self.w[i][j] * float64(h[j])
  }
  return expit(x)
}

func (self *RBM) SampleHiddenUnit(j int, v []int) int {
  p := self.GetHiddenProbability(j, v)
  return bernoulli(self.r, p)
}
func (self *RBM) SampleVisibleUnit(i int, h []int) int {
  p := self.GetVisibleProbability(i, h)
  return bernoulli(self.r, p)
}

func (self *RBM) SampleHiddenLayer(v []int) (h []int) {
  h = make([]int, self.m)
  for j := 0; j < self.m; j++ {
    h[j] = self.SampleHiddenUnit(j, v)
  }
  return
}
func (self *RBM) SampleVisibleLayer(h []int) (v []int) {
  v = make([]int, self.d)
  for i := 0; i < self.d; i++ {
    v[i] = self.SampleVisibleUnit(i, h)
  }
  return
}

func (self *RBM) SampleModel(v []int) (vs, hs [][]int) {
  h1 := self.SampleHiddenLayer(v)
  vs = make([][]int, self.cdt)
  hs = make([][]int, self.cdt)
  vs[0] = self.SampleVisibleLayer(h1)
  hs[0] = self.SampleHiddenLayer(vs[0])
  for t := 1; t < self.cdt; t++ {
    vs[t] = self.SampleVisibleLayer(hs[t - 1])
    hs[t] = self.SampleHiddenLayer(vs[t])
  }
  return
}

func (self *RBM) HiddenUnitExpectation(j int, v []int) float64 {
  return self.GetHiddenProbability(j, v)
}

func (self *RBM) HiddenLayerExpectation(v []int) []float64 {
  ps := make([]float64, self.m)
  for j := 0; j < self.m; j++ {
    ps[j] = self.HiddenUnitExpectation(j, v)
  }
  return ps
}

func (self *RBM) GradientStep(v []int) {
  // TODO: allow using multipel data points at each iteration?
  hExp := self.HiddenLayerExpectation(v)
  vSamples, hSamples := self.SampleModel(v)
  epsilon := 0.05
  // visible unit bias gradient step
  for i := 0; i < self.d; i++ {
    vModelExp := 0.0
    for t := 0; t < self.cdt; t++ {
      vModelExp += float64(vSamples[t][i])
    }
    vModelExp /= float64(self.cdt)
    self.a[i] += epsilon * (float64(v[i]) - vModelExp)
  }
  // hidden unit bias gradient step
  for j := 0; j < self.m; j++ {
    hModelExp := 0.0
    for t := 0; t < self.cdt; t++ {
      hModelExp += float64(hSamples[t][j])
    }
    hModelExp /= float64(self.cdt)
    self.b[j] += epsilon * (hExp[j] - hModelExp)
  }
  // connection weights gradient step
  for i := 0; i < self.d; i++ {
    for j := 0; j < self.m; j++ {
      dataExp := float64(v[i]) * hExp[j]
      modelExp := 0.0
      for t := 0; t < self.cdt; t++ {
        modelExp += float64(vSamples[t][i]) * float64(hSamples[t][j])
      }
      modelExp /= float64(self.cdt)
      self.w[i][j] += epsilon * (dataExp - modelExp)
    }
  }
}

func (self *RBM) Train(v [][]int, iters int, verbose bool) {
  N := len(v)
  for it := 0; it < iters; it++ {
    if verbose && (it + 1) % 1000 == 0 {
      fmt.Printf("Training iteration: %d\n", it + 1)
    }
    n := int(uniform(self.r) * float64(N))
    vn := v[n]
    self.GradientStep(vn)
  }
}

func (self *RBM) GenerateVisible(iters int) []int {
  v := make([]int, self.d)
  for i := 0; i < self.d; i++ {
    v[i] = bernoulli(self.r, 0.5)
  }
  var h []int
  for t := 0; t < iters; t++ {
    h = self.SampleHiddenLayer(v)
    v = self.SampleVisibleLayer(h)
  }
  return v
}
