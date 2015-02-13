RBM: Restricted Boltzmann Machine
=================================

An implementation of an RBM in Golang. Trains via contrastive divergence and
generates via Gibbs sampling.

Example usage:
--------------

```go
package main

import (
  "fmt"
  "os"
  "strings"

  "github.com/aotimme/rbm"
  "github.com/petar/GoMNIST"
)

func main() {
  fmt.Println("Loading data...")
  train, _, err := GoMNIST.Load("./data")
  if err != nil {
    panic(err)
  }
  fmt.Println("Converting data...")
  vs := make([][]int, len(train.Images))
  for i := 0; i < len(vs); i++ {
    img := train.Images[i]
    vs[i] = make([]int, len(img))
    for j := 0; j < len(vs[i]); j++ {
      if img[j] >= 128 {
        vs[i][j] = 1
      }
    }
  }
  // 500 hidden units, T = 25 for contrastive divergence
  mach := rbm.NewRBM(len(train.Images[0]), 500, 25, nil)
  fmt.Println("Training RBM...")
  mach.Train(vs, 50000, true)
  f, err := os.Create("generated.txt")
  if err != nil {
    panic(err)
  }
  fmt.Println("Generating digits...")
  for i := 0; i < 10; i++ {
    v := mach.GenerateVisible(100000)
    str := make([]string, len(v))
    for j, vj := range v {
      str[j] = fmt.Sprintf("%d", vj * 255)
    }
    f.WriteString(fmt.Sprintf("%s\n", strings.Join(str, ",")))
  }
}
```

with the data directory from the repo [GoMNIST](https://github.com/petar/GoMNIST)
in the current directory. Running the script will take quite a while and
will generate 100 digits via Gibbs sampling from the trained RBM. A simple
script to plot the digits in `R` to verify they look reasonable is:

```R
PlotImage <- function(row) {
  m <- matrix(row, byrow=T, nrow=28)
  m <- t(m)[,nrow(m):1]
  image(m, col=gray((0:255)/255))
}
digits <- as.matrix(read.csv("generated.txt", header=F))
# plot any digit
PlotImage(digits[1,])
```
