package local

import (
	"math"

	"github.com/numoru-ia/agent-decision-gate/internal/textnorm"
)

// Features extracts sparse counts from text after accent folding:
// word unigrams, word bigrams and character n-grams (3..5) inside
// space-padded words. Character n-grams make the model robust to typos
// and dropped accents, which is most of real support text.
func Features(text string) map[string]float64 {
	words := textnorm.Words(text)
	f := make(map[string]float64, len(words)*8)
	for i, w := range words {
		f["w:"+w]++
		if i > 0 {
			f["b:"+words[i-1]+"_"+w]++
		}
		p := []rune(" " + w + " ")
		for n := 3; n <= 5; n++ {
			for j := 0; j+n <= len(p); j++ {
				f["c:"+string(p[j:j+n])]++
			}
		}
	}
	return f
}

// sublinearL2 applies 1+log(tf) and L2 normalisation, the usual input for
// a linear model over n-grams.
func sublinearL2(f map[string]float64) map[string]float64 {
	out := make(map[string]float64, len(f))
	norm := 0.0
	for k, v := range f {
		x := 1 + math.Log(v)
		out[k] = x
		norm += x * x
	}
	if norm == 0 {
		return out
	}
	norm = math.Sqrt(norm)
	for k := range out {
		out[k] /= norm
	}
	return out
}
