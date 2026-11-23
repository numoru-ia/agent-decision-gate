package textnorm

import (
	"reflect"
	"testing"
)

func TestFold(t *testing.T) {
	cases := map[string]string{
		"  Cancelación   del CFDI ": "cancelacion del cfdi",
		"AÑO Pingüino":              "ano pinguino",
		"":                          "",
		"línea\n\tnueva":            "linea nueva",
	}
	for in, want := range cases {
		if got := Fold(in); got != want {
			t.Errorf("Fold(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestWords(t *testing.T) {
	got := Words("¿Error CFDI40147? sí, ¡ya!")
	want := []string{"error", "cfdi40147", "si", "ya"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Words = %v, want %v", got, want)
	}
}
