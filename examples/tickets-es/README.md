# tickets-es: SYNTHETIC Spanish e-invoicing support tickets

**Every ticket in this folder is synthetic.** We wrote them for this repo. None comes from a customer, and any RFC that appears is a generic SAT placeholder (`XAXX010101000`, `XEXX010101000`, `EKU9003173C9`).

- `synthetic-train.jsonl`: 300 tickets. `synthetic-test.jsonl`: 100 tickets.
- Labels (see `question.json`): `cfdi_rechazado`, `complemento_pago`, `cancelacion`, `certificado_csd`, `acceso_cuenta`, `facturacion_general`, `escalar_humano`.
- How they were made: 30 hand-written base phrasings per label. 22 go to train and 8 to test, split **by base phrasing**, so no test ticket shares its core sentence with a training ticket. Variants add greetings, sign-offs, misleading sentences borrowed from other labels, dropped accents, lower-casing, typos and chat abbreviations (`xq`, `x fa`, `q`).
- Some bases are confusable on purpose. Cancelling a payment complement is `cancelacion`. A CSD error while cancelling is `certificado_csd`. "Cancel my subscription" is `escalar_humano`. Granting someone permission to stamp but not to cancel is `acceso_cuenta`.
- Regenerate with `go run ./examples/gen` (seed 20261122).
- `rules.json` is a keyword baseline written by the dataset author, who knew every phrasing. It is optimistic.

Do not read results on this set as performance on real tickets.
