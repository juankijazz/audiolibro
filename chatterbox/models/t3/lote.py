# Inferencia de T3 por lotes (varios trozos de texto a la vez) para aprovechar la GPU.
# Mismo muestreo que T3.inference (CFG + penalizacion de repeticion + temperatura + min_p/top_p), pero:
#  - varios textos en paralelo (relleno por la izquierda + mascara de atencion),
#  - sin sincronizaciones GPU<->CPU en cada paso (solo cada 8 pasos se mira si todo ha acabado),
#  - sin ordenaciones ni pasadas por el backend de HuggingFace,
#  - tope de tokens por fila segun la longitud de su texto (corta alucinaciones largas).
import torch
from transformers import DynamicCache


def _muestrear(logits, vistos, temperatura, rep_pen, min_p, top_p):
    """logits (B, V) ya combinados con CFG. Devuelve (B,) tokens."""
    if rep_pen != 1.0:
        pen = torch.where(logits < 0, logits * rep_pen, logits / rep_pen)
        logits = torch.where(vistos, pen, logits)
    if temperatura != 1.0:
        logits = logits / temperatura
    if min_p and min_p > 0.0:
        p = torch.softmax(logits, dim=-1)
        corte = min_p * p.max(dim=-1, keepdim=True).values
        logits = logits.masked_fill(p < corte, float("-inf"))
    if top_p < 1.0:
        orden, idx = torch.sort(logits, descending=True, dim=-1)
        acum = torch.softmax(orden, dim=-1).cumsum(dim=-1)
        quitar = acum - torch.softmax(orden, dim=-1) > top_p
        quitar = torch.zeros_like(quitar).scatter(1, idx, quitar)
        logits = logits.masked_fill(quitar, float("-inf"))
    return torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(1)


@torch.inference_mode()
def inferencia_lote(t3, t3_cond, textos_tokens, limites, *, temperature=0.8, top_p=1.0, min_p=0.05,
                    repetition_penalty=1.2, cfg_weight=0.5, comprobar_cada=8):
    """
    textos_tokens: lista de B tensores 1D (ya con start/stop de texto).
    limites: lista de B enteros, maximo de tokens de voz por fila.
    Devuelve (tokens, completos): lista de B tensores 1D (sin el EOS) y lista de B bools
    (False si la fila se corto por el tope sin llegar a EOS).
    """
    hp = t3.hp
    dev = t3.device
    B = len(textos_tokens)
    usar_cfg = cfg_weight > 0.0
    nb = 2 if usar_cfg else 1
    eos, bos = hp.stop_speech_token, hp.start_speech_token

    cond_emb = t3.prepare_conditioning(t3_cond)                       # (1, Lc, D)
    bos_t = torch.tensor([[bos]], dtype=torch.long, device=dev)
    bos_emb = t3.speech_emb(bos_t) + t3.speech_pos_emb.get_fixed_embedding(0)   # (1, 1, D)

    secuencias = []
    for tk in textos_tokens:
        tk = tk.to(dev).view(1, -1)
        pos = t3.text_pos_emb(tk)
        te = t3.text_emb(tk) + pos
        secuencias.append(torch.cat([cond_emb[0], te[0], bos_emb[0]], dim=0))
    if usar_cfg:
        for tk in textos_tokens:
            tk = tk.to(dev).view(1, -1)
            te = torch.zeros(1, tk.size(1), cond_emb.size(-1), device=dev, dtype=cond_emb.dtype) \
                + t3.text_pos_emb(tk)
            secuencias.append(torch.cat([cond_emb[0], te[0], bos_emb[0]], dim=0))

    lens = torch.tensor([s.size(0) for s in secuencias], device=dev)
    Lmax = int(lens.max())
    D = cond_emb.size(-1)
    filas = len(secuencias)
    E = torch.zeros(filas, Lmax, D, device=dev, dtype=cond_emb.dtype)
    for r, s in enumerate(secuencias):
        E[r, Lmax - s.size(0):] = s                                    # relleno por la izquierda
    pasos_max = int(max(limites))
    mascara = torch.ones(filas, Lmax + pasos_max + 1, device=dev, dtype=torch.long)
    cols = torch.arange(Lmax, device=dev).unsqueeze(0)
    mascara[:, :Lmax] = (cols >= (Lmax - lens).unsqueeze(1)).long()
    pos_ids = (mascara[:, :Lmax].cumsum(dim=1) - 1).clamp(min=0)

    cache = DynamicCache()
    out = t3.tfmr(inputs_embeds=E, attention_mask=mascara[:, :Lmax], position_ids=pos_ids,
                  past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    h = out.last_hidden_state[:, -1]                                   # (filas, D)

    lim = torch.tensor(limites, device=dev)
    hecho = torch.zeros(B, dtype=torch.bool, device=dev)
    salida = torch.full((B, pasos_max), eos, dtype=torch.long, device=dev)
    vistos = torch.zeros(B, t3.speech_head.out_features, dtype=torch.bool, device=dev)
    vistos[:, bos] = True
    longitud = lens[:B].clone()                                        # tokens reales de la fila (para RoPE)
    pasos = 0
    for i in range(pasos_max):
        lg = t3.speech_head(h)                                         # (filas, V)
        if usar_cfg:
            c, u = lg[:B], lg[B:]
            lg = c + cfg_weight * (c - u)
        sig = _muestrear(lg, vistos, temperature, repetition_penalty, min_p, top_p)
        sig = torch.where(hecho, torch.full_like(sig, eos), sig)
        salida[:, i] = sig
        pasos = i + 1
        hecho = hecho | (sig == eos) | (lim <= i + 1)
        vistos.scatter_(1, sig.unsqueeze(1), True)
        if i + 1 == pasos_max:
            break
        if (i + 1) % comprobar_cada == 0 and bool(hecho.all()):
            break
        ne = t3.speech_emb(sig.unsqueeze(1)) + t3.speech_pos_emb.get_fixed_embedding(i + 1)   # (B, 1, D)
        if usar_cfg:
            ne = torch.cat([ne, ne], dim=0)
        pid = (torch.cat([longitud, longitud]) if usar_cfg else longitud).unsqueeze(1) + i
        out = t3.tfmr(inputs_embeds=ne, attention_mask=mascara[:, :Lmax + i + 1], position_ids=pid,
                      past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        h = out.last_hidden_state[:, -1]

    res = salida[:, :pasos].cpu()
    tokens, completos = [], []
    for b in range(B):
        fila = res[b, :int(limites[b])]
        donde = (fila == eos).nonzero()
        if len(donde):
            tokens.append(fila[: int(donde[0])])
            completos.append(True)
        else:
            tokens.append(fila)
            completos.append(False)
    return tokens, completos
