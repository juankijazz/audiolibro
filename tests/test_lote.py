"""Pruebas de la inferencia por lotes de T3 con un modelo diminuto de pesos aleatorios (solo CPU)."""
import copy
import sys
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

try:
    import torch
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.llama_configs import LLAMA_520M_CONFIG_DICT, LLAMA_CONFIGS
    from chatterbox.models.t3.lote import inferencia_lote
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    from chatterbox.models.t3.modules.t3_config import T3Config
    HAY_TORCH = True
except Exception:          # sin torch/transformers (entorno minimo): se omiten
    HAY_TORCH = False


@unittest.skipUnless(HAY_TORCH, "requiere torch y el paquete chatterbox")
class LoteT3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cfg = copy.deepcopy(LLAMA_520M_CONFIG_DICT)
        cfg.update(intermediate_size=128, num_hidden_layers=2,
                   )
        LLAMA_CONFIGS["tiny"] = cfg
        hp = T3Config.multilingual()
        hp.llama_config_name = "tiny"
        cls.t3 = T3(hp).eval()
        cls.hp = hp
        cls.cond = T3Cond(speaker_emb=torch.randn(1, 256),
                          cond_prompt_speech_tokens=torch.randint(0, 6000, (1, 150)),
                          emotion_adv=0.5 * torch.ones(1, 1, 1))

    def texto(self, n):
        return torch.tensor([self.hp.start_text_token] + list(range(10, 10 + n)) + [self.hp.stop_text_token])

    def test_relleno_no_cambia_el_resultado(self):
        """Una fila sola y la misma fila rellenada junto a otras mas largas dan los mismos tokens."""
        kw = dict(temperature=1e-4, min_p=0.0, repetition_penalty=1.0, cfg_weight=0.5)
        a, _ = inferencia_lote(self.t3, self.cond, [self.texto(5)], [12], **kw)
        b, _ = inferencia_lote(self.t3, self.cond, [self.texto(5), self.texto(11), self.texto(8)], [12, 12, 12], **kw)
        self.assertEqual(a[0].tolist(), b[0].tolist())
        self.assertEqual(len(a[0]), 12)

    def test_coincide_con_la_inferencia_original(self):
        torch.manual_seed(1)
        tk = self.texto(7)
        orig = self.t3.inference(t3_cond=self.cond, text_tokens=torch.stack([tk, tk]), max_new_tokens=10,
                                 temperature=1e-4, cfg_weight=0.5, repetition_penalty=1.0, min_p=0.0,
                                 top_p=1.0)[0].tolist()
        nuevo, _ = inferencia_lote(self.t3, self.cond, [tk], [10], temperature=1e-4, min_p=0.0,
                                   repetition_penalty=1.0, cfg_weight=0.5)
        n = min(len(orig), len(nuevo[0]))
        self.assertGreaterEqual(n, 1)
        self.assertEqual(orig[:n], nuevo[0].tolist()[:n])

    def test_tope_por_fila_y_sin_cfg(self):
        toks, completos = inferencia_lote(self.t3, self.cond, [self.texto(4), self.texto(9)], [5, 9],
                                          temperature=1.0, cfg_weight=0.0)
        self.assertLessEqual(len(toks[0]), 5)
        self.assertLessEqual(len(toks[1]), 9)
        self.assertEqual(len(completos), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
