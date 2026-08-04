# Retained Checkpoints

The application resolves checkpoints below `app/checkpoints` by default. Each path can
instead be overridden with the environment variable shown in the table. The SHA-256
digests identify the exact locally retained files; they do not imply that the weights
have already been publicly deposited.

| System | Component | Default relative path | Environment variable | Bytes | SHA-256 |
|---|---|---|---|---:|---|
| AK | direct | `ak/direct.pt` | `CRYOEM_AK_DIRECT_CHECKPOINT` | 121695112 | `549468ca2bc2ac929fa416621eac2fbb67bb3f39aa65f6626fed7c245e12c1aa` |
| AK | staged encoder | `ak/staged_encoder.pt` | `CRYOEM_AK_STAGED_ENCODER_CHECKPOINT` | 13381143 | `edf8bdb61d312382ac8da439a2864c22243a17d38df567c6bfc97efce268cc29` |
| AK | staged decoder | `ak/staged_decoder.pt` | `CRYOEM_AK_STAGED_DECODER_CHECKPOINT` | 66734070 | `cabf27067c5926550c0f885c39be7737541acca061e80c344cf839f74b734f25` |
| Synthetic HER2 | direct | `her2_synthetic/direct.pt` | `CRYOEM_HER2_SYNTHETIC_DIRECT_CHECKPOINT` | 117588936 | `fb8fceddf392748652f931d75c73dda670a2b515743e7f04224db2696818310f` |
| Synthetic HER2 | staged encoder | `her2_synthetic/staged_encoder.pt` | `CRYOEM_HER2_SYNTHETIC_STAGED_ENCODER_CHECKPOINT` | 25964183 | `b35b1a3e0868ac9941150329be1ea45c293516cfbac3460f1b204ace6c54d113` |
| Synthetic HER2 | staged decoder | `her2_synthetic/staged_decoder.pt` | `CRYOEM_HER2_SYNTHETIC_STAGED_DECODER_CHECKPOINT` | 65073270 | `709891cdc379ce5aaa20ef430689cba95ec2deaa077a0f6260d22c5d22ce800b` |
| Experimental HER2 | direct | `her2_experimental/direct.pt` | `CRYOEM_HER2_EXPERIMENTAL_DIRECT_CHECKPOINT` | 117588936 | `3d527837fff2cb4932e57d1f7a2dfcd37e1ec783285028016b6f02cfc51ed076` |
| Experimental HER2 | staged encoder | `her2_experimental/staged_encoder.pt` | `CRYOEM_HER2_EXPERIMENTAL_STAGED_ENCODER_CHECKPOINT` | 15072673 | `4f18aa3106f4deac3d161d922fb651e665b7a4adb18363c205af465721044da7` |
| Experimental HER2 | staged decoder | `her2_experimental/staged_decoder.pt` | `CRYOEM_HER2_EXPERIMENTAL_STAGED_DECODER_CHECKPOINT` | 34868730 | `88f99a2974a73c8de471d0460954e76543fde060640ab47d229593ee7972d37d` |

Set a common checkpoint root with:

```bash
export CRYOEM_APP_CHECKPOINT_DIR=/path/to/checkpoints
```

Verify an installed file with `sha256sum`. PyTorch checkpoint loading can execute
pickle payloads; use only the listed retained files or another trusted source.

Optional example archives must contain `identifiers`, `images`, and `targets` arrays.
Experimental staged examples may additionally contain `staged_targets` because the
retained direct and staged models use different delivered coordinate frames.
