# Remaining download access and UrbanScene3D integration

OpenFly's [Hugging Face repository](https://huggingface.co/datasets/IPEC-COMMUNITY/OpenFly_DataGen)
requires the account holder to accept sharing contact information and the dataset
conditions. Log in there and request/accept access. Create a read token at
https://huggingface.co/settings/tokens with access to this gated dataset.

In an interactive Spark terminal, run:

```sh
python3 /home/iamyanbo/uav-photo-map/configure_hf_access.py
```

Paste the token only into its hidden terminal prompt, not into chat. The helper
verifies the account and an actual gated archive HEAD request before saving a
new token with mode 0600 at the Hugging Face cache location. Existing credentials
are never overwritten. The acquisition tool reads this location automatically.
Accepting terms in the browser alone does not authenticate Spark downloads.

UrbanScene3D's completed archive has SHA-256
`0d6316a6d1089c291c11cd21a24aa3f781fd6973115f5c78f93cc677d5d58cd6`.
It expands to 82,111,029,797 bytes. Inspection found:

- `UrbanScene.uproject` names Unreal Engine 4.27 and runtime module `shiyan`.
- Declared target platforms are `MacNoEditor` and `WindowsNoEditor`.
- There are no packaged binaries and no bundled `Plugins` directory.
- The archive contains 20 map files, including virtual cities and real scans;
  map files/variants are not automatically independent environments.

The next step is to integrate a compatible AirSim plugin, build the project and
package the chosen maps for a supported platform. A packaged Linux x86_64 build
could then be checked with the existing Box64 setup; compatibility on Spark's
ARM64 host remains unverified. This is a build/integration issue, not an incomplete
download or another access gate. Unreal tooling/source availability and disk
requirements must be checked before a large build. No build has been attempted.

The [official dataset terms](https://vcc.tech/UrbanScene3D) restrict use to
noncommercial work and prohibit dataset redistribution. Keep downloaded assets
outside the source repository.
