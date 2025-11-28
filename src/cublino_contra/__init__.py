from gymnasium.envs.registration import register, registry

if "CublinoContra-v0" not in registry:
    register(
        id="CublinoContra-v0",
        entry_point="src.cublino_contra.env:CublinoContraEnv",
    )
