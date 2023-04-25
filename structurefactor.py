from polymer import Config, Polymer

fnames_cfg = ("/storage/janmak98/masterthesis/ouput/mesh/configs/sys_800beeds_400.00lbox_0.00100mu.toml",
              "/storage/janmak98/masterthesis/ouput/mesh2/configs/sys_96beeds_48.00lbox_0.00100mu.toml")

for fname_cfg in fnames_cfg:
    cfg = Config.from_toml(fname_cfg)

    p = Polymer(cfg)
    p.load_traj_gro(overwrite=True)

    Q, S_q = p.get_structure_factor_rdf(overwrite=True)

