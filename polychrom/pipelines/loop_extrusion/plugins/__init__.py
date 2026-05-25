"""Default plugin implementations for the loop-extrusion pipeline.

Each plugin is a plain callable. Users can override any of them in the YAML
config by pointing ``plugins.<slot>.target`` at their own ``module:attr``.
"""
