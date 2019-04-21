# Change Log

## [0.1.3] - 2019.04.21

### Added

- Add function `get_random_rotation_matrix` to `rotation` module, for generating (stacks of) matrices representing random 3D rotations.

### Fixed

- Fix problem with `rotation.rotmat2axang` where passing a single identity matrix raises a `ValueError`, due to a vectorisation issue.

## [0.1.2] - 2019.04.21

### Fixed

- Fix problem with `rotation.vecpair2rotmat` introduced by previous bug fix, whereby the returned matrices for non-anti-parallel vector pairs are incorrect.

## [0.1.1] - 2019.04.18

### Fixed

- Fix problem with `rotation.vecpair2rotmat` where if a single, anti-parallel vector pair is passed, the return is a matrix of `nan`.
