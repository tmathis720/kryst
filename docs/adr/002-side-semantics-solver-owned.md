# ADR-002: Solver owns left/right semantics

- PC always computes `z = M^{-1} x`.
- Left GMRES: Arnoldi on `M^{-1}A`, update `x += V y`, monitor `||M^{-1}r||`.
- Right GMRES: Arnoldi on `AM^{-1}`, store `Z`, update `x += Z y`, monitor `||r||`.
- CG/MINRES: Left only (SPD M). Error if not Left.

