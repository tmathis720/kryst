use criterion::{Criterion, criterion_group, criterion_main};
use kryst::matrix::dist_csr::{
    DistLocalKernelStrategy, DistributedPlanMetrics, HaloOverlapMode, choose_distributed_plan,
};
use kryst::matrix::spmv::SpmvKernel;

fn comm_heavy_metrics() -> DistributedPlanMetrics {
    DistributedPlanMetrics {
        n_local_rows: 8_192,
        local_nnz: 220_000,
        local_diag_nnz: 90_000,
        ghost_nnz: 130_000,
        local_only_rows: 1_350,
        border_rows: 6_842,
        halo_recv_volume: 28_000,
        halo_send_volume: 24_000,
    }
}

fn compute_heavy_metrics() -> DistributedPlanMetrics {
    DistributedPlanMetrics {
        n_local_rows: 8_192,
        local_nnz: 220_000,
        local_diag_nnz: 211_000,
        ghost_nnz: 9_000,
        local_only_rows: 7_420,
        border_rows: 772,
        halo_recv_volume: 1_400,
        halo_send_volume: 1_350,
    }
}

fn bench_distributed_planner(c: &mut Criterion) {
    let comm_heavy = comm_heavy_metrics();
    let compute_heavy = compute_heavy_metrics();

    let comm_diag = choose_distributed_plan(&comm_heavy, Some(SpmvKernel::Scalar));
    assert_eq!(comm_diag.overlap_mode, HaloOverlapMode::Interior);
    assert_eq!(
        comm_diag.kernel_strategy,
        DistLocalKernelStrategy::RowSplitScalar
    );

    let compute_diag = choose_distributed_plan(&compute_heavy, Some(SpmvKernel::Scalar));
    assert_eq!(compute_diag.overlap_mode, HaloOverlapMode::Disabled);
    assert_eq!(
        compute_diag.kernel_strategy,
        DistLocalKernelStrategy::LocalDiagSpmvPlan
    );

    let mut group = c.benchmark_group("dist_spmv_plan_selection");
    group.bench_function("communication_heavy", |b| {
        b.iter(|| choose_distributed_plan(&comm_heavy, Some(SpmvKernel::Scalar)));
    });
    group.bench_function("compute_heavy", |b| {
        b.iter(|| choose_distributed_plan(&compute_heavy, Some(SpmvKernel::Scalar)));
    });
    group.finish();
}

criterion_group!(benches, bench_distributed_planner);
criterion_main!(benches);
